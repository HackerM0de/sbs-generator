import cv2
import numpy as np
import torch
import torchvision.transforms as T
from MiDaS.midas.model_loader import load_model
from multiprocessing import Process, Queue
from tqdm import tqdm
import time
from numba import njit
import subprocess
import os
import signal

cv2.ocl.setUseOpenCL(False)

# Constants
SCALE_FACTOR = 70
INPUT_VIDEO_PATH = "climb.mp4"
OUTPUT_VIDEO_PATH = "out.mp4"
MODEL_TYPE = "dpt_swin2_tiny_256"
MODEL_PATH = "models/" + MODEL_TYPE + ".pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50
PREVIEW_FRAMES = 200  # Set to an integer like 100 to process only the first 100 frames

# Load model in main process
model, transform, _, _ = load_model(DEVICE, MODEL_PATH, MODEL_TYPE, optimize=True, height=None, square=False)

def get_depth_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = transform({"image": image_rgb})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(input_tensor).to(DEVICE).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

@njit
def shiftPixels(image, depthMap, direction, scale_factor):
    height, width = image.shape[:2]
    shifted = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            depth_val = depthMap[y, x] / 255.0
            shift = int(depth_val * scale_factor * direction)
            new_x = x + shift
            if 0 <= new_x < width:
                shifted[y, new_x] = image[y, x]
    return shifted

@njit
def shift_along_axis(arr, shift, axis):
    result = np.zeros_like(arr)
    if axis == 0:
        if shift > 0:
            result[shift:] = arr[:-shift]
        elif shift < 0:
            result[:shift] = arr[-shift:]
        else:
            result[:] = arr
    elif axis == 1:
        if shift > 0:
            result[:, shift:] = arr[:, :-shift]
        elif shift < 0:
            result[:, :shift] = arr[:, -shift:]
        else:
            result[:] = arr
    return result

@njit
def depth_aware_fill(image, depth_map, max_distance=10, depth_thresh=10):
    filled = image.copy()
    mask = (filled[:, :, 0] == 0) & (filled[:, :, 1] == 0) & (filled[:, :, 2] == 0)
    depth = depth_map.astype(np.float32)
    for axis in [0, 1]:
        for direction in [-1, 1]:
            for i in range(1, max_distance + 1):
                shift = i * direction
                shifted_img = shift_along_axis(filled, shift, axis)
                shifted_depth = shift_along_axis(depth, shift, axis)
                shifted_mask = shift_along_axis(mask, shift, axis)
                for y in range(filled.shape[0]):
                    for x in range(filled.shape[1]):
                        if mask[y, x] and not shifted_mask[y, x] and abs(depth[y, x] - shifted_depth[y, x]) < depth_thresh:
                            for c in range(3):
                                filled[y, x, c] = shifted_img[y, x, c]
                            mask[y, x] = False
    return filled

def final_inpaint(filled_image):
    gray = cv2.cvtColor(filled_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(filled_image, mask, 3, cv2.INPAINT_TELEA)

def shift_and_fill(image, depth_map, direction, scale_factor):
    shifted = shiftPixels(image, depth_map, direction, scale_factor)
    filled = depth_aware_fill(shifted, depth_map)
    return final_inpaint(filled)

def worker(task_queue, output_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, frame, depth = task
        left = shift_and_fill(frame, depth, 1, SCALE_FACTOR)
        right = shift_and_fill(frame, depth, -1, SCALE_FACTOR)
        stacked = np.hstack((left, right))
        output_queue.put((idx, stacked))

def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if PREVIEW_FRAMES is not None:
        total_frames = min(total_frames, PREVIEW_FRAMES)

    ffmpeg = subprocess.Popen([
    'ffmpeg',
    '-hide_banner',
    '-loglevel', 'warning',
    '-y',
    '-f', 'rawvideo',
    '-thread_queue_size', '512',  # <- Add this line
    '-pix_fmt', 'bgr24',
    '-s', f'{width * 2}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-thread_queue_size', '512',  # <- Also for audio input
    '-i', INPUT_VIDEO_PATH,
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '18',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-shortest',
    '-pix_fmt', 'yuv420p',
    OUTPUT_VIDEO_PATH
    ], stdin=subprocess.PIPE)

    task_queue = Queue(maxsize=100)
    output_queue = Queue()

    num_workers = max(2, 7)
    workers = [Process(target=worker, args=(task_queue, output_queue)) for _ in range(num_workers)]
    for w in workers:
        w.start()

    print(f"[Main] Starting MiDaS + Workers with {num_workers} processes...")

    midas_pbar = tqdm(total=total_frames, position=0, desc="MiDaS", dynamic_ncols=True, bar_format="{desc}: {n}/{total} [{rate_fmt}]")
    worker_pbar = tqdm(total=total_frames, position=1, desc="Workers", dynamic_ncols=True, bar_format="{desc}: {n}/{total} [{rate_fmt}]")
    time_pbar = tqdm(total=total_frames, position=2, desc="ETA", dynamic_ncols=True)

    written_frames = 0
    frame_idx = 0
    frame_buffer = {}

    start_time = time.time()

    try:
        while written_frames < total_frames:
            if frame_idx < total_frames:
                if not task_queue.full():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    depth = get_depth_from_image(frame)
                    task_queue.put((frame_idx, frame, depth))
                    frame_idx += 1
                    midas_pbar.update(1)

            while not output_queue.empty():
                idx, result = output_queue.get()
                frame_buffer[idx] = result

            while written_frames in frame_buffer:
                frame = frame_buffer.pop(written_frames)
                ffmpeg.stdin.write(frame.tobytes())
                written_frames += 1
                worker_pbar.update(1)

                elapsed = time.time() - start_time
                fps_current = written_frames / elapsed if elapsed > 0 else 0
                remaining = total_frames - written_frames
                eta = remaining / fps_current if fps_current > 0 else 0
                time_pbar.set_description(f"ETA: {eta:.1f}s | Total: {elapsed:.1f}s")
                time_pbar.n = written_frames
                time_pbar.refresh()
    finally:
        cap.release()

        try:
            if ffmpeg.stdin:
                ffmpeg.stdin.flush()
                ffmpeg.stdin.close()
            os.kill(ffmpeg.pid, signal.SIGINT)
        except Exception as e:
            print(f"[ERROR] ffmpeg closing failed: {e}")

        midas_pbar.close()
        worker_pbar.close()
        time_pbar.close()

        for _ in workers:
            task_queue.put(None)
        for w in workers:
            w.join()

        print("[Main] Processing complete.")

process_video()