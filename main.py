import os, shutil
import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

TEMP_DIRECTORY = "temp/"
FRAME_DIRECTORY = TEMP_DIRECTORY + "frame/"
DEPTH_DIRECTORY = TEMP_DIRECTORY + "depth/"
STEREO_DIRECTORY = TEMP_DIRECTORY + "stereo/"
LEFT_EYE_DIRECTORY = STEREO_DIRECTORY + "left/"
RIGHT_EYE_DIRECTORY = STEREO_DIRECTORY + "right/"
MASKED_DIRECTORY = TEMP_DIRECTORY + "masked/"
LEFT_MASKED_DIRECTORY = MASKED_DIRECTORY + "left/"
RIGHT_MASKED_DIRECTORY = MASKED_DIRECTORY + "right/"
INPUT = "in.mp4"
OUTPUT = "out.mp4"

CACHED_FRAMES = False

MODEL_TYPE = "MiDaS_small"

SCALE_FACTOR = 80

MAX_THREADS = os.cpu_count()

def clearDirectory(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path, exist_ok=True)

def get_depth_map(image: np.ndarray) -> np.ndarray:
    # Convert OpenCV image (BGR) to RGB and normalize to [0, 1]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).to(device)

    # Inference
    with torch.no_grad():
        prediction = midas(input_tensor.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map

if not CACHED_FRAMES:
    clearDirectory(FRAME_DIRECTORY)
    outputFramePath = FRAME_DIRECTORY + "frame%d.jpg"
    outputDir = os.makedirs(FRAME_DIRECTORY, exist_ok=True)
    os.system(f"ffmpeg -i {INPUT} -q:v 4 {outputFramePath}")

    clearDirectory(DEPTH_DIRECTORY)

    midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if MODEL_TYPE == "DPT_Large" or MODEL_TYPE == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

clearDirectory(STEREO_DIRECTORY)
clearDirectory(LEFT_EYE_DIRECTORY)
clearDirectory(RIGHT_EYE_DIRECTORY)

def shiftPixels(image, depthMap, direction, scale_factor):
    height, width = image.shape[:2]
    normalized_depth = (depthMap.astype(np.float32) / 255.0) - 0.4
    shifts = (normalized_depth * scale_factor * direction).astype(np.int32)

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    new_x_coords = x_coords + shifts
    valid_mask = (new_x_coords >= 0) & (new_x_coords < width)

    shifted = np.zeros_like(image)
    shifted[valid_mask] = image[y_coords[valid_mask], x_coords[valid_mask]]
    return shifted

def depth_aware_fill(image, depth_map, max_distance=10, depth_thresh=10):
    filled = image.copy()
    mask = np.all(filled == 0, axis=2)
    depth = depth_map.astype(np.float32)

    for axis in [0, 1]:
        for direction in [-1, 1]:
            for i in range(1, max_distance + 1):
                shifted_img = np.roll(filled, shift=i * direction, axis=axis)
                shifted_depth = np.roll(depth, shift=i * direction, axis=axis)
                shifted_mask = np.roll(mask, shift=i * direction, axis=axis)

                fill_condition = mask & (~shifted_mask) & (np.abs(depth - shifted_depth) < depth_thresh)

                # Vectorized fill per channel
                fill_indices = np.where(fill_condition)
                for c in range(3):
                    filled[..., c][fill_indices] = shifted_img[..., c][fill_indices]

                mask[fill_condition] = False

    return filled

def final_inpaint(filled_image):
    gray = cv2.cvtColor(filled_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(filled_image, mask, 3, cv2.INPAINT_TELEA)

def shift_and_fill(image, depth_map, direction, scale_factor):
    shifted = shiftPixels(image, depth_map, direction, scale_factor)
    filled = depth_aware_fill(shifted, depth_map)
    return final_inpaint(filled)

def process_frame(index, colourIm, depthIm):
    try:
        frame_number = colourIm[index].split('frame')[1].split('.')[0]
        curr_colour = cv2.imread(os.path.join(FRAME_DIRECTORY, colourIm[index]))
        curr_depth = cv2.imread(os.path.join(DEPTH_DIRECTORY, depthIm[index]), cv2.IMREAD_GRAYSCALE)

        leftEyeImage = shift_and_fill(curr_colour, curr_depth, 1, SCALE_FACTOR)
        rightEyeImage = shift_and_fill(curr_colour, curr_depth, -1, SCALE_FACTOR)

        cv2.imwrite(os.path.join(LEFT_EYE_DIRECTORY, f'{frame_number}.jpg'), leftEyeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite(os.path.join(RIGHT_EYE_DIRECTORY, f'{frame_number}.jpg'), rightEyeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"Processed frame {frame_number}.")
    except Exception as e:
        print(f"Error processing frame {index}: {e}")

def pixelShiftAndFill():
    colourIm = sorted(os.listdir(FRAME_DIRECTORY))
    depthIm = sorted(os.listdir(DEPTH_DIRECTORY))
    assert len(colourIm) == len(depthIm), "Mismatch in number of color and depth frames."

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_frame, i, colourIm, depthIm) for i in range(len(colourIm))]
        for future in as_completed(futures):
            future.result()

pixelShiftAndFill()

def create_sbs_video():
    
    # Construct the FFmpeg command to create a side-by-side video
    left_frame_pattern = os.path.join(LEFT_EYE_DIRECTORY, "%d.jpg")
    right_frame_pattern = os.path.join(RIGHT_EYE_DIRECTORY, "%d.jpg")

    # Ensure FFmpeg is properly set up to process both frame sequences
    os.system(f"ffmpeg -y -framerate 60 -i {left_frame_pattern} -framerate 60 -i {right_frame_pattern} "
          f"-i in.mp4 -filter_complex '[0:v][1:v]hstack=inputs=2' -map 2:a -c:v libx264 "
          f"-preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -strict experimental out.mp4")
    
create_sbs_video()