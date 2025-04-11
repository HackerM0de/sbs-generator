import os, shutil
import cv2
from PIL import Image
from MiDaS import run
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

MODEL_TYPE = "dpt_swin2_large_384"
MODEL_PATH = f"models/{MODEL_TYPE}.pt"

SCALE_FACTOR = 50

MAX_THREADS = os.cpu_count()

def clearDirectory(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path, exist_ok=True)

#clearDirectory(FRAME_DIRECTORY)
#outputFramePath = FRAME_DIRECTORY + "frame%d.jpg"
#outputDir = os.makedirs(FRAME_DIRECTORY, exist_ok=True)
#os.system(f"ffmpeg -i {INPUT} -q:v 4 {outputFramePath}")

#clearDirectory(DEPTH_DIRECTORY)
#run.run(FRAME_DIRECTORY, DEPTH_DIRECTORY, MODEL_PATH, MODEL_TYPE)

clearDirectory(STEREO_DIRECTORY)
clearDirectory(LEFT_EYE_DIRECTORY)
clearDirectory(RIGHT_EYE_DIRECTORY)

def shift_and_fill(image, depth_map, direction, scale_factor, max_distance=10, depth_thresh=10, do_inpaint=True):
    height, width = image.shape[:2]
    normalized_depth = (depth_map.astype(np.float32) / 255.0) - 0.5
    shifts = (normalized_depth * scale_factor * direction).astype(np.int32)

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    new_x_coords = x_coords + shifts
    valid_mask = (new_x_coords >= 0) & (new_x_coords < width)

    shifted = np.zeros_like(image)
    shifted[y_coords[valid_mask], new_x_coords[valid_mask]] = image[y_coords[valid_mask], x_coords[valid_mask]]

    filled = shifted.copy()
    mask = np.all(filled == 0, axis=2)
    depth = depth_map.astype(np.float32)

    for axis in [0, 1]:
        for direction in [-1, 1]:
            for i in range(1, max_distance + 1):
                shift = i * direction
                rolled_img = np.roll(filled, shift=shift, axis=axis)
                rolled_depth = np.roll(depth, shift=shift, axis=axis)
                rolled_mask = np.roll(mask, shift=shift, axis=axis)

                fillable = mask & (~rolled_mask) & (np.abs(depth - rolled_depth) < depth_thresh)

                for c in range(3):
                    filled[..., c][fillable] = rolled_img[..., c][fillable]

                mask[fillable] = False

    if do_inpaint:
        gray = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        filled = cv2.inpaint(filled, final_mask, 3, cv2.INPAINT_TELEA)

    return filled

def process_frame(colourImPath, depthImPath):
    frame_number = colourImPath.split('frame')[1].split('.')[0]
    colourImage = cv2.imread(os.path.join(FRAME_DIRECTORY, colourImPath))
    depthMap = cv2.imread(os.path.join(DEPTH_DIRECTORY, depthImPath), cv2.IMREAD_GRAYSCALE)

    leftEyeImage = shift_and_fill(colourImage, depthMap, 1, SCALE_FACTOR)
    rightEyeImage = shift_and_fill(colourImage, depthMap, -1, SCALE_FACTOR)

    cv2.imwrite(os.path.join(LEFT_EYE_DIRECTORY, f'{frame_number}.jpg'), leftEyeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(os.path.join(RIGHT_EYE_DIRECTORY, f'{frame_number}.jpg'), rightEyeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"Shifted and filled frame {frame_number}.")

def pixelShiftAndFill():
    colourIm = sorted(os.listdir(FRAME_DIRECTORY))
    depthIm = sorted(os.listdir(DEPTH_DIRECTORY))

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(process_frame, colourIm, depthIm)



pixelShiftAndFill()

def create_sbs_video():
    
    # Construct the FFmpeg command to create a side-by-side video
    left_frame_pattern = os.path.join(LEFT_EYE_DIRECTORY, "%d.jpg")
    right_frame_pattern = os.path.join(RIGHT_EYE_DIRECTORY, "%d.jpg")

    # Ensure FFmpeg is properly set up to process both frame sequences
    os.system(f"ffmpeg -y -framerate 30 -i {left_frame_pattern} -framerate 30 -i {right_frame_pattern} "
          f"-i in.mp4 -filter_complex '[0:v][1:v]hstack=inputs=2' -map 2:a -c:v libx264 "
          f"-preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -strict experimental out.mp4")
    
create_sbs_video()