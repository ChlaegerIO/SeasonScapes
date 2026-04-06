import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

def mask_depth_images_in_folder(folder_path, mask_threshold=0.2):
    """Masks all depth images in a folder, setting values > 0 to 255 and others to 0."""
    threshold_save = []
    for filename in tqdm(os.listdir(folder_path)):
        if 'depth' in filename:  # Check if 'depth' is in the filename
            depth_path = os.path.join(folder_path, filename)
            save_fName = filename.replace('depth', 'eval_mask')
            output_path = os.path.join(folder_path, save_fName)
            try:
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # Load as grayscale
                if depth is None:
                    print(f"Warning: Could not read image {filename}. Skipping.")
                    continue
                # img: [0, 255]
                depth_normed = depth.astype(np.float32) / 255.0  # Normalize to [0, 1]
                depth_bins = np.histogram(depth_normed, bins=256)[0]  # Histogram of depth values
                depth_bins = depth_bins[1:]  # Ignore the first bin (0 value)
                depth_bins = depth_bins / np.sum(depth_bins)  # Normalize histogram
                depth_bins = np.cumsum(depth_bins)  # Cumulative sum of histogram
                threshold_index = np.searchsorted(depth_bins, mask_threshold)  # Find threshold index
                threshold_value = threshold_index / 255.0  # Convert index to value
                threshold_save.append({"file": filename, "threshold": threshold_value})  # Save threshold value
                mask = (depth_normed > threshold_value).astype(np.uint8)*255  # Create mask and convert to uint8
                cv2.imwrite(output_path, mask)  # Save the mask
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # print the threshold values for each file
    for item in threshold_save:
        print(f"File: {item['file']}, Threshold: {item['threshold']}")


if __name__ == "__main__":
    folder_path = '...seasonScapes/paint_engine/Logs/GRIND/LaubH_wayp3_v4_250123-1200/train_eval'
    parser = argparse.ArgumentParser(description="Mask depth images in a folder")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing depth images")
    args = parser.parse_args()
    if args.folder_path:
        folder_path = args.folder_path  # Update folder_path if provided in command line

    MASK_THRESHOLD = 0.45  # Set your desired mask threshold here
    mask_depth_images_in_folder(folder_path, mask_threshold=MASK_THRESHOLD)