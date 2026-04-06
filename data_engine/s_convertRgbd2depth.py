import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

def rgb2depth(folder_path):
    """Converts all RGBD images in a folder to depth images."""
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        output_dir = os.path.join(folder_path, 'tmp_depth')
        save_fName = os.path.join(output_dir, img_path.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)

        try:
            rgbd = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load with all channels including alpha
            if rgbd is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue
            
            # Check if image has 4 channels (RGBA/RGBD)
            if rgbd.ndim != 3 or rgbd.shape[2] != 4:        # !=4 ?
                print(f"Warning: {filename} does not have 4 channels. Shape: {rgbd.shape}. Skipping.")
                continue
            
            # Extract depth channel (4th channel, index 3)
            depth = rgbd[:, :, 3]
            
            # Save depth image
            cv2.imwrite(save_fName, depth)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Conversion completed! Depth images saved in {os.path.join(folder_path, 'tmp_depth')}")


if __name__ == "__main__":
    folder_path = '...data/diffusers_adapted/datasets/rAlldepth_mediumPrompt/source'
    parser = argparse.ArgumentParser(description="Convert rgbd images to depth images in a folder")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing rgbd images")
    args = parser.parse_args()
    if args.folder_path:
        folder_path = args.folder_path  # Update folder_path if provided in command line

    rgb2depth(folder_path)