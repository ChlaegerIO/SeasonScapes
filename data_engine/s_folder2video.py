import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.general import getHomePath, natural_sort_key
import paint_engine.utils as utils


home_path = getHomePath()
# paint_engine/Logs/LaubH_wayp3_v3/inpaint_video
# data/NovelViews/mesh_novelLaubH_wayp3
img_path = Path(home_path, "paint_engine/Logs/REGION_J_behavior/RegionJ240901-1200_8/train_eval/mask")
save_path = Path(home_path, "paint_engine/Logs/REGION_J_behavior/RegionJ240901-1200_8/train_eval/mask_video.mp4")


all_paths = list(img_path.glob('*.png')) + list(img_path.glob('*.jpg'))
sorted_paths = sorted(all_paths, key=natural_sort_key)

frames = []
for img_path in tqdm(sorted_paths):
    if img_path.suffix == ".png" or img_path.suffix == ".jpg":
        img = utils.pil2tensor(Image.open(img_path), 'cpu')
        frames.append(utils.tensor2numpy(img))

if len(frames) == 0:
    print(f"No images found in {img_path}.")
else:
    utils.save_video(np.stack(frames, axis=0), save_path.as_posix())