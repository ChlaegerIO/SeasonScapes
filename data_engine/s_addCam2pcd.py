from pathlib import Path
import argparse

from utils.config import *
from dataVisu_engine.ThreeD_helpers import loadPointCloud_ply, savePointCloud_ply, showPointCloud, addCam2Pcd_Rshot, openTFMatrix


HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
CONFIG_PATH = f"configs/EarthEngine/Scale{SCALE}_{HASH}.json"
PLY_PATH = f"data/EarthEngine/pointcloud_{HASH}.ply"
PLY_SAVE_PATH = f"data/EarthEngine/pointcloud_{HASH}_WiCam.ply"
TRANSFORM_MATRIX_PATH = "data/transformation_matrices/Rshot/transformation_final_v4_flat.json"

argparser = argparse.ArgumentParser(description="Add camera positions to point cloud")
argparser.add_argument("--config_path", type=str, default=CONFIG_PATH, help="Path to the config file")
argparser.add_argument("--ply_path", type=str, default=PLY_PATH, help="Path to the PLY file")
argparser.add_argument("--ply_save_path", type=str, default=PLY_SAVE_PATH, help="Path to save the modified PLY file")
argparser.add_argument("--transforms_path", type=str, default=TRANSFORM_MATRIX_PATH, help="Path to the transformation matrices")
argparser.add_argument("--gps_coords", type=bool, default=False, help="GPS coordinates in transform matrix (True) or pixel coordinates (False)")
args = argparser.parse_args()

print(f"args: {args}")

# paths
home_path = getHomePath()
ply_path = Path(home_path, args.ply_path)
ply_save_path = Path(home_path, args.ply_save_path)
transforms_path = Path(home_path, args.transforms_path)
config_fPath = Path(home_path, args.config_path)

config = Config()
config.loadGEE(config_fPath)
transformMatrices = openTFMatrix(transforms_path)

# processing the ply file
pcd = loadPointCloud_ply(ply_path.parent, ply_path.name)
addCam2Pcd_Rshot(config, pcd, transformMatrices, gps_coords=args.gps_coords)
showPointCloud(pcd)
savePointCloud_ply(pcd, ply_save_path.parent, fName=ply_save_path.name)