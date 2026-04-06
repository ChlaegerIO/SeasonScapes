from pathlib import Path

from utils.config import *
from dataVisu_engine.ThreeD_helpers import savePointCloud_ply, meshFromPointCloud, saveMesh_ply, genPYLfromRGBdepth

HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
CONFIG_PATH = f"configs/EarthEngine/Scale{SCALE}_{HASH}.json"
RGB_PATH = f"data/EarthEngine/Scale{SCALE}LandswissT10cm_{HASH}.png"
DEPTH_PATH = f"data/EarthEngine/Scale{SCALE}DemNASA30m_{HASH}.png"

SAVE_PCD_PATH = f"data/EarthEngine/pointcloud_{HASH}.ply"
SAVE_MESH_PATH = f"data/EarthEngine/mesh_{HASH}.ply"


home_path = getHomePath()
rgb_path = Path(home_path, RGB_PATH)
depth_path = Path(home_path, DEPTH_PATH)
save_pcd_path = Path(home_path, SAVE_PCD_PATH)
save_mesh_path = Path(home_path, SAVE_MESH_PATH)
config_fPath = Path(home_path, CONFIG_PATH)


config = Config()
config.loadGEE(config_fPath)

pcd = genPYLfromRGBdepth(config, rgb_path, depth_path, show=False)
savePointCloud_ply(pcd, save_pcd_path.parent, fName=save_pcd_path.name)

mesh = meshFromPointCloud(pcd, radius=0.01, max_nn=40, width=4, show=False)
saveMesh_ply(mesh, save_mesh_path.parent, fName=save_mesh_path.name)