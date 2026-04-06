from pathlib import Path

from utils.config import *
from dataVisu_engine.ThreeD_helpers import loadPointCloud_ply, savePointCloud_ply, loadTriMesh_ply, saveTriMesh_ply, cutPCDOnAoi, cutMeshOnAoi


PROCESS_PCD = True
PROCESS_MESH = True

AOI_CUT = [7.82, 8.095165, 46.47, 46.8]  # [lonMin, lonMax, latMin, latMax]

HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
CONFIG_PATH = f"configs/EarthEngine/Scale{SCALE}_{HASH}.json"
PCD_PATH = f"data/EarthEngine/pointcloud_{HASH}.ply"
MESH_PATH = f"data/EarthEngine/mesh_{HASH}.ply"

SAVE_PCD_PATH = f"data/EarthEngine/pointcloud_{HASH}_LaubH_v3.ply"
SAVE_MESH_PATH = f"data/EarthEngine/mesh_{HASH}_LaubH_v3.ply"


home_path = getHomePath()
pcd_path = Path(home_path, PCD_PATH)
mesh_path = Path(home_path, MESH_PATH)
save_pcd_path = Path(home_path, SAVE_PCD_PATH)
save_mesh_path = Path(home_path, SAVE_MESH_PATH)
config_fPath = Path(home_path, CONFIG_PATH)

config = Config()
config.loadGEE(config_fPath)


if PROCESS_MESH:
    mesh = loadTriMesh_ply(mesh_path.parent, mesh_path.name)
    mesh = cutMeshOnAoi(config, mesh, AOI_CUT, gps=True)
    saveTriMesh_ply(mesh, save_mesh_path.parent, fName=save_mesh_path.name)

if PROCESS_PCD:
    pcd = loadPointCloud_ply(pcd_path.parent, pcd_path.name)
    pcd = cutPCDOnAoi(config, pcd, AOI_CUT, gps=True)
    savePointCloud_ply(pcd, save_pcd_path.parent, fName=save_pcd_path.name)

config.setGEE_mesh_path(SAVE_MESH_PATH)
config.saveGEE(same_hash=True)