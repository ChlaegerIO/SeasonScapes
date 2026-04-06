import os
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.general import getHomePath
from dataVisu_engine.ThreeD_helpers import loadTriMesh_ply, openTFMatrix, sampleMeshAnywhere
from utils.config import Config

HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
RGBD = False
GPS_COORD = True
IMG_FILTER = False
CONFIG_PATH = f"configs/EarthEngine/Scale{SCALE}_{HASH}.json"
MESH_PATH = f"data/EarthEngine/mesh_{HASH}.ply"
TRANSFORM_MATRIX_PATH = "data/transformation_matrices/Rshot/transformation_final_v4_flat.json"
NOVEL_SCENE_PATH = "data/meshRoundshot_lowRes"


home_path = getHomePath()
config_fPath = Path(home_path, CONFIG_PATH)
ply_path = Path(home_path, MESH_PATH)


config = Config()
config.loadGEE(config_fPath)
transformMatrices = openTFMatrix(Path(TRANSFORM_MATRIX_PATH))

# Load mesh and transformation matrices
mesh = loadTriMesh_ply(ply_path.parent, ply_path.name)

# Set configuration and sample mesh
config.setData_pathScene(NOVEL_SCENE_PATH)
sampleMeshAnywhere(config, mesh, transformMatrices, rgbd=RGBD, gps_coord=GPS_COORD, img_filter=IMG_FILTER)