from dataVisu_engine.cam_adjustIntrinsic import CamAdjustTool
from utils.general import *
from utils.config import Config

######################## CONFIG ########################
HASH = "-8512"
LIDAR_PATH = "data/EarthEngine"   # Path to the ply file loading
LIDAR_FNAME = f"pointcloud_{HASH}.ply"

CONFIG_PATH = "configs/EarthEngine"  # Path to the config file
CONFIG_FNAME = f"Scale60_{HASH}.json"

CAMERA_PATH = "data/2024-10-09/12-00-00/imagesPlane"   # Path to the camera images

TRANSFORM_MATRIX_FILEP = "data/transformation_matrices/transformation_final_v3.json"

DATA_ID = '19_5-MEIKAES_1'   # see list used for render_w2c_1Cam


######################## PROGRAM ########################

home_path = getHomePath()
lidar_path = Path(home_path, LIDAR_PATH)
lidar_fName = LIDAR_FNAME
camera_path = Path(home_path, CAMERA_PATH)
intrinsics_fileP = Path(home_path, TRANSFORM_MATRIX_FILEP)
config_fPath = Path(home_path, CONFIG_PATH, CONFIG_FNAME)

config = Config()
config.loadGEE(config_fPath)
config.loadCamMetadata(config.data_pathMetadata)


######################## TASKS ########################
matchTool = CamAdjustTool(lidar_path, lidar_fName, camera_path, DATA_ID, intrinsics_fileP, config)
matchTool.run()
