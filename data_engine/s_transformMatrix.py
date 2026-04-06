from utils.config import Config
from pathlib import Path

from utils.general import *
from utils.camFormat import *

######################## CONFIG ########################

# OpenGL coordinate system: x-right (East), y-up (Altitude), z-backward (South)
# Rotation of camera from looking North is encoded in the transformation matrix
CONFIG_PATH = 'configs/Livecams-Masterarbeit-Timo-Kleger.csv'
DATA_PATH = 'data/2024-10-09/12-00-00/transformation_matrices'
FILE_NAME = 'transforms_train_Rshot.json'

WHICH_TASK = 'transformsGPS2pix'   # 'gen_transformsGPS_imgSplit', 'transformsGPS2pix'


######################## PROGRAM ########################

# load config
config = Config()
config.setData_pathMetadata(CONFIG_PATH)
config.setData_pathScene(DATA_PATH)

save_path = Path(DATA_PATH, FILE_NAME)

######################## TASKS ########################

if WHICH_TASK == 'gen_transformsGPS_imgSplit':
  gen_transformsGPS_imgSplit(config)
elif WHICH_TASK == 'transformsGPS2pix':
  transformsGPS2pix(config, save_path)
