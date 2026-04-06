import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from data_engine.utils.general import getHomePath
from data_engine.data_pipeline import Data_pipeline
from data_engine.utils.config import Config

RGBD = False

home_path = getHomePath()
gee_config_path = Path(home_path, "configs/EarthEngine/Scale60_-8512.json")
sat_config_path = Path(home_path, "configs/SatCloud_Kachelmann_superHD.json")
sd_config_path = Path(home_path, "configs/sd15_genImg_depth.json")
cam_config_path = Path(home_path, "configs/Cam_-8512.json")

config = Config()
config.loadGEE(gee_config_path)
config.loadCam(cam_config_path)
config.loadSatCloud(sat_config_path)
config.loadSD(sd_config_path)
data = Data_pipeline(config)
data()      # call

data.generate_SD_samples(rgbd=RGBD)