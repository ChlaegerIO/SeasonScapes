from pathlib import Path

from data_engine.utils.general import getHomePath
from data_engine.data_pipeline import Data_pipeline
from data_engine.utils.config import Config

home_path = getHomePath()
gee_config_path = Path(home_path, "configs/EarthEngine/Scale60_-8512.json")
sat_config_path = Path(home_path, "configs/SatCloud_Kachelmann_superHD.json")
sd_config_path = Path(home_path, "configs/sd15_genImg_rgbd.json")
cam_config_path = Path(home_path, "configs/Cam_-8512.json")
folders = ['2024-05-11', '2024-05-12', '2024-07-12', '2024-07-20', '2024-07-25', '2024-09-01', '2024-09-19', '2024-10-12', '2024-10-17', '2024-12-18', '2025-01-20', '2025-01-22']  
RGBD = True
PROMPT_TYPE = "short"  # "short", "medium", "long" (, None)
# partly done: '2024-07-12' until 16:40, 2024-09-01 until 15:50, 2024-10-07 until 13:10
# Done: '2024-05-11', '2024-05-12', 2024-09-19

config = Config()
config.loadGEE(gee_config_path)
config.loadCam(cam_config_path)
config.loadSatCloud(sat_config_path)
config.loadSD(sd_config_path)
data = Data_pipeline(config)

# data.generate_SD_samples(rgbd=RGBD)

data.generate_controlNet_training_data(folders=folders, skip_sampleMesh=True, prompt_type=PROMPT_TYPE)
