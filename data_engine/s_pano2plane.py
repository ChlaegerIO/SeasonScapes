import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse

from data_engine.utils.config import *
from data_engine.data_pipeline import Data_pipeline
from data_engine.utils.general import getHomePath

HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
# either date/time like '2024-07-12/12-00-00' and WHOLE_DAY = False 
# or date like '2024-05-12' and WHOLE_DAY = True
days = ['2024-12-18', '2025-01-06', '2025-02-28',
         '2025-03-18', '2025-04-29', '2025-05-18', '2025-06-13']
FROM_TIME = "10-00-00"  # time to start processing
TO_TIME = "16-00-00"    # time to stop processing

# parse command line arguments
argparser = argparse.ArgumentParser(description="Pano2Plane")
argparser.add_argument("--days", type=str, nargs='+', default=days, help="List of days to process in format YYYY-MM-DD")
argparser.add_argument("--from_time", type=str, default=FROM_TIME, help="Time to start processing")
argparser.add_argument("--to_time", type=str, default=TO_TIME, help="Time to stop processing")
args = argparser.parse_args()

home_path = getHomePath()
config_path = Path(home_path, f"configs/EarthEngine/Scale{SCALE}_{HASH}.json")
configCloud_path = Path(home_path, "configs/SatCloud_Kachelmann_superHD.json")
cam_path = Path(home_path, "configs/Cam_-8512.json")
sd_config_path = Path(home_path, "configs/sd15.json")

config = Config()
config.loadGEE(config_path)
config.loadCam(cam_path)
config.loadSatCloud(configCloud_path)
config.loadSD(sd_config_path)


data_pipeline = Data_pipeline(config)

for time_entry in tqdm(args.days, desc="Processing days"):
    # Process whole day - generate time slots from from_time to to_time
    date_part = time_entry
    from_hour, from_min, _ = map(int, args.from_time.split('-'))
    to_hour, to_min, _ = map(int, args.to_time.split('-'))
    
    # Generate time slots every 10 minutes from from_time to to_time
    current_time = datetime.strptime(f"{date_part} {from_hour:02d}:{from_min:02d}:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(f"{date_part} {to_hour:02d}:{to_min:02d}:00", "%Y-%m-%d %H:%M:%S")
    
    while current_time <= end_time:
        time_str = current_time.strftime("%H-%M-%S")
        
        # Special case for 26-MUER camera with 15min interval
        # Check if this time slot should be processed for MUER camera
        if current_time.minute == 10 or current_time.minute == 40:
            # MUER uses +5 minutes offset for these times
            muer_time = current_time + timedelta(minutes=5)
            muer_time_str = muer_time.strftime("%H-%M-%S")
            muer_time_folder = Path(home_path, "data", date_part, muer_time_str)
            if Path(muer_time_folder, '360').is_dir():
                scene_path = Path(home_path, "data", date_part, muer_time_str)
                data_pipeline.generate_planar_images(scene_path=scene_path)
        
        time_folder = Path(home_path, "data", date_part, time_str)
        
        # check if its a time folder with '360' already
        if Path(time_folder, '360').is_dir():
            scene_path = Path(home_path, "data", date_part, time_str)
            data_pipeline.generate_planar_images(scene_path=scene_path)
        else:
            print(f"Folder {time_folder} does not contain a 360 folder.")
        
        print(f"Processed {scene_path}")
        
        # Move to next 10-minute slot
        current_time += timedelta(minutes=10)

