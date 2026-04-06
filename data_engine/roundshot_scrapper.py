import random
import argparse

from utils.config import Config
from utils.getData import *
from utils.general import *

###################################### settings ######################################
SAVE_PATH = "data/"
METADATA_PATH = "configs/Livecams-Masterarbeit-Timo-Kleger.csv"
SCENE = ""

WHICH_TASK = "missing_urls_times"     # latest, specific_times, missing_urls_times, specific_satellite, stateMachine
NBR_RETRIES = 1             # number of retries if the request was not successful
TIMEOUT = 4                 # seconds
SLEEP_TIME = 4              # seconds to sleep between requests
NBR_FAILS2ABORT = 6         # number of fails to abort the download
DELETE_WRONG = False        # delete wrong images

FROM_DATE = "2025-04-29-07-00-00" # 8.1.2025, 9.1.2025, 7.10.24 9-18Uhr (top mit allem), 5.10.24 12-17Uhr (Nebel), 20.7.24 12Uhr, 12.05.24 (nice), 18.05.24, 29.06.24 (Sahara), 12.7.24 (Gewitter), 25.7.24 (Gewitter), 31.7.24 11-20Uhr (Cu bis Cb), 12.8.24 (Cu bis Cb), 31.12.24-1.1.25 (Polarlichter)
TO_DATE = "2025-04-29-18-00-00"   # Done: 11.5.24, 12.5.24, 12.7.24, 20.7.24, 25.7.24, 1.9.24, 19.9.24, 7.10.24, 12.10.24, 17.10.24, 20.10.24, 22.10.24, 24.10.24, 26.10.24, 29.10.2024, 3.11.24, 12.11.24, 7.12.24, 18.12.24, 19.01.2025, 20.1.25, 22.1.25, 23.1.25

DATE_LIST = ["2025-01-22-08-00-00", "2025-01-23-08-00-00"]


###################################### main program ######################################

argparser = argparse.ArgumentParser(description="Roundshot Scrapper")
argparser.add_argument("--task", type=str, default=WHICH_TASK, help="Task to perform: latest, specific_times, missing_urls_times, specific_satellite, stateMachine")
argparser.add_argument("--from_date", type=str, default=FROM_DATE, help="Start date for the download in format YYYY-MM-DD-HH-MM-SS")
argparser.add_argument("--to_date", type=str, default=TO_DATE, help="End date for the download in format YYYY-MM-DD-HH-MM-SS")
argparser.add_argument("--save_path", type=str, default=SAVE_PATH, help="Path to save the downloaded images")
argparser.add_argument("--nbr_retries", type=int, default=NBR_RETRIES, help="Number of retries if the request was not successful")
argparser.add_argument("--timeout", type=int, default=TIMEOUT, help="Timeout in seconds for the requests")
argparser.add_argument("--sleep_time", type=int, default=SLEEP_TIME, help="Sleep time in seconds between requests")
argparser.add_argument("--nbr_fails2abort", type=int, default=NBR_FAILS2ABORT, help="Number of fails to abort the download")
argparser.add_argument("--delete_wrong", type=bool, default=DELETE_WRONG, help="Delete wrong images after download")
args = argparser.parse_args()


if __name__ == "__main__":
    config = Config()
    config.loadCamMetadata(METADATA_PATH)
    metadata = config.data_camOrigMetadata

    # set paths
    home_path = getHomePath()
    save_path = Path(home_path, args.save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if args.task == "latest":
        download_latest_image(metadata, args.save_path)
    elif args.task == "specific_times":
        download_specific_images(metadata, args.from_date, args.to_date, save_path, 
                                 time_out=TIMEOUT, sleep_time=SLEEP_TIME, retries=NBR_RETRIES, 
                                 fails2abort=NBR_FAILS2ABORT, scene_name=SCENE)
        #download_satellite(args.from_date, args.to_date, args.save_path)
    elif args.task == "missing_urls_times":
        download_missing_images_fast(metadata, args.from_date, args.to_date, save_path, 
                                time_out=TIMEOUT, sleep_time=SLEEP_TIME, scene_name=SCENE, delete_wrong=args.delete_wrong)
        #download_satellite(args.from_date, args.to_date, args.save_path)
    elif args.task == "specific_satellite":
        True
        #download_satellite(args.from_date, args.to_date, args.save_path)

    elif args.task == "stateMachine":
        while True:
            from_date = DATE_LIST[random.randint(0, len(DATE_LIST)-1)]
            # shorter to_date for earlier dates
            if int(from_date.split("-")[0]) <= 2024 and int(from_date.split("-")[1]) < 8:
                hour = int(from_date.split("-")[3]) + 4
                to_date = from_date.split("-")[:3] + [str(hour)] + from_date.split("-")[4:]
                to_date = "-".join(to_date)
            else:
                hour = int(from_date.split("-")[3]) + 8
                to_date = from_date.split("-")[:3] + [str(hour)] + from_date.split("-")[4:]
                to_date = "-".join(to_date)
            
            print("Start round from", from_date, "to", to_date)
            download_missing_images_fast(metadata, from_date, to_date, save_path, 
                                    time_out=TIMEOUT, sleep_time=SLEEP_TIME, scene_name=SCENE)
            
            # wait before next round
            time.sleep(60*random.randint(2, 61))  # 2-61 minutes