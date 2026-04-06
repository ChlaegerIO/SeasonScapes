import requests
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

def download_latest_image(metadata, save_path, time_out=6):
    '''
    Download the latest image from the URL and save it to the specified path
    
    Parameters:
    metadata (dataFrame): A dataFrame containing the metadata of the camera
    save_path (str): The path to save the image
    '''
    # Send a HTTP GET request to the URL
    for id in tqdm(metadata.keys()):
        url = metadata[id]['Download Link']
        response = requests.get(url, timeout=time_out)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the final URL after redirections
            final_url = response.url

            date = final_url.split("/")[-2]
            time_st = final_url.split("/")[-1]
            save_path_all = Path(save_path, date, time_st, id + ".jpg")
            # Ensure the save directory exists
            if not save_path_all.parent.exists():
                save_path_all.parent.mkdir(parents=True)

            # Save the image
            with open(save_path_all, "wb") as file:
                file.write(response.content)
            print(f"Image downloaded " + id)
        else:
            print(f"Failed to download {id}. Status code: {response.status_code}")


def download_specific_images(metadata, from_date, to_date, save_parentPath, 
                             time_out=6, sleep_time=5, retries=1, fails2abort=6, scene_name=""):
    '''
    Download the image from the URL on a specific date and save it to the specified path
    
    Parameters:
    metadata (dataFrame): Dataframe containing the metadata of the camera
    date (str): The start date in format "%Y-%m-%d:%H-%M-%S" to download the image
    to_date (str): The end date in format "%Y-%m-%d:%H-%M-%S" to download the image
    save_parentPath (str): The path to save the image
    time_out (int): The time out for the request
    retries (int): The number of retries if the request was not successful
    fails2abort (int): The number of fails to abort the download
    '''
    to_date = datetime.strptime(to_date, "%Y-%m-%d-%H-%M-%S")
    missing_data = []
    
    # Send a HTTP GET request to the URL
    for id in tqdm(metadata.keys()):
        url = metadata[id]['Download Link']
        try:
            response = requests.get(url, timeout=time_out)
        except Exception as exc:
            print(f"Failed first connection to {id}: {url}. Error: {exc}")
            missing_data.append((id, url))
            continue          
        cnt = 0

        # wait and retry if the request was not successful
        while response.status_code != 200 and cnt < retries:
            time.sleep(sleep_time)
            response = requests.get(url, timeout=time_out)
            cnt += 1
            print(f"Failed first connection to {id}: {url}. Status code: {response.status_code}, {cnt}")

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the final URL after redirections
            final_url = response.url

            url_woDate_list = final_url.split("/")[:-3]
            url_woDate = "/".join(url_woDate_list)
            date = datetime.strptime(from_date, "%Y-%m-%d-%H-%M-%S")

            cnt_fails = 0
            # loop through the dates
            while date <= to_date:
                day = date.strftime("%Y-%m-%d")
                time_st = date.strftime("%H-%M-%S")

                save_path = Path(save_parentPath, day, time_st, "360")
                if not save_path.exists():
                    save_path.mkdir(parents=True)

                new_url = url_woDate + "/" + day + "/" + time_st + "/" + day + "-" + time_st + "_full.jpg"
                try:
                    response = requests.get(new_url, timeout=time_out)
                except Exception as exc:
                    missing_data.append((id, new_url))
                    cnt_fails += 1
                    time.sleep(sleep_time)      # helps if something fails to recover
                    if cnt_fails > fails2abort:
                        missing_data.append((id, url))
                        print(f"Too many missing of {id}: {new_url}. Abort Status code: {response.status_code}. Error: {exc}")
                        break
                    print(f"No response from {id}: {new_url}. Status code: {response.status_code}. Error: {exc}")
                    continue

                # wait and retry if the request was not successful
                cnt = 0
                while response.status_code != 200 and cnt < retries:
                    time.sleep(sleep_time)
                    response = requests.get(new_url, timeout=time_out)
                    cnt += 1
                    cnt_fails += 1
                    print(f"Failed new_url connection with {id}: {new_url}. Status code: {response.status_code}, {cnt}")

                # Check if the request was successful
                if response.status_code == 200:
                    # Save the image
                    file_path = Path(save_path, f'{id}.jpg')
                    with open(file_path.as_posix(), "wb") as file:
                        file.write(response.content)
                    cnt_fails = 0
                else:
                    missing_data.append((id, new_url))

                # increment the date
                if id == "26-MUER":
                    date = date + timedelta(minutes=15)
                else:
                    date = date + timedelta(minutes=10)
    
    print("Downloaded images from " + str(from_date) + " to " + str(to_date) + " finished")
    file_path = Path(save_parentPath, f'{day}/info.txt')
    with open(file_path.as_posix(), "w") as file:
        file.write(scene_name)
        for item in missing_data:
            file.write(f"{item}\n")

def download_missing_images(metadata, from_date, to_date, save_parentPath, 
                            time_out=6, sleep_time=5, scene_name=""):
    '''
    Download the missing images from the URL on a specific date and save it to the specified path
    '''
    day_st = datetime.strptime(from_date, "%Y-%m-%d-%H-%M-%S").strftime("%Y-%m-%d")
    date = datetime.strptime(from_date, "%Y-%m-%d-%H-%M-%S")
    date_end = datetime.strptime(to_date, "%Y-%m-%d-%H-%M-%S")
    missing_data = []
    new_url = ""

    while date <= date_end:
        print(f"{date}/{date_end}", end='\r')
        for id in metadata.keys():
            time_st = date.strftime("%H-%M-%S")
            day_st = date.strftime("%Y-%m-%d")
            # special case for 26-MUER, 15min interval
            if id == "26-MUER" and (date.minute == 10 or date.minute == 40):
                date_tmp = date + timedelta(minutes=5)
                time_st = date_tmp.strftime("%H-%M-%S")
            elif id == "26-MUER" and (date.minute == 20 or date.minute == 50):
                continue
        
            url = metadata[id]['Download Link']
            save_path = Path(save_parentPath, day_st, time_st, "360")
            if not save_path.exists():
                save_path.mkdir(parents=True)

            file_path = Path(save_path, f'{id}.jpg')
            if file_path.exists():
                print(f"Image {id}:{time_st} already downloaded")
                continue    # skip images that have already been downloaded

            try:
                response = requests.get(url, timeout=time_out)
            except Exception as exc:
                print(f"Failed first connection to {id}: {url}. Error: {exc}")
                missing_data.append((id, url))
                time.sleep(sleep_time)    # helps if something fails to recover
                continue
            
            if response.status_code == 200:
                # Extract the final URL after redirections
                final_url = response.url

                url_woDate_list = final_url.split("/")[:-3]
                url_woDate = "/".join(url_woDate_list)
                new_url = url_woDate + "/" + day_st + "/" + time_st + "/" + day_st + "-" + time_st + "_full.jpg"
                
            try:
                response = requests.get(new_url, timeout=time_out)
            except Exception as exc:
                print(f"No response from {id}: {new_url}. Status code: {response.status_code}. Error: {exc}")
                missing_data.append((id, new_url))
                time.sleep(sleep_time)      # helps if something fails to recover
                continue

            # Check if the request was successful
            if response.status_code == 200:
                with open(file_path.as_posix(), "wb") as file:
                    file.write(response.content)
            else:
                missing_data.append((id, new_url))
            
        # increment the date
        date = date + timedelta(minutes=10)
        day_old = day_st
        day_new = date.strftime("%Y-%m-%d")

        if day_old != day_new:
            print(f"Download for {day_old} finished")
            file_path = Path(save_parentPath, f'{day_old}/info.txt')
            with open(file_path.as_posix(), "w") as file:
                file.write(scene_name)
                for item in missing_data:
                    file.write(f"{item}\n")
            missing_data = []
    
    print("Download finished")
    file_path = Path(save_parentPath, f'{day_st}/info.txt')
    with open(file_path.as_posix(), "w") as file:
        file.write(scene_name)
        for item in missing_data:
            file.write(f"{item}\n")

# TODO: download missing directly with base url --> some adresses got mixed 28, 24, 9, (16)
def download_missing_images_fast(metadata, from_date, to_date, save_parentPath, 
                            time_out=6, sleep_time=5, scene_name="", delete_wrong=True):
    '''
    Download the missing images based on the base URL on a specific date and save it to the specified path
    '''
    day_st = datetime.strptime(from_date, "%Y-%m-%d-%H-%M-%S").strftime("%Y-%m-%d")
    date = datetime.strptime(from_date, "%Y-%m-%d-%H-%M-%S")
    date_end = datetime.strptime(to_date, "%Y-%m-%d-%H-%M-%S")
    missing_data = []
    print(f"Start downloading missing images from {day_st} to {date_end.strftime('%Y-%m-%d')} with args:")
    print(f"delete_wrong: {delete_wrong}, time_out: {time_out}, sleep_time: {sleep_time}, scene_name: {scene_name}")

    while date <= date_end:
        print(f"{date}/{date_end}", end='\r')

        time_st = date.strftime("%H-%M-%S")
        day_st = date.strftime("%Y-%m-%d")
        save_path = Path(save_parentPath, day_st, time_st, "360")

        # delete wrong downloaded images
        filenames_to_delete = ["28-SCHILTBIR.jpg", "24-JUNGFJ.jpg", "9-MELCHBONI.jpg", "16-GRINDBEL.jpg", "23-GRINDMJOCH.jpg"]
        for filename in filenames_to_delete:
            file_path = Path(save_path, filename)
            if file_path.exists() and delete_wrong:
                try:
                    file_path.unlink()
                    print(f"Deleted {filename}")
                except Exception as e:
                    print(f"An error while deleting {filename}: {e}")

        for id in metadata.keys():
            time_st = date.strftime("%H-%M-%S")
            day_st = date.strftime("%Y-%m-%d")
            # special case for 26-MUER, 15min interval
            if id == "26-MUER" and (date.minute == 10 or date.minute == 40):
                date_tmp = date + timedelta(minutes=5)
                time_st = date_tmp.strftime("%H-%M-%S")
            elif id == "26-MUER" and (date.minute == 20 or date.minute == 50):
                continue
        
            url_base = metadata[id]['Download base']
            url = url_base + day_st + "/" + time_st + "/" + day_st + "-" + time_st + "_full.jpg"
            save_path = Path(save_parentPath, day_st, time_st, "360")
            if not save_path.exists():
                save_path.mkdir(parents=True)

            file_path = Path(save_path, f'{id}.jpg')
            if file_path.exists():
                print(f"Image {id}:{time_st} already downloaded")
                continue    # skip images that have already been downloaded

            try:
                response = requests.get(url, timeout=time_out)
            except Exception as exc:
                print(f"Failed first connection to {id}: {url}. Error: {exc}")
                missing_data.append((id, url, "no connection"))
                time.sleep(sleep_time)    # helps if something fails to recover
                continue

            # Check if the request was successful
            if response.status_code == 200:
                with open(file_path.as_posix(), "wb") as file:
                    file.write(response.content)
            else:
                print(f"No response from {id}: {url}. Status code: {response.status_code}")
                missing_data.append((id, url, response.status_code))
            
        # increment the date
        date = date + timedelta(minutes=10)
        day_old = day_st
        day_new = date.strftime("%Y-%m-%d")

        if day_old != day_new:
            print(f"Download for {day_old} finished")
            file_path = Path(save_parentPath, f'{day_old}/info.txt')
            with open(file_path.as_posix(), "w") as file:
                file.write(scene_name)
                for item in missing_data:
                    file.write(f"{item}\n")
            missing_data = []
    
    print("Download finished")
    file_path = Path(save_parentPath, f'{day_st}/info.txt')
    with open(file_path.as_posix(), "w") as file:
        file.write(scene_name)
        for item in missing_data:
            file.write(f"{item}\n")