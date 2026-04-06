from pathlib import Path
import json
import csv
import re

def getHomePath() -> Path:
    '''
    Get the home path of the project from root directory
    '''
    script_path = Path(__file__).parent
    home_path = Path(*script_path.parts[:-2])
    return home_path

def openTFMatrix(path: str) -> dict:
    '''
    Open a transformation matrix from a json file with relative path
    '''
    home_path = getHomePath()
    open_path = Path(home_path, path)
    with open(open_path, 'r') as f:
        transformMatrix = json.load(f)

    return transformMatrix

def getPointMatches(pointMatch_path: str) -> dict:
    with open(Path(pointMatch_path).as_posix(), 'r') as f:
        pointMatches = json.load(f)

    return pointMatches

def find_images_with_prefix(folder_path: str, prefix: str) -> list:
    # Create a Path object for the given folder
    path = Path(folder_path)
    # Use the glob method to find files that match the prefix
    matching_images = list(path.glob(f"{prefix}*"))
    return matching_images

def natural_sort_key(path_object):
    """
    Creates a sort key for natural sorting of filenames.
    Splits the filename stem into text and numeric parts.
    Example: "wayp_10" -> ("wayp_", 10), 
    Use function as argument in sorted(all_paths, key=natural_sort_key)
    """
    stem = path_object.stem
    # Try to split into parts (text, number) using regex
    parts = re.split(r'(\d+)', stem)
    # Convert numeric parts to integers, keep text parts as strings
    key_parts = []
    for part in parts:
        if part.isdigit():
            key_parts.append(int(part))
        else:
            key_parts.append(part.lower())  # lowercase letters
    key_parts = [p for p in key_parts if p != '']       # Filter out empty strings
    return tuple(key_parts)