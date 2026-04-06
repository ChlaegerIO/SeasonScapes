from pathlib import Path
import json
import numpy as np

from utils import camFormat
from utils.config import Config
from utils.general import getHomePath, openTFMatrix



if __name__ == "__main__":
    home_path = getHomePath()
    transforms_path = Path(home_path, "data/transformation_matrices/REGION_J/transforms_novelLaubH_wayp_video.json")
    config = Config()
    config.loadGEE(Path(home_path, "configs/EarthEngine/Scale60_-8512_LaubH_v2.json"))

    # Load the transformation matrix
    transformMatrices = openTFMatrix(transforms_path)
    
    # Change gps to pixel coordinates
    for key in transformMatrices.keys():
        if key == 'meta':
            continue
        transformMatrix = np.array(transformMatrices[key]['transform_matrix'])
        t_gps = transformMatrix[:3, 3].tolist()
        t_pix = camFormat.geoCoord2Open3Dpx(config, t_gps)
        transformMatrix[:3, 3] = t_pix

        transformMatrices[key]['transform_matrix'] = transformMatrix.tolist()

    # Save the transformation matrix
    transforms_path = Path(transforms_path.parent, f"{transforms_path.stem}_pix.json")
    with open(transforms_path, 'w') as f:
        json.dump(transformMatrices, f, indent=2, ensure_ascii=False)

