import json
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from utils.camFormat import transform2colmap_camF, transform2colmap_imgF
from utils.general import *

PATH_TRANSFORM_MATRIX = 'data/2024-10-09/12-00-00/Hasliberg'
FILE = 'transformation.json'


tfFile = Path(PATH_TRANSFORM_MATRIX, FILE)
transformFile = openTFMatrix(tfFile)

# Create the cameras.txt file
transform2colmap_camF(transformFile, PATH_TRANSFORM_MATRIX)
transform2colmap_imgF(transformFile, PATH_TRANSFORM_MATRIX)
