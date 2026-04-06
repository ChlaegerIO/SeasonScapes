import sys
import os
import argparse

controlnet_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "controlNet_engine"))
if controlnet_engine_dir not in sys.path:
    sys.path.insert(0, controlnet_engine_dir)
data_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data_engine"))
if data_engine_dir not in sys.path:
    sys.path.insert(0, data_engine_dir)

from data_engine.train_camPose_pipeline import Cam_Optimization

argparser = argparse.ArgumentParser(description="Paint Engine Evaluation")
argparser.add_argument("--transform_matrix_path", type=str, default="data/transformation_matrices/Rshot/transformation_final_v4_flat.json", help="Path to the transformation matrices")   # transformation_2learn_v1_flat, transformation_final_v4_flat 
argparser.add_argument("--gee_config_path", type=str, default="configs/EarthEngine/Scale60_-8512.json", help="Path to the GEE config file")
argparser.add_argument("--log_name", type=str, default="tst", help="Name of the log file")
args = argparser.parse_args()

# read config file as command line argument
transformMatrices_path = args.transform_matrix_path
gee_config_path = args.gee_config_path
log_name = args.log_name

train_camParam = Cam_Optimization(transformMatrices_path, gee_config_path, log_name=log_name)
train_camParam()