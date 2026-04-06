import sys
import os
import argparse


controlnet_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "controlNet_engine"))
if controlnet_engine_dir not in sys.path:
    sys.path.insert(0, controlnet_engine_dir)

from paint_engine.cfg import PaintConfig
from paint_engine.paint_pipeline import Paint_pipeline

# read config file as command line argument
# configs/paint/paint_config_debug.yaml
argparser = argparse.ArgumentParser(description="Paint Engine")
argparser.add_argument("--config_file", type=str, default="configs/paint/paint_config_wayp_highRes.yaml", help="Path to the config file")
args = argparser.parse_args()

# if len(sys.argv) > 1:
#     config_file = sys.argv[1]
# else:
#     config_file = 'configs/paint/paint_config_wayp_highRes.yaml'

paint_cfg = PaintConfig()
paint_cfg.update_from_yaml(args.config_file)

print("\n", args.config_file, "\n")
print("\n", paint_cfg.dataset, "\n")
print("\n", paint_cfg.render, "\n")
print("\n", paint_cfg.diffusion, "\n")
print("\n", paint_cfg.optim, "\n")
print("\n", paint_cfg.log, "\n")

if 'debug' in args.config_file or 'test' in args.config_file or 'Debug' in args.config_file or 'Test' in args.config_file:
    save_debug=True
else:
    save_debug=False

paint = Paint_pipeline(paint_cfg, save_debug=save_debug)

paint()