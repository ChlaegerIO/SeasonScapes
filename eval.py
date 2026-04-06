import sys
import os
import argparse
from pathlib import Path

controlnet_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "controlNet_engine"))
if controlnet_engine_dir not in sys.path:
    sys.path.insert(0, controlnet_engine_dir)

from paint_engine.cfg import PaintConfig
from eval_engine.eval_pipeline import Eval_pipeline

argparser = argparse.ArgumentParser(description="Paint Engine Evaluation")
argparser.add_argument("--config_file", type=str, default="configs/paint/paint_config_eval.yaml", help="Path to the config file")
argparser.add_argument("--eval_path", type=str, default="eval_engine/eval_data", help="Path to the evaluation data")
args = argparser.parse_args()

# read config file as command line argument
config_file = args.config_file
eval_path = args.eval_path


paint_cfg = PaintConfig()
paint_cfg.update_from_yaml(config_file)

print("\n", config_file, "\n")
print("\n", paint_cfg.dataset, "\n")
print("\n", paint_cfg.render, "\n")
print("\n", paint_cfg.diffusion, "\n")
print("\n", paint_cfg.optim, "\n")
print("\n", paint_cfg.log, "\n")


eval = Eval_pipeline(eval_path, paint_cfg, trigger_word="image")
# eval.evaluate()

# TODO: separate them into different scripts
# eval.evaluate_folder()
eval.evaluate_inpaint_folder(with_eval_mask=True)   # for inpainted views