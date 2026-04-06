import sys
import argparse
from share import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_RGBA import MyDataset_RGBA
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, ModelLogger
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Configs
MODEL = './models/cldm_v15_rgba.yaml'
RESUME_PATH = './models/control_sd15_depth.pth'
IMG_LOG_FOLDER = 'r1Srgbd_trainSplit'   # # train100, r1SceneDepthTune
MODEL_LOG_FOLDER = 'model_logs/r1Srgbd_trainSplit'
LOGGER_FREQ = 4000    # for model and img logger
BATCH_SIZE = 4      # 4 maximum batch size that fit 16GB (roundshot img 512x768), 10 max batch size for 48GB GPU MEMORY (tutorial: 512x512)
LEARNING_RATE = 1e-5
SD_LOCKED = True
ONLY_MID_CONTROL = False
NUM_WORKERS = 0
ACCUMULATE_GRAD = 1
TRAIN_DIR = 'r1Srgbd_trainSplit'    # fill100, r1SceneDepth, r1SceneDepthMidas

# load argument
parser = argparse.ArgumentParser(description="Train MultiNet")
parser.add_argument('--no_cNetPretraining', type=str2bool, default=False, 
                    help="Set to True to disable cNet pretraining and use 'control_sd15_ini.ckpt'")
parser.add_argument('--log_dir', type=str, default=None)
args = parser.parse_args()

if args.no_cNetPretraining:
    RESUME_PATH = './models/control_sd15_ini.ckpt'
if args.log_dir is not None:
    IMG_LOG_FOLDER = args.log_dir
    MODEL_LOG_FOLDER = Path(Path(MODEL_LOG_FOLDER).parent, args.log_dir).as_posix()

print("train config:")
print("model", MODEL, "resume", RESUME_PATH, "img_log", IMG_LOG_FOLDER, "model_log", MODEL_LOG_FOLDER)
print("batch", BATCH_SIZE, "lr", LEARNING_RATE, "sd_locked", SD_LOCKED, "only_mid_control", ONLY_MID_CONTROL)
print("num_workers", NUM_WORKERS, "accumulate_grad", ACCUMULATE_GRAD, "train_dir", TRAIN_DIR)

def modify_state_dict_for_4_channels(state_dict):
    for key in state_dict.keys():
        if 'input_hint_block' in key and 'weight' in key:
            old_weight = state_dict[key]
            if old_weight.shape[1] == 3:
                new_weight = torch.zeros(old_weight.shape[0], 4, old_weight.shape[2], old_weight.shape[3], 
                                         device=old_weight.device, dtype=old_weight.dtype)
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3, :, :] = old_weight[:, 0, :, :]  # Duplicate one weights channel
                state_dict[key] = new_weight
    return state_dict

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(MODEL).cpu()
state_dict = load_state_dict(RESUME_PATH, location='cpu')

# Modify the state_dict to handle the new input channels
state_dict = modify_state_dict_for_4_channels(state_dict)
model.load_state_dict(state_dict, strict=False)

model.learning_rate = LEARNING_RATE
model.sd_locked = SD_LOCKED
model.only_mid_control = ONLY_MID_CONTROL

# Misc
dataset = MyDataset_RGBA(train_dir=TRAIN_DIR)
dataloader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
logger = ImageLogger(img_folder=IMG_LOG_FOLDER, batch_frequency=LOGGER_FREQ)
logger_model = ModelLogger(log_dir=MODEL_LOG_FOLDER, batch_frequency=LOGGER_FREQ)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, logger_model], accumulate_grad_batches=ACCUMULATE_GRAD)     # accumulate_grad_batches=2 for more internal steps, multi GPU check code: devices=2, strategy='ddp'

# Train!
trainer.fit(model, dataloader)