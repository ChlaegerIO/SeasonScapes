from share import *
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_depth import MyDataset_depth
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
MODEL = './models/cldm_v15_depth.yaml'
RESUME_PATH = './models/control_sd15_depth.pth'      # control_sd15_ini.ckpt, 
IMG_LOG_FOLDER = 'r1SDepth_trainSplit'   # # train100, r1SceneDepthTune
BATCH_SIZE = 4      # 4 maximum batch size that fit 16GB (roundshot img 512x768), 10 max batch size for 48GB GPU MEMORY (tutorial: 512x512)
LOGGER_FREQ = 1000
LEARNING_RATE = 1e-5
SD_LOCKED = True
ONLY_MID_CONTROL = True         # only mid control, try out false
NUM_WORKERS = 0
ACCUMULATE_GRAD = 1
TRAIN_DIR = 'r1SDepth_trainSplit'    # fill100, r1SceneDepth, r1SceneDepthMidas

checkpoint_callback = ModelCheckpoint(
    monitor='loss',
    save_last=True,
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(MODEL).cpu()
model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'), strict=False)
model.learning_rate = LEARNING_RATE
model.sd_locked = SD_LOCKED
model.only_mid_control = ONLY_MID_CONTROL


# Misc
dataset = MyDataset_depth(train_dir=TRAIN_DIR)
dataloader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
logger = ImageLogger(batch_frequency=LOGGER_FREQ, img_folder=IMG_LOG_FOLDER)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], accumulate_grad_batches=ACCUMULATE_GRAD)     # accumulate_grad_batches=2 for more internal steps, multi GPU check code: devices=2, strategy='ddp'


# Train!
trainer.fit(model, dataloader)
