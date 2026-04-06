import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pytorch_lightning.callbacks import Callback


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

# logger for model checkpoint
class ModelLogger(Callback):
    def __init__(self, log_dir, batch_frequency=2000):
        self.log_dir = log_dir
        self.batch_freq = batch_frequency

    def on_batch_end(self, trainer, pl_module):
        if pl_module.global_step % self.batch_freq == 0:
            path = os.path.join(self.log_dir, f'model_{pl_module.global_step}.ckpt')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            torch.save(pl_module.state_dict(), path)
            print(f'Saved model at [{path}]')
