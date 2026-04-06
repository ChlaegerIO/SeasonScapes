import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3, HWC4
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.midas import MidasDetector


class ControlNet_use():
    def __init__(self, sd_model_path, sd_control_model_path, rgbd=False):
        self.sd_model_path = sd_model_path
        self.sd_control_model_path = sd_control_model_path
        self.rgbd = rgbd

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.model = create_model(sd_model_path).cpu()
        self.model.load_state_dict(load_state_dict(self.sd_control_model_path, location='cuda'), strict=False)
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def run(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, calcDepth):
        with torch.no_grad():
            if self.rgbd:
                input_image = HWC4(input_image)
            else:
                input_image = HWC3(input_image)
            detected_map = resize_image(input_image, image_resolution)
            H, W, C = detected_map.shape
            
            if calcDepth:
                apply_midas = MidasDetector()
                detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
                detected_map = HWC3(detected_map)

                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results