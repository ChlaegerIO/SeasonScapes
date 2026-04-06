# -*- coding: utf-8 -*-

import time
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel    # Install or use own controlNet
# from diffusers import StableDiffusion3InpaintPipeline
# from diffusers import AutoPipelineForInpainting
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from data_engine.utils.general import getHomePath

class inpaintControlNet():
    def __init__(self, config, torch_dtype=torch.float16):
        self.home_path = getHomePath()
        self.ipAdapter = True if config.ip_adapter_image_path else False
        controlnet_list = []
        for cnet_unit in config.controlnet_units:
            if Path(cnet_unit['controlnet_key']).suffix == '.ckpt' or Path(cnet_unit['controlnet_key']).suffix == '.safetensors':
                # load from .ckpt file
                controlnet = ControlNetModel.from_single_file(cnet_unit['controlnet_key'], torch_dtype=torch_dtype)
            else:
                controlnet = ControlNetModel.from_pretrained(cnet_unit['controlnet_key'], torch_dtype=torch_dtype)      # load here own controlNet
            controlnet_list.append(controlnet)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
                                                                 torch_dtype=torch_dtype).to("cuda")
        # pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
        # pipe = StableDiffusion3InpaintPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
        if config.ip_adapter_image_path:
            try:
                ip_path = Path(self.home_path, config.ip_adapter_image_path)
                ip_img = Image.open(ip_path.as_posix())
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
                self.ipAdapter = True
            except Exception as e:
                self.ipAdapter = False
                print(f"\nWARNING: IP adapter image not found, Error {e}")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.sd_model_key, subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"xformers not installed, using default attention, Error {e}")
        pipe.enable_model_cpu_offload()
        self.pipe = pipe


    def inference(self, config):
        """
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # input
        image = Image.open(config.image_path)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        mask = Image.open(config.mask_path)
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # mask = self.pipe.mask_processor.blur(mask, blur_factor=1)      # blurr mask -> also blurs the generated image
        image = self.fill_image(image, mask)

        # condition
        control_img = []
        conditioning_scales = []
        for cnet_unit in config.controlnet_units:
            if cnet_unit['preprocessor'] == 'none':
                condition_image = Image.open(cnet_unit['condition_image_path'])
                condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
            elif cnet_unit['preprocessor'] == 'inpaint_global_harmonious':
                condition_image = self.make_inpaint_condition(image, mask)
            else:
                raise NotImplementedError
            control_img.append(condition_image)
            conditioning_scales.append(cnet_unit['weight'])
        conditioning_scales = conditioning_scales[0] if len(conditioning_scales) == 1 else conditioning_scales

        # ip-adapter
        ip_adapter_image = None
        if config.ip_adapter_image_path and self.ipAdapter:
            ip_adapter_image = Image.open(config.ip_adapter_image_path)
            print("\nusing ip adapter...")
        
        seed = int(time.time()) if config.seed == -1 else config.seed
        generator = torch.manual_seed(int(seed))
        if self.ipAdapter:
            res_image = self.pipe(config.prompt,
                                negative_prompt=config.negative_prompt,
                                image=image,
                                mask_image=mask,
                                control_image=control_img,
                                ip_adapter_image=ip_adapter_image,
                                height=h,
                                width=w,
                                num_images_per_prompt=config.num_images_per_prompt,
                                guidance_scale=config.guidance_scale,
                                num_inference_steps=config.num_inference_steps,
                                strength=config.denoising_strength,
                                generator=generator,
                                controlnet_conditioning_scale=conditioning_scales).images
        else:
            res_image = self.pipe(config.prompt,
                                negative_prompt=config.negative_prompt,
                                image=image,
                                mask_image=mask,
                                control_image=control_img,
                                height=h,
                                width=w,
                                num_images_per_prompt=config.num_images_per_prompt,
                                guidance_scale=config.guidance_scale,
                                num_inference_steps=config.num_inference_steps,
                                strength=config.denoising_strength,
                                generator=generator,
                                controlnet_conditioning_scale=conditioning_scales).images
        
        # res_image = self.pipe(
        #     prompt=self.prompt,
        #     image=image,
        #     mask_image=mask,
        #     guidance_scale=8.0,
        #     num_inference_steps=20,  # steps between 15 and 30 work well for us
        #     strength=0.99,  # make sure to use `strength` below 1.0
        #     generator=generator,
        #     height=h,
        #     width=w,
        # ).images[0]
        # res_image = self.pipe(
        #     prompt=config.prompt,
        #     negative_prompt=config.negative_prompt,
        #     image=image,
        #     mask_image=mask,
        #     num_inference_steps=28, # SD3 often works well with fewer steps (e.g., 28-50)
        #     guidance_scale=7.0,    # Adjust as needed
        #     generator=generator,
        #     height=h,
        #     width=w,
        #     ip_adapter_image=ip_adapter_image,

        #     # SD3 might have other specific parameters like `joint_attention_kwargs`, check docs
        # ).images[0]
        return res_image


    def fill_image(self, image, image_mask, inpaintRadius=3):
        image = np.array(image.convert("RGB"))
        image_mask = (np.array(image_mask.convert("L"))).astype(np.uint8)
        filled_image = cv2.inpaint(image, image_mask, inpaintRadius, cv2.INPAINT_TELEA)

        res_img = Image.fromarray(np.clip(filled_image, 0, 255).astype(np.uint8))
        return res_img


    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image