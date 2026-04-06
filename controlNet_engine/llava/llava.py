import PIL
import torch
import numpy as np
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class Llava_v15():
    '''
    LLaVA v1.5 model for image captioning using one image
    '''
    def __init__(self,device='cuda',
                 llava_ckpt='llava-hf/bakLlava-v1-hf') -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def __call__(self,image:PIL.Image, prompt=None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
        
        prompt = '<image>\n USER: Detaily imagine and describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ' if prompt is None else prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer
    
class Llava_v16():
    '''
    LLaVA v1.6 model for image captioning using nearby Roundshot image and low-res colored mesh
    '''
    def __init__(self, device='cuda', llava_ckpt='llava-hf/llava-v1.6-mistral-7b-hf', max_tokens=200) -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.max_tokens = max_tokens
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            ).to(self.device)
        
    def __call__(self,image_Rshot:PIL.Image, image_mesh:PIL.Image=None, prompt=None):
        
        # input check
        if not isinstance(image_Rshot,PIL.Image.Image):
            if np.amax(image_Rshot) < 1.1:
                image_Rshot = image_Rshot * 255
            image_Rshot = image_Rshot.astype(np.uint8)
            image_Rshot = PIL.Image.fromarray(image_Rshot)
        if not isinstance(image_mesh,PIL.Image.Image):
            if np.amax(image_mesh) < 1.1:
                image_mesh = image_mesh * 255
            image_mesh = image_mesh.astype(np.uint8)
            image_mesh = PIL.Image.fromarray(image_mesh)

        if prompt is None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detaily imagine and describe the scene this image taken from in maximally 75 words."},
                        {"type": "image"},
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Add style keywords from this nearby image to the previous response. Use maximally 50 words."},
                        {"type": "image"},
                    ]
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            images = (image_mesh, image_Rshot)

        elif len(prompt) == 2:
            prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
            images = (image_mesh, image_Rshot)
        else:
            prompt = self.processor.apply_chat_template(prompt)
            images = image_Rshot

        inputs = self.processor(images=images, text=prompt, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        # split the response
        answer = answer.split("[/INST] ")[-1]
        return answer
