import sys
import os

import PIL.Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import cv2
import json
from tqdm import tqdm
import torch
import PIL

from llava.llava import Llava_v16


######################## FUNCTION ########################

class llava_prompt():
    '''
    Append LLaVa prompt to prompt.json
    Args: 
        - max_tokens (int): maximum tokens for LLaVA
    '''
    def __init__(self, max_tokens=300):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.llava = Llava_v16(device=self.device, llava_ckpt='llava-hf/llava-v1.6-mistral-7b-hf', max_tokens=max_tokens)
        print("Using LLaVA v1.6...")

    def append_llava_prompt_moveImg(self, target_folder, source_folder, controlNet_folder, 
                                rgbd=True, resize_fac=4, prompt_type=None):
        '''
        Append LLaVa prompt to prompt.json in controlNet_folder
        Args:
            target_folder (Path): folder with target images
            source_folder (Path): folder with source images
            controlNet_folder (Path): folder to save prompt.json
        '''
        if not Path(target_folder).exists():
            print(f"Folder {target_folder} does not exist!")
            return
        if not Path(source_folder).exists():
            print(f"Folder {source_folder} does not exist!")
            return
        
        target_images = [x for x in Path(target_folder).iterdir() if x.is_file()]
        source_images = [x for x in Path(source_folder).iterdir() if x.is_file()]

        # check prompt.json availability
        prompt_file = Path(controlNet_folder, "prompt.json")
        prompt_data = []
        if os.path.getsize(prompt_file) != 0:
            with open(prompt_file.as_posix(), "r") as f:
                for line in f:
                    prompt_data.append(json.loads(line))

        #llava.model.to('cuda')
        tqdm_bar = tqdm(target_images)
        for target in tqdm_bar:
            tqdm_bar.set_description(f"LLaVA {target}")
            # find corresponding source image
            data_id = target.stem.split("_")[1] + "_" + target.stem.split("_")[2]
            mesh_file = None
            depth_file = None
            for source in source_images:
                if data_id in source.stem:
                    if "depth" in source.stem:
                        depth_file = source
                        mesh_file = source.as_posix().replace("_depth", "")
                    else:
                        mesh_file = source
                        depth_file = source.as_posix().replace(".png", "_depth.png")
                    break

            # copy target, combine mesh and depth, rename
            target_img = PIL.Image.open(target)
            mesh_img = PIL.Image.open(mesh_file)
            depth_img = PIL.Image.open(depth_file)
            names = target.as_posix().split("/")
            img_id = names[-1].split(".")[0]
            time = names[-3]
            day = names[-4]
            fName = f"{day}_{time}_{img_id}.png"

            # ask LLaVA for prompt
            response = None
            if prompt_type is None:
                response = self.llava(target_img, mesh_img)
            elif prompt_type == "short":
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Detaily imagine and describe the scene this image taken from in maximally 5 words."},
                            {"type": "image"},
                        ]
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Add style keywords from this nearby image to the previous response. Use maximally 5 words."},
                            {"type": "image"},
                        ]
                    },
                ]
                response = self.llava(target_img, mesh_img, prompt=conversation)
            elif prompt_type == "medium":
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the scene this image taken from in maximally 20 words."},
                            {"type": "image"},
                        ]
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Add style keywords from this nearby image to the previous response. Use maximally 15 words."},
                            {"type": "image"},
                        ]
                    },
                ]
                response = self.llava(target_img, mesh_img, prompt=conversation)
            elif prompt_type == "long":
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the scene this image taken from in maximally 75 words."},
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
                response = self.llava(target_img, mesh_img, prompt=conversation)

            entry = {
                "source": f"source/{fName}",
                "target": f"target/{fName}",
                "prompt": response
            }

            prompt_data.append(entry)

            # resize and save images
            target_img = target_img.convert("RGB")
            new_width = target_img.width // resize_fac
            new_height = target_img.height // resize_fac
            new_size = (new_width, new_height)
            target_img = target_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            depth_img = depth_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            folder = Path(controlNet_folder, "target")
            if not folder.exists():
                folder.mkdir(parents=True)
            target_img.save(Path(controlNet_folder, "target", f'{fName}'))
            if rgbd:
                mesh_img = mesh_img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                mesh_img = mesh_img.convert("RGB")
                depth_img = depth_img.convert("L")
                r, g, b = mesh_img.split()
                rgba = PIL.Image.merge("RGBA", (r, g, b, depth_img))
                folder = Path(controlNet_folder, "source")
                if not folder.exists():
                    folder.mkdir(parents=True)
                rgba.save(Path(controlNet_folder, "source", f'{fName}'))
            else:
                depth_img.save(Path(controlNet_folder, "source", f'{fName}'))

        # save prompt.json
        with open(prompt_file.as_posix(), "w") as f:
            for entry in prompt_data:
                f.write(json.dumps(entry) + "\n")

            print(f"Saved {prompt_file}")

