from pathlib import Path
import cv2
import json
from tqdm import tqdm
import torch
from einops import rearrange
import PIL

from annotator.midas.api import MiDaSInference
from llava.llava import Llava_v15, Llava_v16

SOURCE_PATH = "r1Srgbd_highRes/source/raw"
TARGET_PATH = "r1Srgbd_highRes/target"

WHICH_TASK = "RGBdepth"   # depth, depth_midas, target_resize, RGBdepth, prompt, prompt_LLaVA_v15, prompt_LLaVA_v16


######################## PROGRAM ########################
home_path = script_path = Path(__file__).parent
training_path = Path(home_path, "training")
source_path = Path(training_path, SOURCE_PATH)
source_parent = source_path.parent
target_parent = Path(training_path, TARGET_PATH)

######################## TASKS ########################

if WHICH_TASK == "depth":
    for file in source_path.iterdir():
        if "depth" in file.stem:
            depth_file = file
            save_file = depth_file.stem.replace("_depth", "")
            d = cv2.imread(str(depth_file))
            d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            d_shape = d.shape
            d_shape = (d_shape[1]//4, d_shape[0]//4)
            d = cv2.resize(d, d_shape, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(source_parent / f"{save_file}.png"), d)
            print(f"Saved {save_file}.png", d.shape)

elif WHICH_TASK == "depth_midas":
    # load midas depth model
    midas_inference = MiDaSInference(model_type="dpt_hybrid")

    # calculate depth images
    for file in source_path.iterdir():
        if "depth" in file.stem:
            continue
        
        # read color image and get depth
        color = cv2.imread(str(file))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        color_shape = color.shape
        color_shape = (color_shape[1]//4, color_shape[0]//4)
        color = cv2.resize(color, color_shape, interpolation=cv2.INTER_AREA)

        # prepare for model
        color = torch.from_numpy(color).float()#.cuda()
        color = color / 127.5 - 1.0
        color = rearrange(color, 'h w c -> 1 c h w')
        mDepth = midas_inference(color)[0]

        # save depth image
        mDepth = mDepth.squeeze(0).numpy()
        mDepth = (mDepth + 1.0) / 2.0        # from -1, 1 to 0, 1
        cv2.imwrite(str(source_parent / f"{file.stem}.png"), mDepth)
        print(f"Saved {file.stem}.png", mDepth.shape, mDepth.min(), mDepth.max())

        # TODO: here probably better when sky is dark not white in rgb images


elif WHICH_TASK == "target_resize":
    for file in target_parent.iterdir():
        t = cv2.imread(str(file))
        t_shape = t.shape
        t_shape = (t_shape[1]//4, t_shape[0]//4)
        t = cv2.resize(t, t_shape, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target_parent / f"{file.stem}.png"), t)
        print(f"Saved {file.stem}.png", t.shape)

        # remove .jpg files
        file.unlink()

elif WHICH_TASK == "RGBdepth":
    rgb_file = Path(home_path, "dummy.txt")
    depth_file = Path(home_path, "dummy.txt")
    for file in sorted(source_path.iterdir(), key=lambda path: int(path.stem.split("_")[0])):
        if "depth" in file.stem:
            depth_file = file
        else:
            rgb_file = file

        if (depth_file.exists() and rgb_file.exists() and 
            depth_file.stem.split("-")[1].split(".")[0].replace("_depth", "") == rgb_file.stem.split("-")[1].split(".")[0]):
            color = cv2.imread(str(rgb_file))
            depth = cv2.imread(str(depth_file))
        
            # add depth to rgb as new channel
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            rgba = cv2.merge((color, depth))
            rgba_shape = rgba.shape
            rgba_shape = (rgba_shape[1]//4, rgba_shape[0]//4)
            rgba = cv2.resize(rgba, rgba_shape, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(source_parent / f"{rgb_file.stem}.png"), rgba)
            print(f"Saved {rgb_file.name} and {depth_file.name}", rgba.shape) 

elif WHICH_TASK == "prompt_manual":
    file = Path(target_parent.parent, "prompt.json")
    source_path = source_path.parent
    with open(file.as_posix(), "w") as f:
        for target in target_parent.iterdir():
            # find corresponding source image
            data_id = target.stem.split("_")[1] + "_" + target.stem.split("_")[2]
            for source in source_path.iterdir():
                if source.is_file() and data_id in source.stem:
                    source_stem = source.stem
                    break

            entry = {
                "source": f"source/{source_stem}.png",
                "target": f"target/{target.stem}.png",
                "prompt": f"Landscape with snow on high mountains with clouds in the sky in early autumn"
            }

            f.write(json.dumps(entry) + "\n")

        print(f"Saved {file.name}")

elif WHICH_TASK == "prompt_LLaVA_v15":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    llava = Llava_v15(device=device, llava_ckpt='llava-hf/bakLlava-v1-hf')
    print("Using LLaVA v1.5")

    file = Path(target_parent.parent, "prompt.json")
    source_path = source_path.parent
    llava.model.to('cuda')
    with open(file.as_posix(), "w") as f:
        for target in tqdm(target_parent.iterdir()):
            # find corresponding source image
            data_id = target.stem.split("_")[1] + "_" + target.stem.split("_")[2]
            source_file = None
            for source in source_path.iterdir():
                if source.is_file() and data_id in source.stem:
                    source_file = source
                    source_stem = source.stem
                    break

            # ask LLaVA for prompt            
            prompt = '<image>\n \
                USER: Detaily imagine and describe the scene this image taken from? \
                \n ASSISTANT: This image is taken from a scene of '

            responce_target = llava(PIL.Image.open(target), prompt)
            #responce_source = llava(PIL.Image.open(source_file), prompt)

            split = str.rfind(responce_target, "ASSISTANT: This image is taken from a scene of ") + len("ASSISTANT: This image is taken from a scene of ")
            responce_target = responce_target[split:]
            #split = str.rfind(prompt_source, "ASSISTANT: This image is taken from a scene of ") + len("ASSISTANT: This image is taken from a scene of ")
            #prompt_source = prompt_source[split:]

            print(f"Description for target {target.stem}: {responce_target}")
            #print(f"Description for source {source.stem}: {responce_source}\n")

            entry = {
                "source": f"source/{source_stem}.png",
                "target": f"target/{target.stem}.png",
                "prompt": responce_target
            }

            f.write(json.dumps(entry) + "\n")

        print(f"Saved {file.name}")

                
elif WHICH_TASK == "prompt_LLaVA_v16":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    llava = Llava_v16(device=device, llava_ckpt='llava-hf/llava-v1.6-mistral-7b-hf', max_tokens=300)
    print("Using LLaVA v1.6")

    file = Path(target_parent.parent, "prompt.json")
    source_path = source_path.parent
    llava.model.to('cuda')
    with open(file.as_posix(), "w") as f:
        for target in tqdm(target_parent.iterdir()):
            # find corresponding source image
            data_id = target.stem.split("_")[1] + "_" + target.stem.split("_")[2]
            source_file = None
            for source in source_path.iterdir():
                if source.is_file() and data_id in source.stem:
                    source_file = source
                    source_stem = source.stem
                    break
            
            mesh_file = None
            raw_path = Path(source_path, "raw")
            for mesh in raw_path.iterdir():
                if mesh.is_file() and data_id in mesh.stem and "_depth" not in mesh.stem:
                    mesh_file = mesh
                    break

            # ask LLaVA for prompt            
            responce = llava(PIL.Image.open(target), PIL.Image.open(mesh_file))

            #print(f"Description for target {target.stem}: {responce}")

            entry = {
                "source": f"source/{source_stem}.png",
                "target": f"target/{target.stem}.png",
                "prompt": responce
            }

            f.write(json.dumps(entry) + "\n")

        print(f"Saved {file.name}")
