import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import geemap as ee
import torch
from PIL import Image
import numpy as np

# data_engine imports
from dataVisu_engine.ThreeD_helpers import *
from dataVisu_engine.TwoD_helpers import *
from dataVisu_engine.GEE_helpers import *
from utils.general import getHomePath
from data_engine.utils.config import Config

# controlNet_engine imports
from controlNet_engine.controlNet_use import ControlNet_use
from controlNet_engine.prepare_dataset import *

class Data_pipeline():
    def __init__(self, config):
        """
        Args:
            config (Config): configuration object, see utils/config.py
        """
        # Checks
        assert isinstance(config, Config), f"config must be of type Config, got {type(config)}"

        self.config = config
        self.home_path = getHomePath()
        self.scene_path = config.data_pathScene
        self.transformMatrix = openTFMatrix(config.data_fileTransformMatrix)
        self.sd_transformMatrix = openTFMatrix(config.sd_transformMatrix_path)


    def generate_planar_images(self, scene_path):
        pano2plane(scene_path, self.transformMatrix)

    def get_gee_data(self):
        # Authenticate and initialize Earth Engine
        ee.Authenticate()
        ee.Initialize(project='seasonScapes-master')

        # process and save models
        _, dem_np, _ = processDEM_model(self.config.GEE)
        _, land_np = processLAND_model(self.config.GEE)
        self.config.loadCamMetadata(self.config.data_pathMetadata)

        land_np, dem_np = normalizeResize(land_np, dem_np, custom_cut=True)
        calcAOI_pixel(self.config, land_np)
        saveLAND_model(self.config, land_np)
        saveDEM_model(self.config, dem_np)

        self.config.saveGEE()

    def generate_novelView_mesh(self):
        pass

    def generate_SD_samples(self, rgbd=True):
        '''
        Generate realistic samples with stable diffusion.
        Mesh is saved in NovelViews folder. Realistic images are saved in scene folder as they vary between scenes.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # TODO: prompt automation, parameter optimization
        prompt = 'Landscape with snow on high mountains with clouds in the sky in early autumn'  # add in the morning, ...

        controlNet = ControlNet_use(self.config.sd_sd_model_path, self.config.sd_control_model_path, rgbd=rgbd)

        # Loop through all control images
        folder = Path(self.home_path, self.config.sd_control_img_folder)
        print(f"Generating for control images {folder}")
        for control_img in folder.iterdir():
            if control_img.is_dir():
                continue
            print(f'Processing {control_img.stem}')
            input_image = Image.open(control_img)
            input_image = np.array(input_image)
            genImages = controlNet.run(input_image, prompt, self.config.sd_a_prompt, self.config.sd_n_prompt, self.config.sd_num_samples, 
                                self.config.sd_image_resolution, self.config.sd_detect_resolution, self.config.sd_ddim_steps, 
                                self.config.sd_guess_mode, self.config.sd_strength, self.config.sd_scale, self.config.sd_seed, 
                                self.config.sd_eta, self.config.sd_calcDepth)
            
            # save first generated image
            gen_img = genImages[1]
            name = control_img.stem
            save_folder = Path(self.home_path, self.config.sd_save_folder)
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            gen_img = Image.fromarray(gen_img)
            gen_img.save(f'{save_folder.as_posix()}/{name}.png')
            
        # save transformation matrix
        #with open(Path(self.home_path, self.config.sd_save_folder, 'transformMatrix.json'), 'w') as f:
        #    json.dump(self.sd_transformMatrix)
                
    def generate_controlNet_training_data(self, folders:list=None, skip_sampleMesh=False, prompt_type=None):
        '''
        Generate training data for controlNet.
        1. Sample color/depth mesh for cameras once
        Loop:
        2. Split up 360° images into several planar images
        3. Build ControlNet training folder
            - target images: yyyy-mm-dd-hh-mm-....png
            - source images: yyyy-mm-dd-hh-mm-....png
            - prompt.json with LLaVa prompt
        Args: 
            folders (list): list of folders to process [2024-11-22, ...]
        '''
        self.llava_prompt = llava_prompt(max_tokens=300)

        # all folders in data folder
        if folders is None:
            data_path = Path(self.home_path, 'data')
            folders = [x for x in data_path.iterdir() if (x.is_dir() and x.stem.startswith('20'))]
        else:
            folders = [Path(self.home_path, 'data', folder) for folder in folders]
        folders_all = [x for folder in folders for x in folder.iterdir() if x.is_dir()]

        if not skip_sampleMesh:
            self.sampleMesh(self.transformMatrix)

        controlTrain_folder = Path(self.home_path, 'controlNet_engine', 'training', f'rAll_{prompt_type}')

        # loop through all folders
        tqdm_bar = tqdm(folders_all)
        for folder in tqdm_bar:
            self.scene_path = folder # TODO: test if I have to split
            tqdm_bar.set_description(f'Processing {folder}')
            self.generate_planar_images(self.scene_path)
            self.build_controlNet_train_folder(self.scene_path, controlTrain_folder, prompt_type=prompt_type)

    def sampleMesh(self, transformMatrix):
        '''
        Sample low-resolution mesh for controlNet training
        '''
        gee_hash = str(self.config.GEE_hash)[0:5]
        mesh_file = f"mesh_{gee_hash}.ply"
        mesh = loadTriMesh_ply(self.config.GEE_save_path, mesh_file)
        sampleImg_TriMesh_all(self.config, mesh, transformMatrix, folder_name='meshRoundshot', filter=False)

    def build_controlNet_train_folder(self, scene_path, controlNet_folder, prompt_type=None):
        '''
        Build ControlNet training folder
        - target images: yyyy-mm-dd-hh-mm-....png
        - source images: yyyy-mm-dd-hh-mm-....png
        - prompt.json with LLaVa prompt
        '''
        # get all images
        target_folder = Path(self.home_path, scene_path, 'imagesPlane')
        source_folder = Path(self.home_path, 'data', 'meshRoundshot')

        # create folder and prompt.json file
        controlNet_folder = Path(self.home_path, controlNet_folder)
        if not controlNet_folder.exists():
            controlNet_folder.mkdir(parents=True)
            prompt_json = controlNet_folder / 'prompt.json'
            with open(prompt_json, 'w') as f:
                f.write('{}')

        # create prompt and copy images
        self.llava_prompt.append_llava_prompt_moveImg(target_folder=target_folder, source_folder=source_folder, controlNet_folder=controlNet_folder, 
                                    rgbd=True, resize_fac=4, prompt_type=prompt_type)


    def __call__(self):
        # TODO: Data pipeline
        # 1. GEE get map
        # 2. get satellite?
        # 3. create ply of aoi --> pcd, mesh
        # 4. sample triMesh in AOI
        # 5. generate diffused real samples
        pass