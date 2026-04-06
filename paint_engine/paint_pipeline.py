import sys
import os

import random
import numbers
import numpy as np
import torch
import torch.nn as nn
import time
import trimesh
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import kaolin as kal
import cv2
import copy
from transformers import pipeline
import requests
try:
    from pytorch3d.renderer.mesh import rasterize_meshes
    from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesUV, MeshRasterizer, MeshRenderer, HardPhongShader, PointLights, RasterizationSettings, BlendParams
    from pytorch3d.renderer.mesh.shading import interpolate_face_attributes
    _has_pytorch3d = True
except ImportError:
    _has_pytorch3d = False
    print("WARNING: PyTorch3D not found. Optimized rasterization is unavailable.")


from data_engine.utils.general import getHomePath, natural_sort_key
from data_engine.utils import camFormat
from data_engine.utils.config import Config
from paint_engine.cfg import PaintConfig
from paint_engine.diffusers_cnet import *
from paint_engine.diffusers_cnet_img2img import img2imgControlNet
from paint_engine import utils

from controlNet_engine.annotator.midas import MidasDetector


class Paint_pipeline():
    """Pipeline for painting known views and inpainting missing views of a mesh
    using a diffusion-based inpainting method.
    Args:
        cfg: PaintConfig object
    """
    def __init__(self, cfg, save_debug=False):
        start_t = time.time()
        self.save_debug = save_debug
        self.cfg = cfg
        self.cfg_dataEngine = self.cfg.dataset.config
        self.home_path = getHomePath()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = getattr(self.cfg.optim, 'batch_size', 1)  # Default batch size to 1 if not specified
        self.mesh_model = TexturedMeshModel(cfg=self.cfg, save_debug=save_debug, device=self.device)
        self.center, self.scale = self.mesh_model.get_normalization_params()
        self._init_dataloaders()    # has to be after get_normalization_params
        self.inpaint_cNet = inpaintControlNet(self.cfg.diffusion.inpaint)
        # self.UVHD_cNet = img2imgControlNet(self.cfg.diffusion.img2img)
        self._seed_everything(self.cfg.optim.seed)
        print("Finished initialization in", round(time.time() - start_t, 1), "seconds\n")
        
        # debug
        if save_debug:
            path = Path(self.cfg.log.exp_path, "debug_init")
            self.mesh_model.export_mesh(path.as_posix())

    def _seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _init_dataloaders(self):
        """
        Initialize datasets and create data loaders 
        for known views and novel views
        """
        # Create dataset for known views
        self.dataset_knownView = DatasetCamera(data_path=self.cfg.dataset.fileTransformMatrix, cfg_dataEngine=self.cfg_dataEngine, 
                                               cfg_render=self.cfg.render, center=self.center, scale=self.scale)
        self.dataset_novelView = DatasetCamera(data_path=self.cfg.diffusion.transformMatrix_novel_path, cfg_dataEngine=self.cfg_dataEngine, 
                                               cfg_render=self.cfg.render, center=self.center, scale=self.scale, load_img=False)
        self.dataset_eval = DatasetCamera(data_path=self.cfg.diffusion.transformMatrix_eval_path, cfg_dataEngine=self.cfg_dataEngine, 
                                          cfg_render=self.cfg.render, center=self.center, scale=self.scale, load_img=False)
        self.dataset_metrics = DatasetCamera(data_path=self.cfg.diffusion.transformMatrix_metrics_path, cfg_dataEngine=self.cfg_dataEngine,
                                               cfg_render=self.cfg.render, center=self.center, scale=self.scale, load_img=False)
        # Create data loader for known views
        self.dataloader_knownView = DataLoader(
            dataset=self.dataset_knownView,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        self.dataloader_novelView = DataLoader(
            dataset=self.dataset_novelView,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        self.dataloader_eval = DataLoader(
            dataset=self.dataset_eval,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        self.dataloader_metrics = DataLoader(
            dataset=self.dataset_metrics,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        print(f"Created dataloaders: Known Views {len(self.dataset_knownView)}, Novel Views {len(self.dataset_novelView)}, Eval Views {len(self.dataset_eval)}, Metrics Views {len(self.dataset_metrics)}")

    def evaluation(self):
        """Evaluation API function, load mesh from dataset.trained_mesh path"""
        save_result_dir = Path(self.cfg.log.exp_path, "eval_metrics")
        # TODO: target 1) render image, 2) img2img for better results
        print(f"Evaluation save to {save_result_dir}")
        self.mesh_model.inpaint_without_forwarding(self.inpaint_cNet, self.dataloader_metrics, save_dir=save_result_dir)
        self.mesh_model.eval(dataloader=self.dataloader_metrics, save_result_dir=save_result_dir)
        

    def __call__(self):
        total_start = time.time()
        start_t = time.time()
        
        # Paint known views
        if not self.cfg.log.ablation_no_paint:
            self.mesh_model.paint_known_views(self.dataloader_knownView, cNet=self.inpaint_cNet, center=self.center, scale=self.scale)
            print(f"\nTime to paint known views took {round(time.time() - start_t, 1)} s\n")

            if self.cfg.log.only_paint:
                self.mesh_model.export_mesh(Path(self.cfg.log.exp_path, "painted_mesh").as_posix())
                raise RuntimeError("Only paint activated - for inpaint deactivate cfg.log.only_paint")
    
        # Inpainting
        if not self.cfg.log.ablation_no_inpaint:
            start_t = time.time()
            self.mesh_model.inpaint(cNet=self.inpaint_cNet, dataloader=self.dataloader_novelView)
            print(f"\nTime to inpaint took {round(time.time() - start_t, 1)} s\n")

        # UV map inpainting --> remove red lines on mesh via uv map
        if not self.cfg.log.ablation_no_uvInpaint and self.cfg.render.uv_inpaint_method != "none":
            start_t = time.time()
            save_result_dir = Path(self.cfg.log.exp_path, "UV_inpaint")
            self.mesh_model.UV_inpaint(cnet=self.inpaint_cNet, outdir=save_result_dir.as_posix())
            print(f"\nTime to inpaint UV took {round(time.time() - start_t, 1)} s\n")

        # Video generation
        start_t = time.time()
        save_result_dir = Path(self.cfg.log.exp_path, "novel_video")
        self.mesh_model.eval(dataloader=self.dataloader_eval, save_result_dir=save_result_dir, save_imges=False)
        print(f"\nTime to generate videos took {round(time.time() - start_t, 1)} s\n")

        # # UVHD - does not improve the result
        # start_t = time.time()
        # save_result_dir = Path(self.cfg.log.exp_path, "UVHD")
        # self.mesh_model.UVHD(cnet=self.UVHD_cNet, outdir=save_result_dir.as_posix())
        # print(f"Time for UVHD took {round(time.time() - start_t, 1)} s")

        # Metric evaluation with Roundshot images
        start_t = time.time()
        self.evaluation()
        print(f"\nTime to evaluate took {round(time.time() - start_t, 1)} s\n")

        print(f"\nTime total: {round(time.time() - total_start, 1)} s\n")


class TexturedMeshModel():
    def __init__(self, cfg, save_debug=False, device=torch.device('cpu')):
        self.cfg = cfg
        self.save_debug = save_debug
        self.cfg_dataEngine = cfg.dataset.config
        self.device = device
        self.default_color = cfg.render.texture_default_color
        self.home_path = getHomePath()
        self.mesh, self.vt, self.ft = self.load_mesh()
        self.mesh.to(self.device)
        self.centroid = None
        self.scale_factor = None
        self.normalize_mesh()
        if self.vt is None or self.ft is None:
            self.vt, self.ft = self.init_texture_map()            # torch.Size([435766, 3])
        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(self.vt.unsqueeze(0), self.ft.long()).detach()#.to(self.device)
        self.texture_resolution = [self.cfg.render.texture_resolution[0], self.cfg.render.texture_resolution[1]]
        self.renderer = Renderer(cfg=self.cfg, mesh_face_num=self.mesh.faces.shape[0], device=self.device, save_debug=save_debug)
        self.texture_img = self.init_texture_img()       # nn.Parameter to gradually update texture
        self.texture_mask = torch.zeros_like(self.texture_img)
        self.postprocess_edge = torch.zeros_like(self.texture_img)
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        self.texture_img_postprocess = None
        self.texture_list = []
        # depth masking known view
        if self.cfg.render.depth_estimation == "midas":
            self.apply_midas = MidasDetector()
        elif self.cfg.render.depth_estimation == "depthAny_v2":
            self.apply_depthAnything = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=self.device)
        else:
            raise ValueError(f"Invalid depth estimation method: {self.cfg.render.depth_estimation} not implemented")
        
    def init_texture_img(self):
        """Initialize the texture image of the mesh from file or default color"""
        if self.cfg.render.initial_texture_path is not None:
            texture_map = Image.open(self.cfg.render.initial_texture_path).convert("RGB").resize(self.texture_resolution)
            texture = torch.Tensor(np.array(texture_map)).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        else:
            texture = torch.ones(1, 3, *self.texture_resolution).to(self.device) * torch.Tensor(self.default_color).reshape(1, 3, 1, 1).to(self.device)
        texture_img = nn.Parameter(texture)
        return texture_img

    def get_normalization_params(self):
        return self.centroid, self.scale_factor

    def paint_known_views(self, dataloader, cNet=None, center=None, scale=None):
        """Paint known views onto the mesh"""
        # loop through all known images
        tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Init: paint known views")
        for idx, cam_data in tqdm_bar:
            # paint the known images
            self.paint_image(cam_data, cNet=cNet, center=center, scale=scale)

        if self.save_debug:
            path = Path(self.cfg.log.exp_path, "debug_paint")
            print("DEBUG: Exporting mesh to", path)
            self.export_mesh(path.as_posix())

    def paint_image(self, cam_data, cNet=None, center=None, scale=None):
        """Paint a known image onto the mesh"""
        # Get camera data
        w2c = cam_data['w2c'].to(self.device)
        view_target = cam_data['image']
        
        intrinsic_matrix = cam_data['intrinsic']        # intrinsic are already scaled
        self.renderer.intrinsics = kal.render.camera.PinholeIntrinsics.from_focal(width=int(self.cfg.render.width*self.cfg.render.scale_img), height=int(self.cfg.render.height*self.cfg.render.scale_img), 
                                                                                  focal_x=intrinsic_matrix[0,0,0], focal_y=intrinsic_matrix[0,1,1], device=self.device)
        # normalized depth for target view mask
        view_target_ui8 = view_target * 255
        view_target_ui8 = view_target_ui8.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if self.cfg.render.depth_estimation == "midas":
            view_depth_inv, _ = self.apply_midas(view_target_ui8)
        elif self.cfg.render.depth_estimation == "depthAny_v2":
            view_target_pil = Image.fromarray(view_target_ui8)
            view_depth_inv = self.apply_depthAnything(view_target_pil)['depth']   # better depth gradient, near things sometimes worse
        
        view_depth_inv = torch.from_numpy(np.array(view_depth_inv)).to(self.device)
        view_target = view_target.to(self.device)

        self.forward_texturing(view_target=view_target, view_depth_inv=view_depth_inv, camera_transform=w2c, view_id=cam_data['cam_id'])

        # inpaint from a higher attitude
        if self.cfg.render.inpaint_from_higher_attitude and self.cfg.render.inpaint_from_higher_attitude_height > 0 and cNet is not None:
            c2w = cam_data['c2w']
            normalized_height = self.cfg.render.inpaint_from_higher_attitude_height * np.array(scale)
            c2w[:, 2, 3] = c2w[:, 2, 3] + normalized_height
            w2c_higher = np.linalg.inv(np.array(c2w))
            w2c_higher = torch.from_numpy(w2c_higher).to(self.device).float()
            cam_data['w2c'] = w2c_higher
            self.inpaint_image(cam_data, cNet, save_dir="inpaint_higher", save_mask=False)
            self.forward_texturing(view_target=view_target, view_depth_inv=view_depth_inv, camera_transform=w2c_higher, view_id=cam_data['cam_id'], paint_mode=False)

            # forward texturing
            cam_id = cam_data['cam_id'][0]
            view_target = utils.pil2tensor(Image.open(Path(self.cfg.log.exp_path, "inpaint_higher", cam_id + "_inpaint_0.png")), self.device)
            view_depth_inv = torch.ones([view_target.shape[2], view_target.shape[3]]).to(self.device)
            self.forward_texturing(view_target=view_target, view_depth_inv=view_depth_inv, camera_transform=w2c_higher, paint_mode=False)


    def inpaint_image(self, cam_data, cNet, save_dir="inpaint", save_mask=True):
        """Inpaint one missing image using a ControlNet"""
        inpaint_used_key = ["image", "depth", "uncolored_mask"]
        img_cfg = self.cfg.diffusion.inpaint

        w2c = cam_data['w2c'].to(self.device)
        cam_id = cam_data['cam_id'][0]
        intrinsic_matrix = cam_data['intrinsic']
        self.renderer.intrinsics = kal.render.camera.PinholeIntrinsics.from_focal(width=int(self.cfg.render.width*self.cfg.render.scale_img), height=int(self.cfg.render.height*self.cfg.render.scale_img), 
                                                                                    focal_x=intrinsic_matrix[0,0,0], focal_y=intrinsic_matrix[0,1,1], device=self.device)

        # projection at w2c
        outputs = self.render(camera_transform=w2c)
        for key in inpaint_used_key:
            if key == "uncolored_mask":
                mask = outputs[key]
                mask[mask>0] = 1
                if self.cfg.diffusion.inpaint.dilate_regions and self.cfg.diffusion.inpaint.dilate_kernel > 0:
                    dilate_kernel_top = self.cfg.diffusion.inpaint.dilate_kernel
                    dilate_kernel_mid = dilate_kernel_top*2
                    dilate_kernel_low = dilate_kernel_top*2 + 2
                    img_height = self.cfg.diffusion.inpaint.height
                    img_width = self.cfg.diffusion.inpaint.width
                    outputs[key] = F.conv2d(mask, torch.ones((1, 1, dilate_kernel_top, dilate_kernel_top), device=self.device), padding=dilate_kernel_top//2)[:, :, :img_height, :img_width]
                    outputs[key][:, :, int(1/2*img_height):img_height, :] = F.conv2d(mask, torch.ones((1, 1, dilate_kernel_mid, dilate_kernel_mid), device=self.device), padding=dilate_kernel_mid//2)[:, :, int(1/2*img_height):img_height, :img_width]
                    outputs[key][:, :, int(2/3*img_height):img_height, :] = F.conv2d(mask, torch.ones((1, 1, dilate_kernel_low, dilate_kernel_low), device=self.device), padding=dilate_kernel_low//2)[:, :, int(2/3*img_height):img_height, :img_width]
                    outputs[key][outputs[key] > 0] = 1
                elif self.cfg.diffusion.inpaint.dilate_kernel > 0:
                    dilate_kernel = self.cfg.diffusion.inpaint.dilate_kernel
                    outputs[key] = F.conv2d(mask, torch.ones((1, 1, dilate_kernel, dilate_kernel), device=self.device), padding=dilate_kernel//2)
                    outputs[key][outputs[key] > 0] = 1
                utils.save_tensor_image(mask, Path(self.cfg.log.exp_path, save_dir, cam_id + f"_maskBefore.png").as_posix())
            if key == "depth" and save_mask:      # save landscape mask for gs training
                gs_mask = outputs[key] > 0
                save_path = Path(self.cfg.log.exp_path, save_dir, cam_id + "_GSmask.png")
                utils.save_tensor_image(gs_mask, save_path.as_posix())
            save_path = Path(self.cfg.log.exp_path, save_dir, cam_id + f"_{key}.png")
            utils.save_tensor_image(outputs[key], save_path.as_posix())

        img_cfg.image_path = Path(self.cfg.log.exp_path, save_dir, cam_id + "_image.png").as_posix()
        img_cfg.mask_path = Path(self.cfg.log.exp_path, save_dir, cam_id + "_uncolored_mask.png").as_posix()
        img_cfg.controlnet_units[0]['condition_image_path'] = Path(self.cfg.log.exp_path, save_dir, cam_id + "_depth.png").as_posix()
        images = cNet.inference(config=img_cfg)
        for i, img in enumerate(images):
            save_path = Path(self.cfg.log.exp_path, save_dir, cam_id + f"_inpaint_{i}.png")
            img = img.resize((int(self.cfg.render.width*self.cfg.render.scale_img), int(self.cfg.render.height*self.cfg.render.scale_img)), resample=Image.BILINEAR)
            img.save(save_path.as_posix())


    def inpaint(self, cNet, dataloader):
        """Inpaint missing views using a ControlNet using the dataloader"""
        tqdm_bar = tqdm(dataloader, total=len(dataloader), desc="Project, inpaint and forward views to mesh...")
        for cam_data in tqdm_bar:
            self.inpaint_image(cam_data, cNet, save_dir="inpaint")

            # forward texturing
            cam_id = cam_data['cam_id'][0]
            w2c = cam_data['w2c'].to(self.device)
            view_target = utils.pil2tensor(Image.open(Path(self.cfg.log.exp_path, "inpaint", cam_id + "_inpaint_0.png")), self.device)
            view_depth_inv = torch.ones([view_target.shape[2], view_target.shape[3]]).to(self.device)
            self.forward_texturing(view_target=view_target, view_depth_inv=view_depth_inv, camera_transform=w2c, paint_mode=False)

        path = Path(self.cfg.log.exp_path, "inpainted_mesh")
        print("Exporting inpainted mesh to", path)
        self.export_mesh(path.as_posix())

    def inpaint_without_forwarding(self, cNet, dataloader, save_dir):
        """Inpaint missing views using a ControlNet"""
        tqdm_bar = tqdm(dataloader, total=len(dataloader), desc="Project, inpaint and forward views to mesh...")
        for cam_data in tqdm_bar:
            self.inpaint_image(cam_data, cNet, save_dir=save_dir, save_mask=False)

    def UV_inpaint(self, cnet, outdir):
        p_cfg = copy.deepcopy(self.cfg.diffusion.inpaint)
        self.export_mesh(outdir, export_texture_only=True)
        albedo_path = Path(self.home_path, outdir, f"albedo.png")
        albedo = cv2.imread(albedo_path.as_posix())
        UV_pos = self.UV_pos_render()
        UV_pos_path = Path(self.home_path, outdir, f"UV_pos.png")
        utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

        # processing mask
        mask_color = [self.cfg.render.texture_default_color[2], self.cfg.render.texture_default_color[1], self.cfg.render.texture_default_color[0]]  # BGR
        mask_color = [int(c*255) for c in mask_color]
        mask_dilated = utils.extract_bg_mask(albedo_path, mask_color=mask_color, dilate_kernel=0)
        mask_path = os.path.join(outdir, f"mask.png")
        cv2.imwrite(mask_path, mask_dilated)

        # UV inpaint with Neighbors
        uv_inpainted_path = Path(self.home_path, outdir, f"UV_inpaint_res.png")
        if self.cfg.render.uv_inpaint_method == "neighbor":
            filled_uv_albedo = cv2.inpaint(albedo, mask_dilated, inpaintRadius=1, flags=cv2.INPAINT_TELEA)   # cv2.INPAINT_NS or cv2.INPAINT_TELEA
            cv2.imwrite(uv_inpainted_path.as_posix(), filled_uv_albedo)
            filled_uv_albedo = cv2.cvtColor(filled_uv_albedo, cv2.COLOR_BGR2RGB)
            filled_uv_albedo = torch.Tensor(np.array(filled_uv_albedo)).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        # UV inpaint with ControlNet
        elif self.cfg.render.uv_inpaint_method == "controlnet":
            p_cfg.image_path = albedo_path
            p_cfg.mask_path =  mask_path
            p_cfg.controlnet_units[0]['condition_image_path'] = UV_pos_path
            p_cfg.controlnet_units[0]['controlnet_key'] = "GeorgeQi/Paint3d_UVPos_Control"
            p_cfg.controlnet_units[0]['weight'] = 0.0
            p_cfg.controlnet_units[1]['weight'] = 1.0
            p_cfg.width = self.texture_resolution[1]
            p_cfg.height = self.texture_resolution[0]

            images = cnet.inference(config=p_cfg)
            new_albedo = images[0]
            new_albedo.save(uv_inpainted_path.as_posix())

            # save mesh with inpainted texture
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = torch.Tensor(np.array(albedo)).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            new_albedo = torch.Tensor(np.array(new_albedo)).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            mask_dilated = torch.from_numpy(mask_dilated).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            filled_uv_albedo = albedo * (1 - mask_dilated) + new_albedo * mask_dilated
        
        self.texture_img = nn.Parameter(filled_uv_albedo)
        outdir = Path(outdir, "uv_inpainted_mesh")
        self.export_mesh(outdir)

        return filled_uv_albedo
    
    def UVHD(self, cnet, outdir):
        "image-to-image controlnet"
        p_cfg = copy.deepcopy(self.cfg.diffusion.img2img)
        self.export_mesh(outdir, export_texture_only=True)  
        albedo_path = os.path.join(outdir, f"albedo.png")
        UV_pos = self.UV_pos_render()
        UV_pos_path = os.path.join(outdir, f"UV_pos.png")
        utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

        # UV inpaint
        p_cfg.image_path = albedo_path
        p_cfg.controlnet_units[0]['condition_image_path'] = UV_pos_path
        p_cfg.controlnet_units[1]['condition_image_path'] = albedo_path
        p_cfg.width = self.texture_resolution[1]
        p_cfg.height = self.texture_resolution[0]

        images = cnet.inference(config=p_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(outdir, f'UV_tile_res_{i}.png')
            img.save(save_path)

        # save mesh with inpainted texture
        # outdir = Path(outdir, "uvhd_mesh")
        # self.export_mesh(outdir)
        return images


    def forward_texturing(self, view_target: torch.tensor, view_depth_inv: torch.tensor, camera_transform:torch.tensor, save_result_dir=None, view_id=None, save=False, paint_mode=True):
        outputs = self.render(camera_transform=camera_transform)
        uncolored_mask_render = outputs['uncolored_mask']  # bchw, [0,1]
        render_cache = outputs['render_cache']
        erode_size = 19
        uncolored_mask_render = torch.from_numpy(
            cv2.erode(uncolored_mask_render[0, 0].detach().cpu().numpy(), np.ones((erode_size, erode_size), np.uint8))).to(
            uncolored_mask_render.device).unsqueeze(0).unsqueeze(0)
        
        # build view mask
        if self.cfg.render.use_depth_mask and paint_mode:
            view_depth_inv = (view_depth_inv - torch.min(view_depth_inv)) / (torch.max(view_depth_inv) - torch.min(view_depth_inv))
            flat_depth = view_depth_inv.flatten().float().cpu()
            boundaries = torch.tensor([1.0 - self.cfg.render.depth_filter_boundaries[1], 1.0 - self.cfg.render.depth_filter_boundaries[0]]).float()
            thresholds = torch.quantile(flat_depth, boundaries).float()
            near_treshold, far_treshold = thresholds[0], thresholds[1]
            mask_near = view_depth_inv > near_treshold
            mask_far = view_depth_inv < far_treshold
            view_target_mask = mask_near & mask_far
        else:
            view_target_mask = torch.ones_like(view_depth_inv)
        view_target_mask = view_target_mask.unsqueeze(0).unsqueeze(0)

        next_texture_map, next_texture_mask, weight_map = self.renderer.forward_texturing_render(
            self.mesh.vertices, self.mesh.faces, self.face_attributes, camera_transform=camera_transform, view_target=view_target,
            view_target_mask=view_target_mask, uncolored_mask=uncolored_mask_render, texture_dims=self.texture_resolution, render_cache=render_cache)
        
        updated_texture_map = next_texture_map * next_texture_mask + self.texture_img * (1 - next_texture_mask)
        
        if save:
            print("Saving forward texture results to", save_result_dir)
            utils.save_tensor_image(view_target, os.path.join(save_result_dir, f"_view_{view_id}_view_target.png"))
            utils.save_tensor_image(view_depth_inv.repeat(1,3,1,1), os.path.join(save_result_dir, f"_view_{view_id}_view_depth_inv.png"))
            utils.save_tensor_image(view_target_mask, os.path.join(save_result_dir, f"_view_{view_id}_view_target_mask.png"))
            utils.save_tensor_image(uncolored_mask_render.repeat(1,3,1,1), os.path.join(save_result_dir, f"_view_{view_id}_uncolored_mask_render.png"))
            utils.save_tensor_image(next_texture_map, os.path.join(save_result_dir, f"_view_{view_id}_next_texture_map.png"))
            utils.save_tensor_image(next_texture_mask, os.path.join(save_result_dir, f"_view_{view_id}_next_texture_mask.png"))
            utils.save_tensor_image(weight_map, os.path.join(save_result_dir, f"_view_{view_id}_weight_map.png"))
            utils.save_tensor_image(updated_texture_map, os.path.join(save_result_dir, f"_view_{view_id}_updated_texture_map.png"))

        if self.save_debug:
            # render updated texture
            common_path = Path(self.home_path, self.cfg.log.exp_path, "debug_forTextRender")
            uv_features = outputs['render_cache']['uv_features']
            old_image = kal.render.mesh.texture_mapping(uv_features, self.texture_img.detach(), mode='bilinear')
            updated_image = kal.render.mesh.texture_mapping(uv_features, updated_texture_map, mode='bilinear')
            mask = outputs['mask']
            old_image = old_image[0]*mask.unsqueeze(-1).squeeze(0)
            updated_image = updated_image[0]*mask.unsqueeze(-1).squeeze(0)
            # utils.save_tensor_image(old_image.permute(0,3,1,2), Path(common_path, f"view_{view_id}_image_old.png").as_posix())
            # utils.save_tensor_image(updated_image.permute(0,3,1,2), Path(common_path, f"view_{view_id}_image_updated.png").as_posix())
            utils.save_tensor_image(outputs['normals'], Path(common_path, f"view_{view_id}_normals.png").as_posix())
            # utils.save_tensor_image(view_depth_inv.unsqueeze(0), Path(common_path, f"view_{view_id}_depth_inv.png").as_posix())
            utils.save_tensor_image(outputs['depth'], Path(common_path, f"view_{view_id}_depth_render.png").as_posix())
            # utils.save_tensor_image(outputs['mask'], Path(common_path, f"view_{view_id}_mask_render.png").as_posix())
            # utils.save_tensor_image(outputs['uncolored_mask'], Path(common_path, f"view_{view_id}_uncolored_mask_render.png").as_posix())
            # utils.save_tensor_image(outputs['image'], Path(common_path, f"view_{view_id}_image_render.png").as_posix())
            # utils.save_tensor_image(uv_features.permute(0,3,1,2), Path(common_path, f"view_{view_id}_uv_features.png").as_posix())
            utils.save_tensor_image(view_target, Path(common_path, f"view_{view_id}_view_target.png").as_posix())
            utils.save_tensor_image(view_depth_inv.repeat(1,3,1,1), Path(common_path, f"_view_{view_id}_view_depth_inv.png").as_posix())
            # utils.save_tensor_image(view_target_mask, Path(common_path, f"_view_{view_id}_view_target_mask.png").as_posix())
            utils.save_tensor_image(uncolored_mask_render.repeat(1,3,1,1), Path(common_path, f"_view_{view_id}_uncolored_mask_render.png").as_posix())
            utils.save_tensor_image(next_texture_map, Path(common_path, f"_view_{view_id}_next_texture_map.png").as_posix())
            utils.save_tensor_image(next_texture_mask, Path(common_path, f"_view_{view_id}_next_texture_mask.png").as_posix())
            # utils.save_tensor_image(weight_map, Path(common_path, f"_view_{view_id}_weight_map.png").as_posix())
            # utils.save_tensor_image(updated_texture_map, Path(common_path, f"_view_{view_id}_updated_texture_map.png").as_posix())
            # utils.save_tensor_image(self.texture_img, Path(common_path, f"texture_img.png").as_posix())

        self.texture_img = nn.Parameter(updated_texture_map)
    
    def load_mesh(self):
        """Load mesh from cfg.dataset.trained_mesh or cfg.dataset.GEE_mesh_path"""
        if self.cfg.render.trained_mesh is not None:
            mesh_path = Path(self.home_path, self.cfg.render.trained_mesh)
        else:
            mesh_path = Path(self.home_path, self.cfg_dataEngine.GEE_mesh_path)
        if not mesh_path.suffix == ".obj" and not mesh_path.suffix == ".off":
            mesh_temp = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)
            mesh_path = os.path.splitext(mesh_path)[0] + "_cvt.obj"
            mesh_path = Path(mesh_path)
            mesh_temp.export(mesh_path.as_posix())
            print("Converting mesh to obj file without material")

        try:
            if ".obj" in mesh_path.as_posix():
                try:
                    mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=True)
                    print(f"Loaded trained mesh from {mesh_path}")
                except Exception as e:
                    print(f"Error {e} loading mesh with materials. Trying without materials.")
                    mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=False)
            elif ".off" in mesh_path:
                mesh = kal.io.off.import_mesh(mesh_path)
        except Exception as e:
            raise ValueError(f"Error loading mesh {mesh_path}: {e}")
        
        try:
            if mesh.uvs.shape[0] == 0 or mesh.face_uvs_idx.shape[0] == 0:
                raise AttributeError
            vt = mesh.uvs                       
            ft = mesh.face_uvs_idx             
        except AttributeError:
            print("INFO: Loaded mesh does not have UV coordinates")
            vt = None
            ft = None

        return mesh, vt, ft
    
    def init_texture_map(self):
        cache_path = Path(self.home_path, self.cfg.log.cache_path)
        mesh_id = str(self.cfg.log.cache_id)
        vt_cache_path, ft_cache_path = Path(cache_path, 'vt_' + mesh_id + '.pth'), Path(cache_path, 'ft_' + str(mesh_id) + '.pth')
        print(f"vt_cache_path: {vt_cache_path}, \nft_cache_path: {ft_cache_path}")
        if cache_path.exists():
            cache_exists_flag = vt_cache_path.exists() and ft_cache_path.exists()
        else:
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_exists_flag = False

        run_xatlas = False
        if cache_exists_flag:
            vt = torch.load(vt_cache_path, weights_only=True).to(self.device)
            ft = torch.load(ft_cache_path, weights_only=True).to(self.device)
            print(f'loaded cached uv map {cache_path}: vt={vt.shape} ft={ft.shape}')
        else:
            run_xatlas = True

        if run_xatlas or self.cfg.render.force_run_xatlas:
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            print(f'running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
            time_start = time.time()

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            # set options: https://cocalc.com/github/godotengine/godot/blob/master/thirdparty/xatlas/xatlas.h
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4        # try: 5 --> higher = better quality, but longer
            chart_options.max_chart_area = 0.1
            chart_options.fix_winding = True
            pack_options = xatlas.PackOptions()
            pack_options.padding = 0  # Set padding per uv island
            pack_options.texels_per_unit = 1024
            pack_options.bilinear = True
            pack_options.blockAlign = True
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(self.device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(self.device)
            if cache_path is not None:
                torch.save(vt.cpu(), vt_cache_path)
                torch.save(ft.cpu(), ft_cache_path)
            
            # set mesh uvs
            self.mesh.uvs = vt
            self.mesh.face_uvs_idx = ft
            print(f'xatlas finished in {time.time() - time_start:.2f}s')

        return vt, ft

    def normalize_mesh(self):
        """Normalize mesh vertices to be centered at origin and fit in [-1, 1].
        Note: Only call once"""
        vertices = self.mesh.vertices
        
        # Center the mesh
        self.centroid = vertices.mean(dim=0)
        vertices_centered = vertices - self.centroid
        
        # Scale to [-1, 1]
        max_extent = torch.max(torch.abs(vertices_centered))
        self.scale_factor = 1.0 / max_extent
        self.mesh.vertices = vertices_centered * self.scale_factor
        
        print(f"Normalized mesh: centroid {self.centroid}, scale {self.scale_factor}")

    def denormalize_mesh(self):
        """Denormalize mesh vertices back to original coordinates"""
        vertices = self.mesh['vertices']
        if self.centroid is None or self.scale_factor is None:
            raise ValueError("Mesh not normalized. Call normalize_mesh() first.")
            return

        vertices_denormalized = vertices / self.scale_factor + self.centroid
        self.mesh['vertices'] = vertices_denormalized
        self.centroid = None
        self.scale_factor = None
    
    def render(self, camera_transform=None, use_meta_texture=False, render_cache=None, img_size=None):
        if render_cache is None:
            assert camera_transform is not None
        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img

        rgb, depth, mask, uncolored_mask, normals, render_cache = self.renderer.render_single_view_texture(
            self.mesh.vertices, self.mesh.faces, self.face_attributes, texture_img, camera_transform=camera_transform,
            render_cache=render_cache, img_size=img_size, texture_default_color=self.default_color)
        if not use_meta_texture:
            rgb = rgb.clamp(0, 1)

        return {'image': rgb, 'mask': mask.detach(), 'uncolored_mask': uncolored_mask, 'depth': depth,
                'normals': normals, 'render_cache': render_cache, 'texture_map': texture_img}


    def UV_pos_render(self):
        UV_pos = self.renderer.UV_pos_render(self.mesh.vertices, self.mesh.faces, self.face_attributes,
                                    texture_dims=self.texture_resolution)
        return UV_pos
    
    @torch.no_grad()
    def export_mesh(self, path, export_texture_only=False):
        texture_img = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_img = Image.fromarray((texture_img[0].cpu().detach().numpy() * 255).astype(np.uint8))
        if not os.path.exists(path):
            os.makedirs(path)
        texture_img.save(os.path.join(path, f'albedo.png'))

        if self.texture_img_postprocess is not None:
            texture_img_post = self.texture_img_postprocess.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
            texture_img_post = Image.fromarray((texture_img_post[0].cpu().detach().numpy() * 255).astype(np.uint8))
            os.system("mv {} {}".format(texture_img.save(os.path.join(path, f'albedo.png')),
                                        texture_img.save(os.path.join(path, f'albedo_before.png'))))
            texture_img_post.save(os.path.join(path, f'albedo.png'))

        if export_texture_only: return 0

        v, f = self.mesh.vertices, self.mesh.faces.int()
        # denormalize mesh --> because if used again, it does not match to camera poses
        if self.centroid is not None and self.scale_factor is not None:
            v = v / self.scale_factor + self.centroid
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]
        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'mesh.obj')
        mtl_file = os.path.join(path, f'mesh.mtl')

        print(f'writing obj mesh to {obj_file} with: vertices:{v_np.shape} uv:{vt_np.shape} faces:{f_np.shape}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib mesh.mtl \n')

            for v in v_np:     
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            for v in vt_np:    
                fp.write(f'vt {v[0]} {v[1]} \n')

            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd albedo.png \n')

    @torch.no_grad()
    def eval(self, dataloader, save_result_dir, save_imges=True):
        all_render_out_frames = []
        all_render_rgb_frames = []
        tqdm_bar = tqdm(dataloader, total=len(dataloader), desc="Rendering...")
        for cam_data in tqdm_bar:
            w2c = cam_data['w2c'].to(self.device)
            cam_id = cam_data['cam_id'][0]
            intrinsic_matrix = cam_data['intrinsic']
            self.renderer.intrinsics = kal.render.camera.PinholeIntrinsics.from_focal(width=int(self.cfg.render.width*self.cfg.render.scale_img), height=int(self.cfg.render.height*self.cfg.render.scale_img), 
                                                                                      focal_x=intrinsic_matrix[0,0,0], focal_y=intrinsic_matrix[0,1,1], device=self.device)
            outputs = self.render(camera_transform=w2c)
            render_outs = utils.tensor2numpy(torch.cat([outputs['image'], 
                                                        outputs['depth'].repeat(1, 3, 1, 1), 
                                                        outputs['normals']], dim=3))
            all_render_rgb_frames.append(utils.tensor2numpy(outputs['image']))
            all_render_out_frames.append(render_outs)

            depth_normed = outputs['depth'].squeeze(0).cpu().numpy()
            depth_bins = np.histogram(depth_normed, bins=256)[0]  # Histogram of depth values
            depth_bins = depth_bins[1:]  # Ignore the first bin (0 value)
            depth_bins = depth_bins / np.sum(depth_bins)  # Normalize histogram
            depth_bins = np.cumsum(depth_bins)  # Cumulative sum of histogram
            threshold_index = np.searchsorted(depth_bins, 0.45)  # Find threshold index, 20% of lowest pixels
            threshold_value = threshold_index / 255.0  # Convert index to value
            eval_mask = (depth_normed > threshold_value)
            eval_mask = torch.from_numpy(eval_mask).unsqueeze(0).to(self.device)
           
            if save_imges:
                utils.save_tensor_image(outputs['image'], os.path.join(save_result_dir, cam_id + "_image.png"))
                utils.save_tensor_image(outputs['depth'].repeat(1, 3, 1, 1), os.path.join(save_result_dir, cam_id + "_depth.png"))
                utils.save_tensor_image(outputs['normals'], os.path.join(save_result_dir, cam_id + "_normals.png"))
                utils.save_tensor_image(outputs['uncolored_mask'], os.path.join(save_result_dir, cam_id + "_uncolored_mask.png"))
                utils.save_tensor_image(eval_mask, os.path.join(save_result_dir, cam_id + "_eval_mask.png"))
            
        # Needs newest imageio, imageio-ffmpeg
        if not Path(save_result_dir).exists():
            os.makedirs(save_result_dir, exist_ok=True)
        utils.save_video(np.stack(all_render_rgb_frames, axis=0), os.path.join(save_result_dir, "render_rgb_mesh.mp4"))
        print(f"Save RGB video to {save_result_dir}, all_render_rgb_frames: {len(all_render_rgb_frames)}, {all_render_rgb_frames[0].shape}")
        utils.save_video(np.stack(all_render_out_frames, axis=0), os.path.join(save_result_dir, "render_all.mp4"))
        print(f"Save all videos {len(all_render_out_frames)}, all_render_out_frames: {all_render_out_frames[0].shape}")

        # video of inpainted frames, has to be sorted by name
        inpaint_dir = Path(self.cfg.log.exp_path, "inpaint")

        all_paths = list(inpaint_dir.glob('*.png')) + list(inpaint_dir.glob('*.jpg'))
        sorted_paths = sorted(all_paths, key=natural_sort_key)
        inpainted_frames = []        
        for img_path in tqdm(sorted_paths):
            if 'inpaint_0' in img_path.name:
                img = utils.pil2tensor(Image.open(img_path), self.device)
                inpainted_frames.append(utils.tensor2numpy(img))
        try:
            utils.save_video(np.stack(inpainted_frames, axis=0), os.path.join(inpaint_dir, "inpainted_frames.mp4"))
        except Exception as e:
            print(f"Error saving inpainted frames video: {e}")


class DatasetCamera(Dataset):
    """
    Dataset class for loading known views from a transforms_train_gps.json file
    and converting GPS coordinates to pixel coordinates.
    """
    def __init__(
        self,
        data_path: str,
        cfg_dataEngine,    # config object of data_engine
        cfg_render,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        center: torch.Tensor = None,
        scale: torch.Tensor = None,
        load_img: bool = True,
    ):
        """
        - data_path: Path to transform matrix json file
        - dataset_cfg: config object with GEE parameters
        - transforms_file: Path to the transforms_train.json file pixel or gps
        - device: 
        - center: normalization center of coordinates
        - scale: normalize scale of coordinates
        - load_img: Load image if True

        transforms json file can have
        "meta": {
            "mode": "file", "waypoints",
            "waypoints": list of transformation matrices waypoints,
            "nbr_points": total number of points between first and last waypoints,
            "interpolation": 'linear' or 'smooth',
        }
        """
        self.dataset_cfg = cfg_dataEngine
        self.cfg_render = cfg_render
        self.device = device
        self.center = center
        self.scale = scale
        self.load_img = load_img
        
        home_path = getHomePath()
        
        # Load transform file
        self.transform_path = Path(home_path, data_path)
        with open(self.transform_path, 'r') as f:
            self.transforms = json.load(f)
        mode = None
        try:
            mode = self.transforms['meta']['mode']
            if mode == 'waypoints':
                self.gen_waypTransforms()

                # save transforms
                save_path = Path(self.transform_path.parent, self.transform_path.stem + "_all.json")
                transform_json = copy.deepcopy(self.transforms)     # deepcopy as it is mutable type and specify the address
                for key, value in transform_json.items():
                    # save in pixel coordinates
                    c2w = np.array(value['transform_matrix'])
                    gps_coords = c2w[:3, 3].tolist()
                    t_pix = camFormat.geoCoord2Open3Dpx(self.dataset_cfg, gps_coords)
                    c2w[:3, 3] = t_pix
                    value['transform_matrix'] = c2w.tolist()
                    transform_json[key]['transform_matrix'] = value['transform_matrix']
                with open(save_path, 'w') as f:
                    json.dump(transform_json, f, indent=2)
            else:
                raise ValueError(f"Invalid mode: {mode}")        
        except KeyError:
            print("INFO: No waypoint mode in transforms file meta")

        # Process transforms
        self.cam_data = {}
        self.cam_ids = []
        self.process_transforms()

    def gen_waypTransforms(self):
        """
        Generate transformation matrices from waypoints
        Args:
            waypoints: list of transformation matrices with gps coordinates and attitude
            nbr_points: total number of points between first and last waypoints
            interpolation: 'linear' or 'smooth'
        """
        waypoints = [torch.Tensor(self.transforms[key]["transform_matrix"]) for key in self.transforms['meta']['waypoints']]
        waypoints_fx = [self.transforms[key]["fx"] for key in self.transforms['meta']['waypoints']]
        waypoints_fy = [self.transforms[key]["fy"] for key in self.transforms['meta']['waypoints']]
        waypoints_cx = [self.transforms[key]["cx"] for key in self.transforms['meta']['waypoints']]
        waypoints_cy = [self.transforms[key]["cy"] for key in self.transforms['meta']['waypoints']]
        waypoints_k1 = [self.transforms[key]["k1"] for key in self.transforms['meta']['waypoints']]
        waypoints_k2 = [self.transforms[key]["k2"] for key in self.transforms['meta']['waypoints']]
        waypoints_p1 = [self.transforms[key]["p1"] for key in self.transforms['meta']['waypoints']]
        waypoints_p2 = [self.transforms[key]["p2"] for key in self.transforms['meta']['waypoints']]
        waypoints_k3 = [self.transforms[key]["k3"] for key in self.transforms['meta']['waypoints']]
        waypoints_k4 = [self.transforms[key]["k4"] for key in self.transforms['meta']['waypoints']]
        waypoints_k5 = [self.transforms[key]["k5"] for key in self.transforms['meta']['waypoints']]
        waypoints_k6 = [self.transforms[key]["k6"] for key in self.transforms['meta']['waypoints']]
        if isinstance(self.transforms['meta']['nbr_points'], numbers.Number):
            nbr_points_all = [self.transforms['meta']['nbr_points']]
        else:
            nbr_points_all = [int(nbr_points) for nbr_points in self.transforms['meta']['nbr_points']]
        interpolation = self.transforms['meta']['interpolation']

        assert len(waypoints) >= 2, f"At least two waypoints required, but got {waypoints}"
        assert interpolation in ['linear', 'smooth'], f"Invalid interpolation type: {interpolation}"

        # Generate waypoints
        self.transforms = {}
        if interpolation == 'linear':
            assert len(waypoints) == (len(nbr_points_all)+1), f"waypoints-1 should match length of nbr_points for linear interpolation, but got {len(waypoints)} and {len(nbr_points_all)}"
            for i in range(len(waypoints)-1):
                nbr_points = nbr_points_all[i]
                assert nbr_points > 1, f"At least two points required, but got {nbr_points}"
                start = waypoints[i]
                end = waypoints[i+1]
                trajectory = camFormat.linear_interpolate_transforms(start, end, nbr_points)
                fx_interpolation = np.linspace(waypoints_fx[i], waypoints_fx[i+1], nbr_points)
                fy_interpolation = np.linspace(waypoints_fy[i], waypoints_fy[i+1], nbr_points)
                cx_interpolation = np.linspace(waypoints_cx[i], waypoints_cx[i+1], nbr_points)
                cy_interpolation = np.linspace(waypoints_cy[i], waypoints_cy[i+1], nbr_points)
                k1_interpolation = np.linspace(waypoints_k1[i], waypoints_k1[i+1], nbr_points)
                k2_interpolation = np.linspace(waypoints_k2[i], waypoints_k2[i+1], nbr_points)
                p1_interpolation = np.linspace(waypoints_p1[i], waypoints_p1[i+1], nbr_points)
                p2_interpolation = np.linspace(waypoints_p2[i], waypoints_p2[i+1], nbr_points)
                k3_interpolation = np.linspace(waypoints_k3[i], waypoints_k3[i+1], nbr_points)
                k4_interpolation = np.linspace(waypoints_k4[i], waypoints_k4[i+1], nbr_points)
                k5_interpolation = np.linspace(waypoints_k5[i], waypoints_k5[i+1], nbr_points)
                k6_interpolation = np.linspace(waypoints_k6[i], waypoints_k6[i+1], nbr_points)
                for j in range(trajectory.shape[0]):
                    self.transforms[f"wayp_{i}_{j}"] = {}
                    self.transforms[f"wayp_{i}_{j}"]["cam_id"] = f"wayp_{i}_{j}"
                    self.transforms[f"wayp_{i}_{j}"]["file_path"] = f"./images/wayp_{i}_{j}_inpaint_0.png"
                    self.transforms[f"wayp_{i}_{j}"]["mask_path"] = f"./mask/wayp_{i}_{j}_GSmask.png"
                    self.transforms[f"wayp_{i}_{j}"]["depth_path"] = f"./depth/wayp_{i}_{j}_depth.png"
                    self.transforms[f"wayp_{i}_{j}"]["wid"] = self.cfg_render.width
                    self.transforms[f"wayp_{i}_{j}"]["hei"] = self.cfg_render.height
                    self.transforms[f"wayp_{i}_{j}"]["cx"] = cx_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["cy"] = cy_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["fx"] = fx_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["fy"] = fy_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k1"] = k1_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k2"] = k2_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["p1"] = p1_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["p2"] = p2_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k3"] = k3_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k4"] = k4_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k5"] = k5_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["k6"] = k6_interpolation[j]
                    self.transforms[f"wayp_{i}_{j}"]["transform_matrix"] = trajectory[j, :, :]

        elif interpolation == 'smooth':
            assert len(nbr_points_all) == 1, "Only one nbr_points required for smooth interpolation"
            nbr_points = nbr_points_all[0]
            waypoints = np.stack(waypoints, axis=0)
            trajectory = camFormat.smooth_interpolate_waypoints(waypoints, nbr_points)
            fx_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_fx, axis=0), nbr_points)
            fy_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_fy, axis=0), nbr_points)
            cx_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_cx, axis=0), nbr_points)
            cy_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_cy, axis=0), nbr_points)
            k1_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k1, axis=0), nbr_points)
            k2_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k2, axis=0), nbr_points)
            p1_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_p1, axis=0), nbr_points)
            p2_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_p2, axis=0), nbr_points)
            k3_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k3, axis=0), nbr_points)
            k4_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k4, axis=0), nbr_points)
            k5_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k5, axis=0), nbr_points)
            k6_interpolation = camFormat.smooth_interpolate_values(np.stack(waypoints_k6, axis=0), nbr_points)
            for i in range(trajectory.shape[0]):
                self.transforms[f"wayp_{i}"] = {}
                self.transforms[f"wayp_{i}"]["cam_id"] = f"wayp_{i}"
                self.transforms[f"wayp_{i}"]["file_path"] = f"./images/wayp_{i}_inpaint_0.png"
                self.transforms[f"wayp_{i}"]["mask_path"] = f"./mask/wayp_{i}_GSmask.png"
                self.transforms[f"wayp_{i}"]["depth_path"] = f"./depth/wayp_{i}_depth.png"
                self.transforms[f"wayp_{i}"]["wid"] = self.cfg_render.width
                self.transforms[f"wayp_{i}"]["hei"] = self.cfg_render.height
                self.transforms[f"wayp_{i}"]["cx"] = cx_interpolation[i]
                self.transforms[f"wayp_{i}"]["cy"] = cy_interpolation[i]
                self.transforms[f"wayp_{i}"]["fx"] = fx_interpolation[i]
                self.transforms[f"wayp_{i}"]["fy"] = fy_interpolation[i]
                self.transforms[f"wayp_{i}"]["k1"] = k1_interpolation[i]
                self.transforms[f"wayp_{i}"]["k2"] = k2_interpolation[i]
                self.transforms[f"wayp_{i}"]["p1"] = p1_interpolation[i]
                self.transforms[f"wayp_{i}"]["p2"] = p2_interpolation[i]
                self.transforms[f"wayp_{i}"]["k3"] = k3_interpolation[i]
                self.transforms[f"wayp_{i}"]["k4"] = k4_interpolation[i]
                self.transforms[f"wayp_{i}"]["k5"] = k5_interpolation[i]
                self.transforms[f"wayp_{i}"]["k6"] = k6_interpolation[i]
                self.transforms[f"wayp_{i}"]["transform_matrix"] = trajectory[i, :, :]

        else:
            raise ValueError(f"Invalid interpolation type: {interpolation}")

        
    def process_transforms(self):
        """
        Process the transforms file:
        1. Extract relevant camera parameters
        2. Convert GPS coordinates to pixel coordinates
        3. Normalize pixel coordinates and transform to kaolin format
        """
        for cam_id, cam in self.transforms.items():
            # Checks
            if not isinstance(cam, dict) or 'transform_matrix' not in cam:
                continue
            
            # Create camera data entry
            camera_data = {
                'cam_id': cam.get('cam_id', cam_id),
                'file_path': cam.get('file_path', ''),
                'width': int(cam.get('wid', 768)*self.cfg_render.scale_img),
                'height': int(cam.get('hei', 512)*self.cfg_render.scale_img),
                'cx': float(cam.get('cx', 384.0))*self.cfg_render.scale_img,
                'cy': float(cam.get('cy', 256.0))*self.cfg_render.scale_img,
                'fx': float(cam.get('fx', 460.8))*self.cfg_render.scale_img,
                'fy': float(cam.get('fy', 512.0))*self.cfg_render.scale_img,
                'distortion': [
                    float(cam.get('k1', 0)), 
                    float(cam.get('k2', 0)), 
                    float(cam.get('p1', 0)), 
                    float(cam.get('p2', 0)),
                    float(cam.get('k3', 0)), 
                    float(cam.get('k4', 0)), 
                    float(cam.get('k5', 0)), 
                    float(cam.get('k6', 0))
                ],
            }
            
            # Get transform matrix
            c2w = np.array(cam['transform_matrix'])
            assert c2w.shape == (4, 4)
            
            # Extract GPS coordinates from transform matrix
            gps_coords = c2w[:3, 3].tolist()
            if self.cfg_render.gps:            
                assert gps_coords[0] > -180 and gps_coords[0] < 180 
                assert gps_coords[1] > -90 and gps_coords[1] < 90

                pix_coords = camFormat.geoCoord2Open3Dpx(self.dataset_cfg, gps_coords)
                c2w[:3, 3] = pix_coords

            # Normalize pixel coordinates
            if self.center is not None and self.scale is not None:
                c2w[:3, 3] = (c2w[:3, 3] - np.array(self.center)) * np.array(self.scale)

            c2w = camFormat.opengl_to_kao(c2w)        # my renderer
            w2c = np.linalg.inv(c2w)
                       
            camera_data['c2w'] = c2w
            camera_data['w2c'] = w2c
            
            # Add to camera data dictionary
            if self.load_img:
                try:
                    camera_file_path = Path(getHomePath().parent, camera_data['file_path'])
                    image = Image.open(camera_file_path)
                    self.cam_ids.append(cam_id)
                    self.cam_data[cam_id] = camera_data
                except Exception as e:
                    print(f"\nWARNING: No image from {cam_id}: {e}")
            else:
                self.cam_ids.append(cam_id)
                self.cam_data[cam_id] = camera_data
            
    def __len__(self):
        return len(self.cam_ids)
    
    def __getitem__(self, idx):
        """
        Get the camera data and image for a given index
        Args:
            idx: Index of the camera to return
        Returns:
            Dictionary with camera data and image
        """
        cam_id = self.cam_ids[idx]
        camera_data = self.cam_data[cam_id]
        
        # Load image if file path exists and is valid
        image = None
        camera_file_path = Path(getHomePath().parent, camera_data['file_path'])
        if self.load_img and camera_file_path.exists():
            image = np.array(Image.open(camera_file_path).convert('RGB').resize((camera_data['width'], camera_data['height']), resample=Image.BILINEAR))
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)            # Convert to RGB if grayscale
            image = torch.tensor(image.astype(np.float32) / 255.0, dtype=torch.float32).permute(2, 0, 1)  # (B)CHW and to float [0, 1]
        else:
            image = torch.zeros((3, camera_data['height'], camera_data['width']), dtype=torch.float32)  # Placeholder if image not found

        return {
            'cam_id': cam_id,
            'c2w': torch.tensor(camera_data['c2w'], dtype=torch.float32),
            'w2c': torch.tensor(camera_data['w2c'], dtype=torch.float32),
            'intrinsic': torch.tensor([
                [camera_data['fx'], 0, camera_data['cx']],
                [0, camera_data['fy'], camera_data['cy']],
                [0, 0, 1]
            ], dtype=torch.float32),
            'image': image,             # image can be None, if no image found
            'width': camera_data['width'],
            'height': camera_data['height']
        }
        
    def get_camera_data(self):
        """Return camera data dictionary"""
        return self.cam_data
    
    def get_camera_ids(self):
        """Return list of camera IDs"""
        return self.cam_ids

class Renderer:
    def __init__(self, cfg, mesh_face_num, device, save_debug=False):
        self.device = device
        self.save_debug = save_debug
        self.cfg = cfg
        self.render_cfg = cfg.render
        self.batch_size = self.render_cfg.batch_size_internal
        self.fox = self.render_cfg.width*self.cfg.render.scale_img / 2       # default
        self.foy = self.render_cfg.height*self.cfg.render.scale_img / 2       # default
        self.render_angle_thres = self.render_cfg.render_angle_thres
        self.calcu_uncolored_mode = self.render_cfg.calcu_uncolored_mode
        self.img_size = (int(self.render_cfg.width*self.cfg.render.scale_img), int(self.render_cfg.height*self.cfg.render.scale_img))
        self.interpolation_mode = self.render_cfg.texture_interpolation_mode
        assert self.interpolation_mode in ['nearest', 'bilinear', 'bicubic'], \
            f'no interpolation mode: {self.interpolation_mode}'
        assert self.calcu_uncolored_mode in ['WarpGrid', 'FACE_ID', 'DIFF', 'HSV'], \
            f'no uncolored mask caculation mode: {self.calcu_uncolored_mode}'

        self.intrinsics = kal.render.camera.PinholeIntrinsics.from_focal(width=self.img_size[0], height=self.img_size[1], focal_x=self.fox, focal_y=self.foy, device=device)
        self.mesh_face_num = mesh_face_num + 1
        self.seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)

        try:
            import nvdiffrast
            # self.backend = 'nvdiffrast'
            self.backend = 'cuda'
        except ImportError:
            self.backend = 'cuda'
        print(f"Using renderer backend: {self.backend}")

    def clear_seen_faces(self):
        self.seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)

    def normalize_depth(self, depth_map):
        #assert depth_map.max() <= 0.0, 'depth map should be negative'  # why was this ones used?
        object_mask = depth_map != 0
        min_val = 0.5
        try:
            depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                    depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        except:
            print("\nWARNING: Depth map normalization failed")
        return depth_map

    def UV_pos_render(self, verts, faces, uv_face_attr, texture_dims,):
        """
        :param verts: (V, 3)
        :param faces: (F, 3)
        :param uv_face_attr: shape (1, F, 3, 2), range [0, 1]
        :param camera_transform: (3, 4)
        :param view_target:
        :param texture_dims:
        :return:
        """
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        mesh_out_of_range = False
        if x.min() < -1 or x.max() > 1 or y.min() < -1 or y.max() > 1 or z.min() < -1 or z.max() > 1:
            mesh_out_of_range = True

        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_world = face_vertices_world.to(self.device)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=self.device)
        uv_position, face_idx = kal.render.mesh.rasterize(texture_dims[0], texture_dims[1], face_vertices_z, uv_face_attr * 2 - 1,
                                                face_features=face_vertices_world, backend=self.backend)
        uv_position = torch.clamp(uv_position, -1, 1)

        uv_position = uv_position / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position
 
    def forward_texturing_render(self, verts, faces, uv_face_attr, camera_transform,
                                 view_target, view_target_mask, uncolored_mask, texture_dims, render_cache=None):
        """
        :param verts: (V, 3)
        :param faces: (F, 3)
        :param uv_face_attr: shape (1, F, 3, 2), range [0, 1]
        :param camera_transform: (3, 4)
        :param view_target:
        :param texture_dims:
        :param render_cache:
        :return: next_texture_map, next_texture_update_area, normal_map
        """
        texture_h, texture_w = texture_dims[0], texture_dims[1]
        view_target_h, view_target_w = self.img_size[1], self.img_size[0]
        face_vertices_camera, face_vertices_image, face_normals = self.prepare_vertices(
            verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics,
            camera_transform=camera_transform)
        face_vertices_z = face_vertices_camera[:, :, :, -1].contiguous()

        face_vertices_image = face_vertices_image.contiguous()
        uv_face_attr = uv_face_attr.contiguous()

        if render_cache is None:
            if _has_pytorch3d:
                face_vertices_camera, face_vertices_image, _ = self.prepare_vertices_ndc(
                    verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics, camera_transform=camera_transform)
                depth, face_idx, uv_features = self.render_depth_plus_pytorch3d(face_vertices_image, face_vertices_camera, view_target_h, view_target_w, device=self.device, uv_faces=uv_face_attr)
            else:
                depth, face_idx, uv_features = self.render_depth_plus_batched(face_vertices_image, face_vertices_camera, view_target_h, view_target_w, device=self.device, uv_faces=uv_face_attr, batch_size=self.batch_size)
        else:
            depth, face_idx_img = render_cache['depth_map'], render_cache['face_idx']
        valid_face_idx = list(np.unique(face_idx_img.data.cpu().numpy()))

        seen_cons_from_view = torch.zeros(1, self.mesh_face_num - 1, device=self.device)
        seen_cons_from_view[:, valid_face_idx] = 1 
        seen_cons_from_view = seen_cons_from_view.contiguous()

        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=verts.size(0), faces=faces.cpu(), face_normals=face_normals[0].cpu(),)  # V,3
        vertex_normals = torch.from_numpy(vertex_normals).unsqueeze(0).float().to(self.device)
        face_vertices_normals = kal.ops.mesh.index_vertices_by_faces(vertex_normals, faces.to(self.device))

        face_vertices_normal_z = face_vertices_normals[:, :, :, 2:3].contiguous()  # [1, F, 3, n]
        if seen_cons_from_view.sum() == 0:
            print("\nWARNING: No seen valid faces faces")
            normal_map = torch.zeros(1, texture_h, texture_w, 1, device=self.device)
        else:
            normal_map, normal_map_idx = kal.render.mesh.rasterize(texture_h, texture_w, face_vertices_z,
                                                    uv_face_attr * 2 - 1, face_features=face_vertices_normal_z,
                                                    valid_faces=seen_cons_from_view.to(torch.bool), backend=self.backend)

        cos_thres = np.cos(self.render_angle_thres / 180 * np.pi)
        normal_map[normal_map < cos_thres] = 0
        valid_face_mask = normal_map.clone()
        valid_face_mask = self.normalize_depth(valid_face_mask)
        valid_face_mask[valid_face_mask > 0] = 1

        uv3d = torch.zeros_like(face_vertices_z, device=self.device)
        if seen_cons_from_view.sum() == 0:
            uv_features_inv = torch.zeros(1, texture_h, texture_w, 2, device=self.device)
        else:
            uv_features_inv, face_idx_inv = kal.render.mesh.rasterize(texture_h, texture_w, uv3d, uv_face_attr * 2 - 1, face_features=face_vertices_image, 
                                                                      valid_faces = seen_cons_from_view.to(torch.bool), backend=self.backend)
        # mapping
        uv_features_inv = (uv_features_inv + 1) / 2
        # apply mask to view_target
        view_target = view_target
        next_texture_map = kal.render.mesh.texture_mapping(uv_features_inv, view_target, mode=self.interpolation_mode)
        next_texture_map_mask = kal.render.mesh.texture_mapping(uv_features_inv, view_target_mask.float(), mode=self.interpolation_mode)
        next_texture_map_mask = (next_texture_map_mask > 0.5).float()
        

        # TODO: replace background UV map pixels (black) around islands with valid neighbor, or local average 
        # def random_selection_conv2d(input_tensor, kernel_size):
        #     batch_size, channels, height, width = input_tensor.size()
        #     padding = kernel_size // 2

        #     # Pad the input tensor to handle edges
        #     padded_input = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

        #     # Use unfold to extract sliding local blocks
        #     unfolded = padded_input.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        #     unfolded = unfolded.contiguous().view(batch_size, channels, height, width, -1)

        #     # Randomly select one value from each block
        #     random_indices = torch.randint(0, kernel_size*kernel_size, (batch_size, channels, height, width), device=self.device)
        #     selected_values = unfolded.gather(4, random_indices.unsqueeze(-1)).squeeze(-1)

        #     return selected_values
        
        # def random_valid_neighbor(image_tensor, val_mask):



        # # Define a function to replace black pixels
        # kernelsize = 7  # odd number
        # def replace_black_pixels(image_tensor, kernelsize=7):
        #     image_tensor = image_tensor.permute(0, 3, 1, 2) # Change to (B, C, H, W)
        #     # Create a mask for black pixels (assuming black is [0, 0, 0])
        #     black_mask = (image_tensor.sum(dim=1, keepdim=True) == 0).float()

        #     # Define a kernel for neighboring pixels (e.g., 3x3 kernel)
        #     kernel = torch.ones(kernelsize, kernelsize).to(self.device)
        #     kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

        #     # Apply convolution to the mask to count the number of valid (non-black) neighbors
        #     val_mask = torch.ones_like(black_mask) - black_mask
        #     # valid_neighbors = F.conv2d(val_mask, kernel, padding=kernel.shape[2]//2)

        #     # Apply convolution to each channel separately
        #     # kernel = torch.ones(image_tensor.shape[1], 1, kernelsize, kernelsize).to(self.device)
        #     # neighbor_sum = F.conv2d(image_tensor, kernel, padding=kernel.shape[2]//2, groups=image_tensor.shape[1])

        #     # Compute the average of valid neighbors
        #     # neighbor_avg = neighbor_sum / valid_neighbors.clamp(min=1)

        #     one_neighbor = random_valid_neighbor(image_tensor, val_mask)

        #     # Replace black pixels with the average of neighboring pixels
        #     # image_tensor = image_tensor * (1 - black_mask) + neighbor_avg * black_mask
        #     image_tensor = image_tensor * (1 - black_mask) + one_neighbor * black_mask

        #     if self.save_debug:
        #         utils.save_tensor_image(black_mask, Path(self.cfg.log.exp_path, "debug_forTextRender", "forward_texturing_render", "replace_black_pixels.png").as_posix())

        #     return image_tensor.permute(0, 2, 3, 1)   # Change back to (B, H, W, C)
         
        # # Replace black pixels in the next_texture_map
        # next_texture_map = replace_black_pixels(next_texture_map)
        # valid_face_mask = replace_black_pixels(valid_face_mask)
        # next_texture_map_mask = F.conv2d(next_texture_map_mask.permute(0, 3, 1, 2), torch.ones((1, 1, kernelsize, kernelsize), device=self.device), padding=kernelsize//2)
        # next_texture_map_mask[next_texture_map_mask > 0.1] = 1
        # next_texture_map_mask[next_texture_map_mask <= 0.1] = 0
        # next_texture_map_mask = next_texture_map_mask.permute(0, 2, 3, 1)
        

        next_texture_map = next_texture_map * valid_face_mask * next_texture_map_mask
        next_texture_update_area = kal.render.mesh.texture_mapping(uv_features_inv, uncolored_mask.repeat(1,3,1,1), mode=self.interpolation_mode)
        next_texture_update_area[next_texture_update_area > 0] = 1
        next_texture_update_area = next_texture_update_area * valid_face_mask * next_texture_map_mask

        normal_map = normal_map.permute(0, 3, 1, 2)
        next_texture_map = next_texture_map.permute(0, 3, 1, 2,)
        next_texture_update_area = next_texture_update_area.permute(0, 3, 1, 2,)

        if self.save_debug:
            save_folder = Path(self.cfg.log.exp_path, "debug_forTextRender", "forward_texturing_render")
            save_folder.mkdir(parents=True, exist_ok=True)
            uv_features_inv = uv_features_inv.permute(0, 3, 1, 2)
            uv_features_inv = torch.cat([uv_features_inv, torch.zeros_like(uv_features_inv)], dim=1)
            depth = self.normalize_depth(depth)
            normal_map = self.normalize_depth(normal_map)
            utils.save_tensor_image(depth.permute(0,3,1,2), Path(save_folder, "depth.png").as_posix())
            utils.save_tensor_image(normal_map, Path(save_folder, "uv_normal_normed.png").as_posix())
            utils.save_tensor_image(valid_face_mask.permute(0,3,1,2), Path(save_folder, "uv_valid_face_mask.png").as_posix())
            utils.save_tensor_image(uv_features_inv, Path(save_folder, "uv_features_inv.png").as_posix())
            utils.save_tensor_image(next_texture_map, Path(save_folder, "uv_next_texture.png").as_posix())
            utils.save_tensor_image(next_texture_update_area, Path(save_folder, "uv_next_texture_update_area.png").as_posix())
            utils.save_tensor_image(view_target, Path(save_folder, "view_target.png").as_posix())
            utils.save_tensor_image(view_target_mask, Path(save_folder, "view_target_mask.png").as_posix())
            utils.save_tensor_image(next_texture_map_mask.permute(0,3,1,2), Path(save_folder, "next_texture_map_mask.png").as_posix())
        
        return next_texture_map, next_texture_update_area, normal_map


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, camera_transform,
                                   render_cache=None, img_size=None, texture_default_color=[0.8, 0.1, 0.8]):
        img_size = self.img_size if img_size is None else img_size

        if render_cache is None:  
            # uv_face_attr shape ([1, 435766, 3, 2]) in [0, 1]
            # face_vertices_camera shape ([1, 435766, 3, 3])
            # face_vertices_image shape ([1, 435766, 3, 2])
            # uv_features shape ([1, 2048, 3072, 2])
            # depth_map shape ([1, 2048, 3072, 1])
            
            # rasterize depth plus uv_features
            if _has_pytorch3d:
                face_vertices_camera, face_vertices_image, face_normals = self.prepare_vertices_ndc(verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics, camera_transform=camera_transform)
                depth, face_idx, uv_features = self.render_depth_plus_pytorch3d(face_vertices_image, face_vertices_camera[:, :, :, -1], img_size[1], img_size[0], device=self.device, uv_faces=uv_face_attr)
            else:
                face_vertices_camera, face_vertices_image, face_normals = self.prepare_vertices(verts.to(self.device), faces.to(self.device), intrinsics=self.intrinsics, camera_transform=camera_transform)
                depth, face_idx, uv_features = self.render_depth_plus_batched(face_vertices_image, face_vertices_camera[:, :, :, -1], img_size[1], img_size[0], device=self.device, uv_faces=uv_face_attr, batch_size=self.batch_size)

            depth_map = self.normalize_depth(depth)
        else:
            print("INFO: Using cached render cache")
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], \
                                                             render_cache['face_idx'], render_cache['depth_map']
        mask = (face_idx > 0).float()[..., None]
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
        uncolored_mask = None

        # inpaint small holes using https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html and https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
        if self.render_cfg.inpaint_openCV:
            image_features_uint8 = image_features.cpu().detach().numpy() * 255
            image_features_uint8 = image_features_uint8.astype(np.uint8)
            hsv_img = cv2.cvtColor(image_features_uint8[0], cv2.COLOR_RGB2HSV)
            
            # Define the range for red color in HSV
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8), connectivity=8)
            connected_size_thresh = 300

            inpaint_mask = np.zeros_like(red_mask, dtype=np.uint8)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < connected_size_thresh:
                    inpaint_mask[labels == label] = 255

            image_features_uint8[0] = cv2.inpaint(image_features_uint8[0], inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # save inpainting mask
            if self.save_debug:
                print(f"num_labels: {num_labels}")
                rand_num = np.random.randint(0, 10)
                save_folder = Path(self.cfg.log.exp_path, "debug_inpaintCV2")
                save_folder.mkdir(parents=True, exist_ok=True)
                inpaint_mask = cv2.cvtColor(inpaint_mask, cv2.COLOR_GRAY2RGB)
                utils.save_tensor_image(torch.tensor(inpaint_mask/255.0).permute(2, 0, 1).unsqueeze(0), Path(save_folder, f"{rand_num}_inpaint_mask.png").as_posix())
                utils.save_tensor_image(torch.tensor(image_features_uint8/255.0).permute(0, 3, 1, 2), Path(save_folder, f"{rand_num}_image_features_inpainted.png").as_posix())
                utils.save_tensor_image(torch.tensor(image_features).permute(0, 3, 1, 2), Path(save_folder, f"{rand_num}_image_features.png").as_posix())

            image_features = torch.tensor(image_features_uint8).to(self.device) / 255.0

        if render_cache is None:
            if self.calcu_uncolored_mode == "WarpGrid":
                texture_map_diff = (texture_map.detach() - torch.tensor(texture_default_color).view(1, 3, 1, 1).
                                    to(self.device)).abs().sum(axis=1)
                uncolored_texture_map = (texture_map_diff < self.render_cfg.filter_background).float().unsqueeze(0)    # mask threshold: dft 0.1

                uncolored_mask = kal.render.mesh.texture_mapping(uv_features, uncolored_texture_map,
                                                                 mode=self.interpolation_mode)
            elif self.calcu_uncolored_mode == "FACE_ID":
                check = (face_idx > -1) & (face_idx < self.mesh_face_num - 1)
                next_seen_faces = torch.zeros(1, self.mesh_face_num, 1, device=self.device)
                next_seen_faces[:, face_idx[check].view(-1)] = 1.0
                next_seen_faces = ((next_seen_faces - self.seen_faces) > 0).float()
                uncolored_mask = next_seen_faces[0][face_idx, :]
                self.seen_faces = ((next_seen_faces + self.seen_faces) > 0).float()
            elif self.calcu_uncolored_mode == "DIFF":
                diff = (image_features.permute(0, 3, 1, 2).clamp(0, 1).detach() - torch.tensor(texture_default_color).view(1, 3, 1, 1).to(self.device)).abs().sum(axis=1)         # in range [0, 3] if default_color = 0, typical 1.5
                uncolored_mask = (diff < self.render_cfg.filter_background).float().unsqueeze(-1).clamp(0, 1).detach()    # mask threshold: dft 0.1
            elif self.calcu_uncolored_mode == "HSV":
                image_features_uint8 = image_features.cpu().detach().numpy() * 255
                image_features_uint8 = image_features_uint8.astype(np.uint8)
                hsv_img = cv2.cvtColor(image_features_uint8[0], cv2.COLOR_RGB2HSV)
            
                # Define the range for red color in HSV, TODO: not hardcoded red --> configuration or automatic
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([15, 255, 255])
                lower_red2 = np.array([165, 50, 50])
                upper_red2 = np.array([180, 255, 255])

                red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
                red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
                uncolored_mask = torch.tensor(cv2.bitwise_or(red_mask1, red_mask2)).unsqueeze(0).unsqueeze(-1).to(self.device) / 255.0
            
            if self.render_cfg.mask_sky:        # mask the sky
                uncolored_mask = (uncolored_mask * mask).permute(0, 3, 1, 2)
            uncolored_mask = uncolored_mask.permute(0, 3, 1, 2)

        image_features = image_features * mask + 1 * (1 - mask)           # mask out the sky in image
        normals_image = face_normals[0][face_idx, :]

        render_cache = {'uv_features': uv_features, 'face_normals': face_normals, 'face_idx': face_idx, 'depth_map': depth_map}
        return image_features.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2), \
               mask.permute(0, 3, 1, 2), uncolored_mask, normals_image.permute(0, 3, 1, 2), render_cache

    def prepare_vertices(self, vertices, faces, intrinsics, camera_transform):
        padded_vertices = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.)
        if len(camera_transform.shape) == 2:
            camera_transform = camera_transform.unsqueeze(0)
        if camera_transform.shape[1] == 4:        # want 3x4
            camera_transform = camera_transform[:, :3, :].transpose(1, 2)
        vertices_camera = (padded_vertices @ camera_transform)

        vertices_image = intrinsics.transform(vertices_camera)[:, :, :2]

        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
        face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, faces)
        face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        return face_vertices_camera, face_vertices_image, face_normals
    
    def prepare_vertices_ndc(self, vertices, faces, intrinsics, camera_transform):
        """
        Transforms vertices to clip space and then NDC, another method than projection with intrinsic matrix, used to scale with custom 3*focal_x

        Args:
            vertices: 3D vertices in world space.
            faces: Face indices.
            intrinsics: The camera's projection matrix (4x4).
            camera_transform: The camera's transformation matrix (3x4).

        Returns:
            face_vertices_camera: Vertices in camera space.
            face_vertices_ndc: Vertices in NDC space.
            face_normals: Face normals.
        """
        padded_vertices = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.)
        if len(camera_transform.shape) == 2:
            camera_transform = camera_transform.unsqueeze(0)
        if camera_transform.shape[1] == 4:        # want 3x4
            camera_transform = camera_transform[:, :3, :].transpose(1, 2)
        vertices_camera = (padded_vertices @ camera_transform)

        # Apply projection matrix (camera_transform to clip space)
        near = 0.01
        far = 1.5
        cx = self.img_size[0] / 2 + intrinsics.x0
        cy = self.img_size[1] / 2 + intrinsics.y0
        # 3:2 aspect ratio for focal length
        projection_matrix = torch.tensor([[3*intrinsics.focal_x/self.img_size[0], 0, 2*cx/self.img_size[0]-1, 0],
                                          [0, 2*intrinsics.focal_y/self.img_size[1], 2*cy/self.img_size[1]-1, 0],
                                          [0, 0, (far+near)/(far-near), -far*near/(far-near)],
                                          [0, 0, 1, 0]], device=vertices.device)
        
        vertices_camera_pad = torch.nn.functional.pad(vertices_camera, (0, 1), mode='constant', value=1.)
        vertices_camera_pad = vertices_camera_pad.unsqueeze(0)
        vertices_clip = torch.matmul(vertices_camera_pad, projection_matrix.transpose(0, 1))
        
        # Perform W-division (clip space to NDC)
        vertices_clip = vertices_clip.squeeze(0)        # (B, N, 4)
        vertices_ndc = vertices_clip[:, :, :3] / (vertices_clip[:, :, 3].unsqueeze(2) + 1e-6)

        # Get face vertices in camera and NDC space
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
        face_vertices_ndc = kal.ops.mesh.index_vertices_by_faces(vertices_ndc, faces)
        face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

        return face_vertices_camera, face_vertices_ndc[:, :, :, :2], face_normals
    
    def render_depth_plus_batched(self, face_vertices_image, face_vertices_z, view_target_h, view_target_w, device, uv_faces=None, batch_size=1024):
        """
        Rasterize a depth map from the given face vertices and z-values."
        Args:
        - face_vertices_image: Tensor (1, num_faces, 3, 2) with 2D vertices of each face
        - face_vertices_z: Tensor (1, num_faces, 3) with z-values of each face
        - view_target_h: Height of the target image
        - view_target_w: Width of the target image
        - device: Device to use
        - uv_faces: Tensor (1, num_faces, 3, 2) with uv coordinates of each face
        - batch_size: Number of faces to process in each batch
        - render_scale: Scale of the rendered image
        Returns:
        - depth_image: Tensor (1, view_target_h, view_target_w, 1)
        - face_idx_buffer: Tensor (1, view_target_h, view_target_w)
        - uv_image: Tensor (1, view_target_h, view_target_w, 2)
        """
        render_h = view_target_h
        render_w = view_target_w

        # Initialize the depth image and face index buffer
        empty_depth_image = torch.zeros(1, render_h, render_w, 1, device=device)
        empty_uv_face_image = torch.zeros(1, render_h, render_w, 2, device=device)
        face_idx_img = torch.zeros(1, render_h, render_w, device=device, dtype=torch.long)

        # Total number of triangles
        num_faces = len(face_vertices_image[0])

        # Process triangles in batches
        tqdm_bar = tqdm(range(0, num_faces, batch_size), total=(num_faces + batch_size - 1) // batch_size, desc="Rasterize image")
        for batch_start in tqdm_bar:
            batch_end = min(batch_start + batch_size, num_faces)
            batch_indices = torch.arange(batch_start, batch_end, device=device)
            batch_vertices_2d = face_vertices_image[0, batch_start:batch_end]  # Shape: (batch_size, 3, 2)
            batch_z_values = face_vertices_z[0, batch_start:batch_end]  # Shape: (batch_size, 3)
            if uv_faces is not None:
                batch_uv_values = uv_faces[0, batch_start:batch_end]  # Shape: (batch_size, 3, 2)
            else:
                batch_uv_values = None

            # Filter out triangles with invalid z-values
            valid_mask = (batch_z_values > 0).any(dim=-1)
            if not valid_mask.any():
                continue
            batch_vertices_2d = batch_vertices_2d[valid_mask]  # Shape: (valid_batch_size, 3, 2)
            batch_z_values = batch_z_values[valid_mask]
            batch_indices = batch_indices[valid_mask]
            if uv_faces is not None:
                batch_uv_values = batch_uv_values[valid_mask]

            # Flip v-coordinate
            batch_vertices_2d = batch_vertices_2d.clone()
            batch_vertices_2d[:, :, 1] = -batch_vertices_2d[:, :, 1]

            # Convert normalized coordinates ([-1, 1]) to pixel coordinates
            unnormalized_vertices_2d = batch_vertices_2d.clone()
            unnormalized_vertices_2d[:, :, 0] = (((batch_vertices_2d[:, :, 0] + 1) / 2) * render_w)
            unnormalized_vertices_2d[:, :, 1] = (((batch_vertices_2d[:, :, 1] + 1) / 2) * render_h)

            # Compute bounding boxes for all triangles in the batch
            unnormalized_vertices_2d_int = unnormalized_vertices_2d.to(torch.int32)
            u_min = unnormalized_vertices_2d_int[:, :, 0].min(dim=1).values
            u_max = unnormalized_vertices_2d_int[:, :, 0].max(dim=1).values
            v_min = unnormalized_vertices_2d_int[:, :, 1].min(dim=1).values
            v_max = unnormalized_vertices_2d_int[:, :, 1].max(dim=1).values

            # Skip triangles that are outside the image or have extreme coordinates
            outside_mask = (
                (u_max < 0) | (u_min >= render_w) | (v_max < 0) | (v_min >= render_h) |
                (u_max > render_w * 1.2) | (u_min < -render_w * 0.2) |
                (v_max > render_h * 1.2) | (v_min < -render_h * 0.2)
            )
            if outside_mask.all():
                continue
            batch_vertices_2d = batch_vertices_2d[~outside_mask]
            batch_z_values = batch_z_values[~outside_mask]
            batch_indices = batch_indices[~outside_mask]
            u_min, u_max = u_min[~outside_mask], u_max[~outside_mask]
            v_min, v_max = v_min[~outside_mask], v_max[~outside_mask]
            if uv_faces is not None:
                batch_uv_values = batch_uv_values[~outside_mask]

            if batch_vertices_2d.shape[0] == 0:
                continue

            # Clamp to image size
            u_min = torch.clamp(u_min, 0, render_w - 1)
            u_max = torch.clamp(u_max, 0, render_w - 1)
            v_min = torch.clamp(v_min, 0, render_h - 1)
            v_max = torch.clamp(v_max, 0, render_h - 1)

            # Convert to float for barycentric computation
            unnormalized_vertices_2d = unnormalized_vertices_2d[~outside_mask].to(torch.float32)

            # Process each triangle in the batch
            for i in range(unnormalized_vertices_2d.shape[0]):
                u_min_i, u_max_i = int(u_min[i]), int(u_max[i])
                v_min_i, v_max_i = int(v_min[i]), int(v_max[i])

                # Skip if the bounding box is too large
                bbox_area = (u_max_i - u_min_i + 1) * (v_max_i - v_min_i + 1)
                if bbox_area > (render_h * render_w * 0.2):
                    print(f"WARNING: Large bounding box for triangle {batch_indices[i]}: u_min={u_min_i}, u_max={u_max_i}, v_min={v_min_i}, v_max={v_max_i}")
                    continue

                # Vectorized rasterization for the current triangle
                u_coords, v_coords = torch.meshgrid(
                    torch.arange(u_min_i, u_max_i + 1, device=device),
                    torch.arange(v_min_i, v_max_i + 1, device=device),
                    indexing='xy'
                )
                pixels = torch.stack([u_coords, v_coords], dim=-1).float()

                # Compute barycentric coordinates
                v0, v1, v2 = unnormalized_vertices_2d[i, 0], unnormalized_vertices_2d[i, 1], unnormalized_vertices_2d[i, 2]
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                v0p = pixels - v0.view(1, 1, 2)
                d00 = torch.dot(v0v1, v0v1)
                d01 = torch.dot(v0v1, v0v2)
                d11 = torch.dot(v0v2, v0v2)
                d20 = torch.sum(v0p * v0v1.view(1, 1, 2), dim=-1)
                d21 = torch.sum(v0p * v0v2.view(1, 1, 2), dim=-1)

                denom = d00 * d11 - d01 * d01
                if denom == 0:
                    continue
                inv_denom = 1.0 / denom
                bary1 = (d11 * d20 - d01 * d21) * inv_denom
                bary2 = (d00 * d21 - d01 * d20) * inv_denom
                bary0 = 1.0 - bary1 - bary2

                inside_triangle = (bary0 >= 0) & (bary1 >= 0) & (bary2 >= 0)
                if not inside_triangle.any():
                    continue

                # Interpolate z-values
                z_values_i = batch_z_values[i].to(torch.float32)
                z_interpolated = (bary0 * z_values_i[0] + bary1 * z_values_i[1] + bary2 * z_values_i[2])

                # Depth test and update
                current_depth = empty_depth_image[0, v_min_i:v_max_i + 1, u_min_i:u_max_i + 1, 0]
                depth_exists = (current_depth > 0)
                depth_closer = (z_interpolated < current_depth)
                update_mask = inside_triangle & (~depth_exists | (depth_exists & depth_closer))

                empty_depth_image[0, v_min_i:v_max_i + 1, u_min_i:u_max_i + 1, 0][update_mask] = z_interpolated[update_mask]
                face_idx_img[0, v_min_i:v_max_i + 1, u_min_i:u_max_i + 1][update_mask] = batch_indices[i]

                if uv_faces is not None:           # Interpolate u and v coordinates
                    uv_values_i = batch_uv_values[i].to(torch.float32)
                    u_interpolated = (bary0 * uv_values_i[0, 0] + bary1 * uv_values_i[1, 0] + bary2 * uv_values_i[2, 0])
                    v_interpolated = (bary0 * uv_values_i[0, 1] + bary1 * uv_values_i[1, 1] + bary2 * uv_values_i[2, 1])
                    uv_interpolated = torch.stack([u_interpolated, v_interpolated], dim=-1)
                    empty_uv_face_image[0, v_min_i:v_max_i + 1, u_min_i:u_max_i + 1][update_mask] = uv_interpolated[update_mask]

        return empty_depth_image, face_idx_img, empty_uv_face_image
    
    def render_depth_plus_pytorch3d(self, face_vertices_image, face_vertices_z, view_target_h, view_target_w, device, uv_faces=None, render_scale=1.0):
        """
        Rasterize a depth map, face index map, and UV map using PyTorch3D for speed.

        Args:
        - face_vertices_image: Tensor (1, num_faces, 3, 2) with 2D vertices (NDC XY) of each face. Assumes +Y is up.
        - face_vertices_z: Tensor (1, num_faces, 3) with view-space z-values (positive depth) of each face vertex.
        - view_target_h: Height of the target image
        - view_target_w: Width of the target image
        - device: Device to use
        - uv_faces: Optional Tensor (1, num_faces, 3, 2) with uv coordinates of each face vertex.
        - render_scale: Scale factor for rendering resolution (default 1.0).

        Returns:
        - depth_image: Tensor (1, view_target_h, view_target_w, 1)
        - face_idx_buffer: Tensor (1, view_target_h, view_target_w)
        - uv_image: Tensor (1, view_target_h, view_target_w, 2)
        """
        batch_size = 1 # This function processes one mesh at a time
        num_faces = face_vertices_image.shape[1]
        render_h = int(view_target_h * render_scale)
        render_w = int(view_target_w * render_scale)

        face_vertices_image = face_vertices_image.to(device)
        face_vertices_z = face_vertices_z.to(device)
        if uv_faces is not None:
            uv_faces = uv_faces.to(device)

        # PyTorch3D's NDC conventions (+X right, +Y up, +Z into screen)        
        face_vertices_image[..., 1] *= -1       # Flip Y to match PyTorch3D's conventions
        verts_packed = torch.cat(
            (face_vertices_image, face_vertices_z.unsqueeze(-1)),
            dim=-1
        ) # Shape: (1, num_faces, 3, 3)

        verts_list = [verts_packed[0].reshape(-1, 3)]
        faces_list = [torch.arange(num_faces * 3, device=device).reshape(num_faces, 3)]
        pytorch3d_mesh = Meshes(verts=verts_list, faces=faces_list)

        # Adjust bin_size and max_faces_per_bin for performance/memory trade-off, `bin_size=0` coarse-to-fine approach is often fast.
        raster_settings = RasterizationSettings(
            image_size=(render_h, render_w), blur_radius=1e-6, faces_per_pixel=1, perspective_correct=True, cull_backfaces=False, clip_barycentric_coords=False)

        # - pix_to_face: (N, H, W, faces_per_pixel) LongTensor mapping pixels to face indices (-1 for background)
        # - zbuf: (N, H, W, faces_per_pixel) FloatTensor depth buffer (lower values are closer)
        # - bary_coords: (N, H, W, faces_per_pixel, 3) FloatTensor barycentric coordinates
        pix_to_face, zbuf, bary_coords, _ = rasterize_meshes(
            meshes=pytorch3d_mesh,
            image_size=(render_h, render_w),
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            perspective_correct=raster_settings.perspective_correct,
            clip_barycentric_coords=raster_settings.clip_barycentric_coords,
            cull_backfaces=raster_settings.cull_backfaces
        )

        face_idx_buffer = pix_to_face.squeeze(-1)
        face_idx_buffer = torch.where(face_idx_buffer >= 0, face_idx_buffer, torch.tensor(0, dtype=torch.long, device=device))

        depth_image = zbuf.squeeze(-1).unsqueeze(-1)
        background_mask = (pix_to_face.squeeze(-1) < 0).unsqueeze(-1)
        depth_image = torch.where(background_mask, torch.tensor(0.0, device=device), depth_image)
        depth_image = torch.clamp(depth_image, min=0.0)

        # 3. UV Image
        if uv_faces is not None:
            uv_image = interpolate_face_attributes(
                pix_to_face, bary_coords, uv_faces[0]
            ) # Output shape: (N, H, W, faces_per_pixel, AttrDim=2)
            uv_image = uv_image.squeeze(-2) # Squeeze faces_per_pixel dim -> (N, H, W, 2)
            uv_image = torch.where(background_mask.expand_as(uv_image), torch.tensor(0.0, device=device), uv_image)
        else:
            uv_image = torch.zeros(batch_size, render_h, render_w, 2, device=device)

        if render_scale != 1.0:        # Handle potential upsampling if render_scale was not 1.0
            depth_image = F.interpolate(depth_image.permute(0, 3, 1, 2), size=(view_target_h, view_target_w), mode='nearest').permute(0, 2, 3, 1)
            face_idx_buffer = F.interpolate(face_idx_buffer.unsqueeze(1).float(), size=(view_target_h, view_target_w), mode='nearest').long().squeeze(1)
            uv_image = F.interpolate(uv_image.permute(0, 3, 1, 2), size=(view_target_h, view_target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


        return depth_image, face_idx_buffer, uv_image
