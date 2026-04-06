import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field, asdict, is_dataclass, MISSING
from pathlib import Path
from typing import List, Any, Dict, get_type_hints
import yaml

from data_engine.utils.config import Config
from data_engine.utils.general import getHomePath


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    width: int = 3072      # w, h
    height: int = 2048
    scale_img: float = 1.0 # scale rendering including focal length
    calcu_uncolored_mode: str = "HSV"  # FACE_ID, DIFF, WarpGrid, HSV
    texture_interpolation_mode: str = 'bilinear'
    texture_default_color: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.8])
    render_angle_thres: float = 68
    texture_resolution: List[int] = field(default_factory=lambda: [1024, 1024])  # h w
    force_run_xatlas: bool = False
    trained_mesh: str = None
    initial_texture_path: Path = None
    use_depth_mask: bool = False
    depth_estimation: str = "depthAny_v2"       # "midas", "depthAny_v2"
    depth_filter_boundaries: List[float] = field(default_factory=lambda: [0.25, 0.7])
    batch_size_internal: int = 10000
    filter_background: float = 0.9
    mask_sky: bool = False
    gps: bool = True
    inpaint_openCV: bool = False
    uv_inpaint_method: str = "neighbor"  # "neighbor", "controlnet", "none"
    inpaint_from_higher_attitude: bool = False          # if true, inpaint from higher attitude at painting views
    inpaint_from_higher_attitude_height: float = 0.0    # height in pixel coordinates (~meters/60) to inpaint from higher attitude, 60=scale

@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    seed: int = 0
    lr: float = 1e-2
    train_step: int = 200

@dataclass
class LogConfig:
    exp_path: str = "paint_engine/Logs/LaubH"            # path to save output
    cache_path: str = "paint_engine/cache"
    cache_id: str = "-8512_LaubH"                             # cache id
    only_paint: bool = False                        # only paint, no inpainting if true
    ablation_no_paint: bool = False                # no paint, only inpainting
    ablation_no_inpaint: bool = False              # no inpainting, only paint
    ablation_no_uvInpaint: bool = False           # no UV inpainting

@dataclass
class DataConfig:
    """ dataset engine configuration"""
    gee_config_path: str = "configs/EarthEngine/Scale60_-8512_LaubH.json"
    sat_config_path: str = "configs/SatCloud_Kachelmann_superHD.json"
    sd_config_path: str = "configs/sd15.json"
    cam_config_path: str = "configs/Cam_-8512.json"
    fileTransformMatrix: str = "data/transformation_matrices/transforms_knownView.json"
    config = None

    def __post_init__(self):
        if self.config is None:
            self.config = Config()
            self.config.loadGEE(Path(getHomePath().as_posix(), self.gee_config_path))
            self.config.loadCam(Path(getHomePath().as_posix(), self.cam_config_path))
            self.config.loadSatCloud(Path(getHomePath().as_posix(), self.sat_config_path))
            self.config.loadSD(Path(getHomePath().as_posix(), self.sd_config_path))

    def update_config(self):
        self.config.loadGEE(Path(getHomePath().as_posix(), self.gee_config_path))
        self.config.loadCam(Path(getHomePath().as_posix(), self.cam_config_path))
        self.config.loadSatCloud(Path(getHomePath().as_posix(), self.sat_config_path))
        self.config.loadSD(Path(getHomePath().as_posix(), self.sd_config_path))

@dataclass
class Inpaint:
    """ Parameters for the inpaint process """
    sd_model_key: str = "runwayml/stable-diffusion-v1-5"
    image_path: str = ""
    mask_path: str = ""
    ip_adapter_image_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    width: int = 1024
    height: int = 512
    num_images_per_prompt: int = 1
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    denoising_strength: float = 1.0
    dilate_kernel: int = 1
    dilate_regions: bool = False
    controlnet_units: List[dict] = field(default_factory=lambda: [
        {
            "preprocessor": "none",
            "controlnet_key": "lllyasviel/control_v11f1p_sd15_depth",
            "condition_image_path": "",
            "weight": 1.0
        },
        {
            "preprocessor": "inpaint_global_harmonious",
            "controlnet_key": "lllyasviel/control_v11p_sd15_inpaint",
            "condition_image_path": "",
            "weight": 0.5
        }
    ])

@dataclass
class UVinpaint:
    """ Parameters for the UVpaint process """
    sd_model_key: str = "GeorgeQi/realisticVisionV13_v13"
    image_path: str = ""
    ip_adapter_image_path: str = ""
    mask_path: str = ""
    prompt: str = "UV map, (Landscape with realistic mountains:1.3), high quality, best quality"
    negative_prompt: str = "blur, low quality, noisy image, over-exposed, shadow"
    seed: int = -1
    width: int = 2048
    height: int = 3072
    num_images_per_prompt: int = 1
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    denoising_strength: float = 0.75
    dilate_kernel: int = 1
    dilate_regions: bool = False
    controlnet_units: List[dict] = field(default_factory=lambda: [
        {
            "preprocessor": "none",
            "controlnet_key": "GeorgeQi/Paint3d_UVPos_Control",
            "condition_image_path": "",
            "weight": 1.0
        },
        {
            "preprocessor": "inpaint_global_harmonious",
            "controlnet_key": "lllyasviel/control_v11f1e_sd15_tile",
            "condition_image_path": "",
            "weight": 1.0
        }
    ])

@dataclass
class DiffusionConfig:
    """ Parameters for the diffusion process """
    name: str = "stable diffusion v1.5"
    a_prompt: str = "best quality, extremely detailed"
    n_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples: int = 1
    image_resolution: int = 512
    detect_resolution: int = 384
    ddim_steps: int = 51
    guess_mode: bool = False
    strength: float = 1.0
    scale: float = 4.9
    seed: int = 17172222
    eta: int = 0
    calcDepth: bool = False
    transformMatrix_novel_path: str = "./data/transformation_matrices/transforms_novelView.json"   
    transformMatrix_eval_path: str = "./data/transformation_matrices/transforms_novelView.json"
    transformMatrix_metrics_path: str = "./data/transformation_matrices/transforms_eval.json"
    inpaint: Inpaint = field(default_factory=Inpaint)
    img2img: UVinpaint = field(default_factory=UVinpaint)


@dataclass
class PaintConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def save_to_yaml(self, file_path: Path):
        """
        Save the configuration to a YAML file.
        
        Args:
            file_path: Path where to save the YAML file
        """
        # Convert to dictionary
        config_dict = asdict(self)
        
        # Handle Path objects
        def path_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
        
        yaml.add_representer(Path, path_representer)
        
        # Create directory if it doesn't exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to YAML file
        with file_path.open('w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Configuration saved to {file_path}")
    
    def update_from_yaml(self, file_path: Path):
        """
        Update the configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML file
        """
        file_path = Path(file_path)
        
        # Read from YAML file
        with file_path.open('r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configuration recursively
        self.update_from_dict(config_dict)
        self.dataset.update_config()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        Update the configuration recursively from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values to update
        """
        for field_name, field_value in config_dict.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                
                # update the dataclass recursively
                if is_dataclass(current_value) and isinstance(field_value, dict):
                    current_value.update_from_dict(field_value)
                else:
                    # Otherwise, just replace the value
                    setattr(self, field_name, field_value)


# Apply the update_from_dict, create_from_dict method to all dataclasses
for cls in [LogConfig, RenderConfig, OptimConfig, DataConfig, Inpaint, UVinpaint, DiffusionConfig]:
    setattr(cls, 'update_from_dict', PaintConfig.update_from_dict)


# testing or manually saving Config to YAML
# paint_cfg = PaintConfig()
# paint_cfg.save_to_yaml('configs/paint/paint_config.yaml')
# paint_cfg.update_from_yaml('configs/paint/paint_config_waypoints.yaml')
# print("\n", paint_cfg.dataset, "\n")
# print("\n", paint_cfg.render, "\n")
# print("\n", paint_cfg.diffusion, "\n")
# print("\n", paint_cfg.optim, "\n")
# print("\n", paint_cfg.log, "\n")