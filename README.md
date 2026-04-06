# Season Scapes: Learning Large-scale Re-lightable 3D Landscapes with Seasonal Variation from Sparse Webcams
[![Python 3.9.21](https://img.shields.io/badge/python-3.9.21-blue.svg)](https://www.python.org/downloads/release/python-3921/) ![Static Badge](https://img.shields.io/badge/Cuda-11.8-blue) 

When using xformer:

![Static Badge](https://img.shields.io/badge/Cuda-12.4-blue)

## Setup Environment
With xformer optimization (CUDA is v12.4):

    conda env create --file environment_new.yml

Without xformer (CUDA is v11.8):

    conda env create --file environment.yml

Download the dataset from Huggingface: https://huggingface.co/datasets/ChlaegerIO/SeasonScapes and store it in `data/` folder (default).

## Overview
The codebase has 3 main components:

1. A `data_engine` to 
   - download datasets (Google Earth data, Roundshot webcam data - Roundshot downlaods are limited for an IP address -> use dataset from huggingface instead),
   - preprocess and visualize the dataset (sec3.2), 
   - a 3D-to-2D point matching tool (sec3.3) and 
   - pose optimization training (sec3.3)
2. A `paint_engine` with a 3D mesh texturing pipeline to paint and inpaint a 3D mesh and render images for Gaussian Splatting (sec3.4 - 3.6)
3. An `eval_engine` to evaluate the results (PSNR, SSIM and LPIPS)

<img width="4072" height="2181" alt="main_graph" src="https://github.com/user-attachments/assets/77864463-e9a3-4a84-8adb-b176c6dd2395" />


## How to run experiments
1. Split up the webcam images at specified timestamps (in the file)

        python ./data_engine/s_pano2plane.py

2. Create new camera transformation file(s) or use existing ones for known views, novel view path and evaluation in `data/transformation_matrices/...`
3. Create or use existing painting configuration in `configs/paint/...` and check most important painting opations
4. Run painting pipeline

        python -u ./paint.py configs/paint/REGION_J/paint_config_240901-1200.yaml
5. For evaluation copy target images and evaluation file from `paint_engine/Logs/.../eval_metrics` to `eval_engine/eval_data/...` including the generated eval_mask
6. Run evaluation

       python -u ./eval.py --eval_path eval_engine/eval_data/paint_JungfRegion_behavior/240901-1200_2

### Most important painting opations
- `calcu_uncolored_mode: HSV`: mode how to calculate inpaint mask before dilation
- `fileTransformMatrix`: known view transformation file, check correct timestamp
- `ip_adapter_image_path`: path to IP-Adapter image
- `dilate_kernel`: dilate kernel for inpainting mask
- `transformMatrix_novel_path`: novel view path, also check eval, metrics path
- `cache_id`: which cached UV mapping to use
- `exp_path`: where experiments are saved
- `texture_resolution`: resolution of UV map

## Dataset
The dataset consists of 360° webcam images and downloaded Google Earth Digital Elevation Model (DEM) and Satellite image. The data is stored in folders `<data>/<time>` and the landscape images and point clouds are stored in `EarthEngine` folder assigned with a hash and a corresponding configuration file in configs. The DEM is stored in meters and the values can be in [-32767, 32767]. We have downloaded 28 days between 11.05.2024 and 23.01.2025 mostly between 10 am. and 6 pm.

Roundshot 360° webcam and split up example
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/3bff46bf-1f27-42f7-9c0d-febd1e919698"/>
    <img src="https://github.com/user-attachments/assets/05f4a02a-218b-408f-8779-37cf42dc26d3" alt="im1" width="24.6%"/>
    <img src="https://github.com/user-attachments/assets/29ea1299-6085-4efb-843b-974b8dbc24c1" alt="im2" width="24.6%"/>
    <img src="https://github.com/user-attachments/assets/87b02947-3145-4aea-9de4-fa03c3b603c8" alt="im3" width="24.6%"/>
    <img src="https://github.com/user-attachments/assets/4d68a79a-dd9b-4ca6-adda-a424dfac4a30" alt="im4" width="24.6%"/>
</div>
<br>

Google Earth data 
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/56a49701-8583-451b-9c78-347e9a2816e7" alt="satellite" width="49.7%"/>
    <img src="https://github.com/user-attachments/assets/ba6f7ef5-93df-4cc6-9ff6-9b820ca8ddb5" alt="DEM" width="49.7%"/>
</div>

## Repository structure
### File structure
```
seasonScapes
│
└───configs
│   │   Livecams-....csv --> from roundshot company
│   │   SatCloud_....json --> configs for cloud satellite
│   │   sd15_....json --> stable diffusion configuration
│   │
│   └───EarthEngine
│   │   └───configuration for point cloud data
│   │
│   └───paint
│       └───configuration for painting pipeline
│   
└───data
|   |
│   └───2024-10-09 --> testing scene
│   │   |    sat_2024_....jpg --> cloud satellite grayscale or measured temperature
│   │   |   
│   │   └───360 --> original images
│   │   └───images --> splitted up images (cylindrical)
│   │   └───imagesPlane --> splitted up images (planar)
|   |
│   └───EarthEngine
│       |   pointcloud --> point clouds
│       |   Scale60DEM --> hight model
│       └───Scale60Land --> satellite without clouds for point cloud
│
└───data_engine
│   |   files to run are here ...
│   |
│   └───dataPose_optimization --> pose optimization model
│   └───dataVisu_engine --> library for visualizations and renderings
│   └───utils --> libraries
│   └───Logs --> tensorboard logs: tensorboard --logdir=data_engine/Logs
│
└───paint_engine
│   |   paint pipeline and util files
│   |
│   └───cache --> UV texture map cached 
│   └───Logs --> painted images and mesh output
│
└───eval_engine
    |   eval pipeline and util files
    |
    └───eval_data --> data to be evaluated

```
### Important python files
`python data_engine/transformMatrix.py`: Create transformation matrices, it is dependent on configs from roundshot

`python data_engine/GEE_processing.py`: Download earth engine satellite and height model

`python data_engine/roundshot_scrapper.py`: Download webcam images

`python data_engine/s_create_ply.py`: create the point cloud ply and mesh from satellite and DEM

`python data_engine/s_pano2plane.py`: Split up Roundshot webcams to planar images

`python data_engine/s_sample_earthMesh.py`: sample depth from the mesh

`python data_engine/visualize_ply.py`: Different visualisation task of the point cloud or mesh

`python data_engine/data_preprocess_analysis.py`: Data comparison and preprocessing tasks

`python data_engine/3Dto2DPointMatching.py`: Tool to match 3D and 2D points

`python data_engine/AdjustCamIntrinsics.py`: Tool to test intrinsics, pose changes

`python data_engine/train_camPose_pointMatched.py`: Train camera pose optimization

`paint_engine/paint_pipeline.py`: Paint pipeline

`paint_engine/cfg.py`: Default configuration and definitions

`eval_engine/eval_pipeline.py`: Evaluation


