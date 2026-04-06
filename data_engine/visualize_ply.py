import time

from dataVisu_engine.ThreeD_helpers import *
from dataVisu_engine.TwoD_helpers import *
from utils.general import *
from utils.config import *
from utils.camFormat import *

######################## CONFIG ########################
HASH = "-8512"          # -8929: highRes (5), -8512: lowRes (60)
SCALE = 60
CONFIG_PATH = "configs/EarthEngine"  # Path to the config file
CONFIG_FNAME = f"Scale{SCALE}_{HASH}.json"

PLY_PATH = "data/EarthEngine"   # Path to the ply file loading
PLY_PCNAME = f"pointcloud_{HASH}.ply"       # pointcloud_{HASH}.ply, _WiCloud
PLY_MESHNAME = f"mesh_{HASH}.ply" # Name of the mesh file, _LaubH

DEPTH_NAME = f"Scale{SCALE}DemNASA30m_{HASH}.png"
SAT_NAME = "sat_2024_10_09_preprocess.jpg"
SATCFG_NAME = "SatCloud_Kachelmann_superHD.json"

NOVEL_SCENE_PATH = "data/NovelViews"
SCENE_PATH = "data/2024-10-09/12-00-00"

TRANSFORM_MATRIX_PATH = "data/transformation_matrices/transforms_novelLaubH_wayp_all.json"        # transformation_final_v3.json
TRANSFORM_MATRIX_NOVEL_PATH = "data/NovelViews/transformation_test_near.json"

FILTER = False  # filter far away depth values in sampled image, config_file: SimFilter.json in SimImages
RGBD = True     # save RGBD images

NUM_RAND_SAMPLES = 1000

AOI_CUT = [7.880222, 8.023, 46.538060, 46.681889]  # [lonMin, lonMax, latMin, latMax]

WHICH_TASK = 'show_ply'   # show_ply, show_mesh_from_pcd, addCloud, 
                # render_w2c, render_w2c_1Cam, render_w2c_warped, 
                # render_w2c_1Cam_warped, render_w2c_1Cam_pano, render_w2c_panos, samplePCAnywhere_1, samplePCAnywhere
                # sampleMeshRandom, sampleMeshOfAoi
                # visualize_ply

CAM_ID = '3-NIEDH'   # see list used for render_w2c_1Cam

######################## PROGRAM ########################
home_path = getHomePath()
ply_path = Path(home_path, PLY_PATH)
satCloud_path = Path(home_path, SCENE_PATH, SAT_NAME)
depth_path = Path(home_path, PLY_PATH, DEPTH_NAME)
config_fPath = Path(home_path, CONFIG_PATH, CONFIG_FNAME)
configCloud_fPath = Path(home_path, Path(CONFIG_PATH).parent, SATCFG_NAME)
transforms_path = Path(TRANSFORM_MATRIX_PATH)

config = Config()
config.loadGEE(config_fPath)
config.loadSatCloud(configCloud_fPath)
transformMatrices = openTFMatrix(transforms_path)


######################## TASKS ########################
if WHICH_TASK == 'show_ply':
    loadPointCloud_ply(ply_path, PLY_PCNAME, show=True)
    loadMesh_ply(ply_path, PLY_MESHNAME, show=True)

elif WHICH_TASK == 'show_mesh_from_pcd':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    duration = time.time()
    mesh = meshFromPointCloud(pcd, radius=0.01, max_nn=40, width=8, show=False)
    print('Time mesh creation', time.time()-duration)
    saveMesh_ply(mesh, ply_path, fName=PLY_MESHNAME)
   
elif WHICH_TASK == 'sample_triMesh_1':
    mesh = loadTriMesh_ply(ply_path, PLY_MESHNAME, show=False)
    for key in transformMatrices[CAM_ID].keys():
        if CAM_ID in key:
            frame_id = key
            break
    
    print('Used frame ID:', frame_id)
    transformMatrix = np.array(transformMatrices[CAM_ID][frame_id]['transform_matrix'])
    
    t_gps = transformMatrix[:3, 3].tolist()
    t_pix = geoCoord2Open3Dpx(config, t_gps)
    transformMatrix[:3, 3] = np.array(t_pix)

    intrinsics = {'fx': transformMatrices[CAM_ID][frame_id]['fx'],
                    'fy': transformMatrices[CAM_ID][frame_id]['fy'],
                    'cx': transformMatrices[CAM_ID]['cx'],
                    'cy': transformMatrices[CAM_ID]['cy'],
                    'wid': transformMatrices[CAM_ID]['wid'],
                    'hei': transformMatrices[CAM_ID]['hei']
                  }
    
    save_path = Path(Path(config.data_fileTransformMatrix).parent, 'mesh')
    print('Save to:', save_path)

    sampleImg_TriMesh(save_path, mesh, transformMatrix, intrinsics, frame_id, filter=FILTER, rgbd=RGBD, show=False)

elif WHICH_TASK == 'addCloud':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    satCloud_img = loadGrayImg(satCloud_path)
    height_img = loadUnchangedImg(depth_path)
    addCloud2Ply(config, pcd, satCloud_img, height_img)
    showPointCloud(pcd)

    fName = PLY_PCNAME.replace('.ply', '_WiCloud.ply')
    savePointCloud_ply(pcd, ply_path, fName=fName)

elif WHICH_TASK == 'render_w2c':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    sampleImgCam_OpenCV(config, pcd, transformMatrices, point_size=20, depth_max=300, filtering=FILTER)

elif WHICH_TASK == 'render_w2c_1Cam':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    distortion = None
    sampleImg1Cam_OpenCV(config, pcd, transformMatrices, CAM_ID, distortion=distortion, point_size=5, depth_max=1000, filtering=FILTER, show=True)

elif WHICH_TASK == 'render_w2c_warped':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    sampleImgCam_OpenCV(config, pcd, transformMatrices, point_size=20, depth_max=300, warping2Cyl=True)

elif WHICH_TASK == 'render_w2c_1Cam_warped':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    distortion = None
    sampleImg1Cam_OpenCV(config, pcd, transformMatrices, CAM_ID, point_size=5, depth_max=1000, warping2Cyl=True, show=True)

elif WHICH_TASK == 'render_w2c_1Cam_pano':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    distortion = None
    samplePano1Cam_OpenCV(config, pcd, transformMatrices, CAM_ID, distortion=distortion, point_size=10, depth_max=1000)

elif WHICH_TASK == 'render_w2c_panos':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)
    distortion = None
    samplePanos_OpenCV(config, pcd, transformMatrices, distortion=distortion, point_size=10, depth_max=2000)

elif WHICH_TASK == 'samplePCAnywhere_1':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)

    transformMatrices = openTFMatrix(TRANSFORM_MATRIX_NOVEL_PATH)
    config.setData_pathScene(NOVEL_SCENE_PATH)

    sample1ImgAnywhere_OpenCV(config, pcd, transformMatrices, CAM_ID, point_size=5, depth_max=1000, show=True)

elif WHICH_TASK == 'samplePCAnywhere':
    pcd = loadPointCloud_ply(ply_path, PLY_PCNAME)

    transformMatrices = openTFMatrix(TRANSFORM_MATRIX_NOVEL_PATH)
    config.setData_pathScene(NOVEL_SCENE_PATH)

    sampleImgAnywhere_OpenCV(config, pcd, transformMatrices, point_size=5, depth_max=1000)

elif WHICH_TASK == 'sampleMeshRandom':
    mesh = loadTriMesh_ply(ply_path, PLY_MESHNAME)
    config.setData_pathScene(NOVEL_SCENE_PATH)

    save_folder = 'meshRand'
    sample_MeshRandomly(config, mesh, NUM_RAND_SAMPLES, save_folder, aoi=AOI_CUT, seed=42)

elif WHICH_TASK == 'sampleMeshOfAoi':
    mesh = loadTriMesh_ply(ply_path, PLY_MESHNAME)
    config.setData_pathScene(NOVEL_SCENE_PATH)

    save_folder = 'meshAoi'
    sampleMeshOfAoi(config, mesh, NUM_RAND_SAMPLES, save_folder, AOI_CUT)

elif WHICH_TASK == 'visualize_ply':
    title = 'Visualize PLY'
    path = Path(ply_path, PLY_MESHNAME)
    visualize_ply(title, path)