from pathlib import Path

from dataVisu_engine.TwoD_helpers import *
from utils.general import *
from utils.config import Config


######################## CONFIG ########################
SCENE_PATH = 'data/2024-10-09/12-00-00'
SCENE_PATH2 = 'data/2024-07-25/12-00-00'
GEE_PATH = 'data/EarthEngine'

SATCLOUD_FNAME = 'sat_2024_10_09_10_00_1115_1427.jpg'
SATTEMP_FNAME = 'sat_2024_10_09_10_00_1115_1428_temp.jpg'
SAT_FNAME = 'Scale60LandswissT10cm_3277686686935559486.png'

TRANSFORM_MATRIX_PATH = f'data/transformation_matrices/transformation_2learn_v1.json'
TRANSFORM_MATRIX_TRAIN_PATH = f'data/transformation_matrices/transformation_final_v3_flat.json'

WHICH_TASK = 'compare_realVsSamples'     # compare_realVsSamples, compare_realVsSample_warp, compare_realVsSample_1, 
            # compare_PlanevsSamples, overlay_realSamples,
            # overlay_realSample_warp, overlay_realSample_1, overlay_PlanevsSamples, compare_warping, overlay_warping, overlay_pano_1,
            # overlay_panos, compare_pano_1, compare_panos, pano2plane_1, pano2plane, compare_CycVsPlane_1, compare_CycVsPlane,
            # compare_CycVsPlane_cmp, compare_CycVsPlane_ovl, overlay_satCloud, camCutout, compare_poseAtOtherTime_1, compare_poseAtOtherTime,
            # compare_poseTraining_1, compare_poseTraining, analyze_fxFy, plot_camDensity

CAM_ID = '1-HEISCH'  # 1-HEISCH, 2-BSPIEZ, 3-NIEDH, 4-FLSPA, 5-MEIKAES, 6-MEIMAEG, 7-MEIALPT, 8-MELCHDRF, 9-MELCHBONI, 25-JUNGFOST, 29-HARDKULM, 30-BRROTHG


######################## PROGRAM ########################
# set used paths
# home, scene paths
home_path = getHomePath()
scene_path = Path(home_path, SCENE_PATH)
scene_path2 = Path(home_path, SCENE_PATH2)
# image and sampled/simulated images paths
real_path = Path(scene_path, 'images')
imgCyl_path = real_path
realPlane_path = Path(scene_path, 'imagesPlane')
realPlane_path2 = Path(scene_path2, 'imagesPlane')
sampled_path = Path(home_path, 'data/SimImages_WiCloud')
# cloud satellite paths
satCloud_fPath = Path(scene_path, SATCLOUD_FNAME)
satCloudTemp_fPath = Path(scene_path, SATTEMP_FNAME)
sat_fPath = Path(home_path, GEE_PATH, SAT_FNAME)
satRoadNet_fPath = Path(home_path, GEE_PATH, 'satelliteMeta', 'kach_roadNet.png')
# cutout path
cutout_path = Path(home_path, Path(GEE_PATH).parent, 'PanoramasGen4MitRand')
# compare paths
imgPlaneCmp_path = Path(home_path, 'data/ReVsIm_plane')
imgCylCmp_path = Path(scene_path, 'ReVsIm_cyc')
imgCmpCycPlane_path = Path(scene_path, 'CycVsPlane_img')

# config
config = Config()
config.loadCamMetadata(config.data_pathMetadata)
camMetadata = config.data_camOrigMetadata

######################## TASKS ########################
if WHICH_TASK == 'compare_realVsSamples':
    compareRealvsSampled(real_path, sampled_path, save_path=imgCylCmp_path)
elif WHICH_TASK == 'compare_realVsSample_warp':
    compareRealvsSampled(real_path, sampled_path, save_path=Path(scene_path, 'ReVsIm_warp'), warp=True)
elif WHICH_TASK == 'compare_realVsSample_1':
    compareRealvsSampled(real_path, sampled_path, save_path=imgCylCmp_path, cam_id=CAM_ID, show=True)
elif WHICH_TASK == 'compare_PlanevsSamples':
    compareRealvsSampled(realPlane_path, sampled_path, save_path=imgPlaneCmp_path)
elif WHICH_TASK == 'overlay_realSamples':
    overlayRealvsSampled(real_path, sampled_path, save_path=imgCylCmp_path)
elif WHICH_TASK == 'overlay_realSample_warp':
    overlayRealvsSampled(real_path, sampled_path, save_path=Path(scene_path, 'ReVsIm_warp'), warp=True, show=False)
elif WHICH_TASK == 'overlay_realSample_1':
    overlayRealvsSampled(real_path, sampled_path, save_path=imgCylCmp_path, cam_id=CAM_ID, show=True)
elif WHICH_TASK == 'overlay_PlanevsSamples':
    overlayRealvsSampled(realPlane_path, sampled_path, save_path=imgPlaneCmp_path)
elif WHICH_TASK == 'compare_warping':
    compareWarping(scene_path, camMetadata, task='compare', show=False)
elif WHICH_TASK == 'overlay_warping':
    compareWarping(scene_path, camMetadata, task='overlay', show=False)
elif WHICH_TASK == 'overlay_pano_1':
    overlay1Pano(scene_path, camMetadata, CAM_ID, show=True)
elif WHICH_TASK == 'overlay_panos':
    overlayPanos(scene_path, camMetadata)
elif WHICH_TASK == 'compare_pano_1':
    compare1Pano(scene_path, camMetadata, CAM_ID, show=True)
elif WHICH_TASK == 'compare_panos':
    comparePanos(scene_path, camMetadata)
elif WHICH_TASK == 'pano2plane_1':
    transformMatrix = openTFMatrix(TRANSFORM_MATRIX_PATH)
    pano2plane_1(scene_path, transformMatrix, CAM_ID, show=True)
elif WHICH_TASK == 'pano2plane':
    transformMatrix = openTFMatrix(TRANSFORM_MATRIX_PATH)
    pano2plane(scene_path, transformMatrix)
elif WHICH_TASK == 'compare_CycVsPlane_1':
    compareSplittedPano(imgCyl_path, realPlane_path, save_path=imgCmpCycPlane_path, cam_id=CAM_ID, show=True)
elif WHICH_TASK == 'compare_CycVsPlane':
    compareSplittedPano(imgCyl_path, realPlane_path, save_path=imgCmpCycPlane_path)
elif WHICH_TASK == 'compare_CycVsPlane_cmp':
    compareCycVsPlane(scene_path, imgCylCmp_path, imgPlaneCmp_path, save_path=Path(scene_path, 'CycVsPlane'), task='compare')
elif WHICH_TASK == 'compare_CycVsPlane_ovl':
    compareCycVsPlane(scene_path, imgCylCmp_path, imgPlaneCmp_path, save_path=Path(scene_path, 'CycVsPlane'), task='overlay')
elif WHICH_TASK == 'overlay_satCloud':
    # NOTE: (150, -31 deg), (117, -12 deg) --> image to temperature
    #       (10 deg, 1000m), (-50 deg, 10000m) --> temperature to height
    # There is 1.satellite and 2.temperature dataset
    # Temperature data can be used to get the cloud height
    DELTA_VTEMP = -19/33
    NULL_VTEMP = 55.36
    DELTA_TEMPH = -9000/60
    NULL_TEMPH = 2500
    satCloudconfig = {
            'name': 'KachSuperHD',
            'satelliteMeta_path': 'data/EarthEngine/satelliteMeta',
            'cloudThresh': 110,
            'cropLon_min': 127, 
            'cropLon_max': 673, 
            'cropLat_min': 79, 
            'cropLat_max': 561,
            'delta_vTemp': DELTA_VTEMP,
            'null_vTemp': NULL_VTEMP,
            'delta_tempH': DELTA_TEMPH,
            'null_tempH': NULL_TEMPH,
            'avg_cloudHeight': 500,
            }
    overlaySatCloud(sat_fPath, satCloud_fPath, satCloudTemp_fPath, satRoadNet_fPath, satCloudconfig, show=False)

    config.setSatCloud(satCloudconfig)
    config.saveSatCloud()
elif WHICH_TASK == 'camCutout':
    camCutout(cutout_path)
elif WHICH_TASK == 'compare_poseAtOtherTime_1':
    comparePoseAtOtherTime_1(realPlane_path, realPlane_path2, CAM_ID, show=True)
elif WHICH_TASK == 'compare_poseAtOtherTime':
    comparePoseAtOtherTime(realPlane_path, realPlane_path2)
elif WHICH_TASK == 'compare_poseTraining_1':
    transfMatrix_before = openTFMatrix(TRANSFORM_MATRIX_PATH)
    transfMatrix_after = openTFMatrix(TRANSFORM_MATRIX_TRAIN_PATH)
    comparePoseTraining_1(realPlane_path, transfMatrix_before, transfMatrix_after, CAM_ID, config, show=True)
elif WHICH_TASK == 'compare_poseTraining':
    transfMatrix_before = openTFMatrix(TRANSFORM_MATRIX_PATH)
    transfMatrix_after = openTFMatrix(TRANSFORM_MATRIX_TRAIN_PATH)
    comparePoseTraining(realPlane_path, transfMatrix_before, transfMatrix_after, config)
elif WHICH_TASK == 'analyze_fxFy':
    transfMatrix_after = openTFMatrix(TRANSFORM_MATRIX_TRAIN_PATH)
    analyze_fxFy(transfMatrix_after, TRANSFORM_MATRIX_TRAIN_PATH, show=True)
elif WHICH_TASK == 'plot_camDensity':
    transformMatrix = openTFMatrix(TRANSFORM_MATRIX_PATH)
    plot_camDensity(transformMatrix, scene_path, show=True)