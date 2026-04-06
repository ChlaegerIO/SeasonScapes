import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline
from typing import Union, Tuple
import cv2
import copy

from .general import *


def get_rotation_angle(rotation_matrix, rotSeq='xyz', degree=True):
    '''
    Extract the rotation angle from the rotation matrix with rotation sequence: default xyz
    '''
    rotation = Rotation.from_matrix(rotation_matrix)
    # Extract the rotation angle in degrees
    rotation_angle = rotation.as_euler(rotSeq, degrees=degree)
    
    return rotation_angle

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def transform2colmap_camF(transformFile, data_path='data/2024-10-09/12-00-00'):
    '''
    Create the cameras.txt file for COLMAP
    Input:
    - transformFile: transformation matrix as json
    - data_path: relative path from parent folder to the data folder
    '''
    #loop through the cameras
    cameras = []
    for idx in range(len(transformFile)):
        cam_model = transformFile[str(idx)]['cam_id']
        width = transformFile[str(idx)]['wid']
        height = transformFile[str(idx)]['hei']
        fx = transformFile[str(idx)]['fx']
        fy = transformFile[str(idx)]['fy']
        cx = transformFile[str(idx)]['cx']
        cy = transformFile[str(idx)]['cy']
        camera = [idx, cam_model, width, height, fx, fy, cx, cy]
        cameras.append(camera)
    
    home_path = getHomePath()
    camFolder_path = Path(home_path, data_path, 'sparse/model')
    camFile_path = Path(camFolder_path, 'cameras.txt')

    if not camFolder_path.exists():
        Path(camFolder_path).mkdir(parents=True)

    with open(camFile_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: {}\n'.format(len(cameras)))
        for camera in cameras:
            f.write(' '.join(map(str, camera)) + '\n')

def transform2colmap_imgF(transformFile, data_path):
    '''
    Create the images.txt file for COLMAP
    Input:
    - transformFile: transformation matrix as json
    - data_path: relative path from parent folder to the data folder
    '''
    #loop through the images
    images = []
    cnt = 0
    for cam_id in transformFile.keys():
        fr_num = 0
        for data_id in transformFile[cam_id]:
            if cam_id not in data_id:
                continue   # skip metadata
            imgName = Path(transformFile[cam_id][data_id]['file_path']).parts[-1]
            transformMatrix = np.array(transformFile[cam_id][data_id]['transform_matrix'])
            # Extract the quaternion and translation
            R = transformMatrix[:3,:3]
            T = transformMatrix[:3,3]

            rotation = Rotation.from_euler('X', 180, degrees=True)
            rotMat = rotation.as_matrix()
            R = np.matmul(rotMat, R)

            Q = rotmat2qvec(R)
            Qw, Qx, Qy, Qz = Q[0], Q[1], Q[2], Q[3]
            Tx = T[0]
            Ty = T[1]
            Tz = T[2]
            image = [cnt, Qw,Qx,Qy,Qz,Tx,Ty,Tz, fr_num, imgName]
            images.append(image)
            cnt += 1
            fr_num += 1
    
    home_path = getHomePath()
    camFolder_path = Path(home_path, data_path, 'sparse/model')
    imgFile_path = Path(camFolder_path, 'images.txt')

    if not camFolder_path.exists():
        Path(camFolder_path).mkdir(parents=True)

    with open(imgFile_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(images)))
        for image in images:
            f.write(' '.join(map(str, image)) + '\n')
            f.write('\n')

def getCamTransform_np(translation, rotation_angle, rotation_axis='z'):
    '''
    Create the transformation matrix for the camera
    XYZ: intrinsic, local rotations, or xyz: extrinsic global reference rotations
    Input:
    - translation: translation vector
    - rotation_angle: rotation angle in degrees
    - rotation_axis: rotation axis
    '''
    # Create translation matrix
    T = np.eye(4)
    T[:3, 3] = translation

    # Create rotation matrix
    R = np.eye(4)
    rotation = Rotation.from_euler(rotation_axis, rotation_angle, degrees=True)
    R[:3, :3] = rotation.as_matrix()

    # Combine translation and rotation
    transform_matrix = np.dot(T, R)

    return transform_matrix

def rotation(rotation_angle, rotation_axis, degrees=True):
    '''
    Create the rotation matrix
    Input:
    - rotation_angle: rotation angle in degrees or list of angles
    - rotation_axis: rotation axis
    - degrees: True if the angle is in degrees, False if in radians
    '''
    rotation = Rotation.from_euler(rotation_axis, rotation_angle, degrees=degrees)
    return rotation.as_matrix()

def opengl_to_opencv(c2w):
    """
    Change opengl to opencv in camera to world matrix
    """
    # flip y, z axis
    c2w[:, 1] = -c2w[:, 1]
    c2w[:, 2] = -c2w[:, 2]
    return c2w

def opengl_to_kao(c2w):
    """
    Change opengl to kao (left-handed))
    """
    c2w[:, 0] = -c2w[:, 0]
    c2w[:, 1] = -c2w[:, 1]
    c2w[:, 2] = -c2w[:, 2]
    return c2w

def opengl_to_pytorch3d(c2w):
    """
    Change opengl to pytorch3d, flip x, z axis
    """
    c2w[:, 0] = -c2w[:, 0]
    c2w[:, 2] = -c2w[:, 2]
    return c2w

def opencv_to_pyroch3d(c2w):
    """
    Change opencv to pytorch3d, flip x, y axis
    """
    c2w[:, 0] = -c2w[:, 0]
    c2w[:, 1] = -c2w[:, 1]
    return c2w

def kao_to_opengl(c2w):
    """
    Change kao to opengl (right-handed))
    """
    c2w[:, 0] = -c2w[:, 0]
    c2w[:, 1] = -c2w[:, 1]
    c2w[:, 2] = -c2w[:, 2]
    return c2w

def opengl_to_init(rot):
    '''
    Rotate camera to initial coordinate system from OpenGL coordinate system
        from: x East, y Up, z South (camera looking to North)
        to: x South, y East, z Up
    Input:
    - rot: rotation matrix as numpy array
    '''
    rot_view = rotation(-90, 'z', degrees=True)
    rot = rot_view @ rot
    rot_yUp = rotation(-90, 'x', degrees=True)
    rot = rot_yUp @ rot
    return rot

def opencv_to_opengl(c2w):
    """
    Change opencv to opengl in camera to world matrix
    """
    c2w[:, 1] = -c2w[:, 1]
    # flip z axis
    c2w[:, 2] = -c2w[:, 2]
    return c2w

def opencv_to_init(c2w):
    """
    Rotate camera to initial coordinates from opencv c2w or rotation matrix
        from: x East, y Down, z South (camera looking to North)
        to: x South, y East, z Up
    Output:
    - rotation matrix
    """
    c2w = opencv_to_opengl(c2w)
    rot = c2w[:3, :3]
    rot = opengl_to_init(rot)
    return rot

def init_to_opengl(rot):
    '''
    Rotate camera to OpenGL coordinate system from standard coordinate system
        from: x South, y East, z Up
        to: x East, y Up, z South (camera looking to North)
    Input:
    - rot: rotation matrix as numpy array
    '''
    rot_yUp = rotation(90, 'x', degrees=True)
    rot = rot_yUp @ rot
    rot_view = rotation(90, 'z', degrees=True)
    rot = rot_view @ rot
    return rot

def init_to_pytorch3d(rot):
    '''
    Rotate camera to Pytorch3D coordinate system from standard coordinate system
        from: x South, y East, z Up
        to: x East, y Up, z South (camera looking to North)
    Input:
    - rot: rotation matrix as numpy array
    '''
    rotyUp = rotation(90, 'x', degrees=True)
    rot = rotyUp @ rot
    rot_view = rotation(-90, 'z', degrees=True)
    rot = rot_view @ rot
    return rot

def init_to_opencv(rot):
    '''
    Rotate camera to OpenCV coordinate system from standard coordinate system
        from: x South, y East, z Up
        to: x East, y Down, z South (camera looking to North)
    Input:
    - rot: rotation matrix as numpy array
    '''
    rot = init_to_opengl(rot)
    rot = opengl_to_opencv(rot)
    return rot

def geoCoord2Open3Dpx(config, coordIn : list, change_xy=True, invert_lat=True, invert_long=False):
    """
    Convert geo coordinates to pixel coordinates
    
    Input:
    - config: Config() object with GEE parameters
    - coord: [lon, lat, at] list
    
    Return:
    - [lat px, lon px, at]: list if change_xy is True
    """
    # List are mutable and are handled like a pointer! Make a copy list
    coord = copy.copy(coordIn)
    
    LONG_PX = config.GEE_aoi_pixel[2][0]
    LAT_PX = config.GEE_aoi_pixel[2][1]
    LONG_MAX = config.GEE_aoi[2][0]
    LAT_MAX = config.GEE_aoi[2][1]
    LONG_MIN = config.GEE_aoi[0][0]
    LAT_MIN = config.GEE_aoi[0][1]

    coordRet = [(coord[0] - LONG_MIN) / (LONG_MAX - LONG_MIN) * LONG_PX, 
             (coord[1] - LAT_MIN) / (LAT_MAX - LAT_MIN) * LAT_PX,
             coord[2]]
    if change_xy:
        coordTmp = coordRet
        coordRet = [coordTmp[1], coordTmp[0], coordTmp[2]]
        if invert_lat:    # flip lat axis
            coordRet[0] = LAT_PX - coordRet[0]    
        if invert_long:   # flip long axis
            coordRet[1] = LONG_PX - coordRet[1]
    else:
        if invert_lat:
            coordRet[1] = LAT_PX - coordRet[1]
        if invert_long:
            coordRet[0] = LONG_PX - coordRet[0]

    coordRet[2] = coordRet[2]/ config.GEE_pxScale

    return coordRet

def Open3Dpx2geoCoord(config, coordIn, change_xy=True, invert_x=True, invert_y=False):
    """
    Convert pixel coordinates to geo coordinates
    Input:
        config: Config() object
        coordIn: [lat px, lon px, at] list right handed
    Return:
        [lon, lat, at]: list if change_xy is True
    """
    # List are mutable and are handled like a pointer! Make a copy list
    coord = copy.copy(coordIn)

    LONG_PX = config.GEE_aoi_pixel[2][0]
    LAT_PX = config.GEE_aoi_pixel[2][1]
    LONG_MAX = config.GEE_aoi[2][0]
    LAT_MAX = config.GEE_aoi[2][1]
    LONG_MIN = config.GEE_aoi[0][0]
    LAT_MIN = config.GEE_aoi[0][1]

    if invert_x:    # flip x axis
        coord[0] = LAT_PX - coord[0]    
    if invert_y:   # flip y axis
        coord[1] = LONG_PX - coord[1]
    if change_xy:
        coordTmp = coord
        coord = [coordTmp[1], coordTmp[0], coordTmp[2]]

    coordRet = [coord[0] / LONG_PX * (LONG_MAX - LONG_MIN) + LONG_MIN, 
                coord[1] / LAT_PX * (LAT_MAX - LAT_MIN) + LAT_MIN,
                coord[2]]
    
    coordRet[2] = coordRet[2] * config.GEE_pxScale

    return coordRet

def mask_points_cameraAxis(points3d: np.ndarray, w2c: np.ndarray, msk_min, msk_max, otherMasked: np.ndarray = None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Masks out points3d that are behind the camera and outside of [msk_min, msk_max] i.e. for cv2.projectPoints
    Same masking applied to otherMasked if not None e.g. colors
    """
    # transform world to camera and mask
    w2c_hat = w2c[:3, :]
    points3d_loc = points3d.T
    points3d_loc = np.vstack((points3d_loc, np.ones((1, points3d_loc.shape[1]))))   # 4xN
    points3d_loc = w2c_hat @ points3d_loc  # 3xN
    points3d_loc = points3d_loc.T  # Nx3

    z_mask = (points3d_loc[:, 2] > msk_min) & (points3d_loc[:, 2] < msk_max)    # mask out points behind the camera
    points3d_loc = points3d_loc[z_mask]     # Nmx3

    # transform back to world
    c2w = np.linalg.inv(w2c)
    c2w_hat = c2w[:3, :]
    points3d_loc = points3d_loc.T
    points3d_loc = np.vstack((points3d_loc, np.ones((1, points3d_loc.shape[1]))))   # 4xNm
    points3d_loc = c2w_hat @ points3d_loc  # 3xNm
    points3d_loc = points3d_loc.T  # Nmx3

    if otherMasked is not None:
        otherMasked = otherMasked[z_mask]

    return points3d_loc, otherMasked

def cylindrical_to_plane_projection(panorama, fov_hor, fov_vert, yaw, pitch, output_resolution, pano_hor_fov=360, pano_vert_fov=60):
    """
    Corrected function for projecting a cylindrical panorama to a plane image.
    :param panorama: Cylindrical panorama image (NumPy array, HxWxC)
    :param fov_hor: Horizontal field of view (in degrees) of the output image.
    :param fov_vert: Vertical field of view (in degrees) of the output image.
    :param yaw: Horizontal viewing direction (in degrees).
    :param pitch: Vertical viewing direction (in degrees).
    :param output_resolution: Tuple (width, height) of the output plane image.
    :param pano_h_fov: Horizontal field of view (in degrees) of the panorama image.
    :param pano_v_fov: Vertical field of view (in degrees) of the panorama image.
    :return: Plane image (NumPy array, height x width x channels).
    """
    panorama_h, panorama_w, channels = panorama.shape
    output_w, output_h = output_resolution

    # Convert FOVs, yaw, and pitch to radians
    fov_h_rad = np.radians(fov_hor)
    fov_v_rad = np.radians(fov_vert)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    pano_h_fov_rad = np.radians(pano_hor_fov)
    pano_v_fov_rad = np.radians(pano_vert_fov)

    # Create normalized grid for the output image (from -1 to 1)
    x = np.linspace(-np.tan(fov_h_rad / 2), np.tan(fov_h_rad / 2), output_w)
    y = np.linspace(-np.tan(fov_v_rad / 2), np.tan(fov_v_rad / 2), output_h)
    xv, yv = np.meshgrid(x, y)

    # Calculate spherical angles (theta and phi) for each pixel in the output image
    theta = np.arctan2(xv, 1) + yaw_rad  # Horizontal angle (yaw) in radians
    phi = np.arctan2(yv, np.sqrt(xv**2 + 1)) + pitch_rad  # Vertical angle (pitch) in radians

    # Map angles to cylindrical coordinates
    u = (theta / pano_h_fov_rad) % 1  # Normalize horizontal angle to [0, 1] based on pano_h_fov
    v = ((phi + pano_v_fov_rad / 2) / pano_v_fov_rad) % 1  # Normalize vertical angle to [0, 1]

    # Convert normalized coordinates to pixel indices in the panorama
    u_img = (u * panorama_w).astype(np.float32)
    v_img = ((v) * panorama_h).astype(np.float32)  # Invert vertical axis for image coordinates

    # Use OpenCV to remap the panorama image to the output plane
    map_x, map_y = u_img, v_img
    output_plane = cv2.remap(
        panorama, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    return output_plane, map_x, map_y

def camID2num(cam_id):
    """
    Convert camera ID to number
    """
    return int(cam_id.split('-')[0]) - 1

def getIntrinsics(transformationMatrices, cam_id, img_id):
    """
    Get the intrinsic parameters as matrix from cam_id, img_id
    Args:
    - transformationMatrices: dictionary with all transformation matrices
    - cam_id: camera ID
    - img_id: image ID
    """
    intrinsics = np.ones((3, 3))
    intrinsics[0, 0] = transformationMatrices[cam_id][img_id]['fx']
    intrinsics[1, 1] = transformationMatrices[cam_id][img_id]['fy']
    intrinsics[0, 2] = transformationMatrices[cam_id]['cx']
    intrinsics[1, 2] = transformationMatrices[cam_id]['cy']

    return intrinsics

def getDistortion(transformationMatrices, cam_id, img_id):
    """
    Get the distortion parameters as matrix
    Args:
    - transformationMatrices: dictionary with all transformation matrices
    - cam_id: camera ID
    - img_id: image ID
    """
    distortion = np.zeros(8)
    distortion[0] = transformationMatrices[cam_id][img_id]['k1']
    distortion[1] = transformationMatrices[cam_id][img_id]['k2']
    distortion[2] = transformationMatrices[cam_id][img_id]['p1']
    distortion[3] = transformationMatrices[cam_id][img_id]['p2']
    distortion[4] = transformationMatrices[cam_id][img_id]['k3']
    distortion[5] = transformationMatrices[cam_id][img_id]['k4']
    distortion[6] = transformationMatrices[cam_id][img_id]['k5']
    distortion[7] = transformationMatrices[cam_id][img_id]['k6']

    return distortion

def getW2c(transformationMatrices, cam_id, img_id, config=None, inPix=False):
    """
    Get the world to camera matrix
    Args:
    - transformationMatrices: dictionary with all transformation matrices
    - cam_id: camera ID
    - img_id: image ID
    """
    c2w = np.array(transformationMatrices[cam_id][img_id]['transform_matrix'])

    if inPix:
        t_gps = c2w[:3, 3].tolist()
        t_pix = geoCoord2Open3Dpx(config, t_gps)
        c2w[:3, 3] = t_pix
        
    w2c = np.linalg.inv(c2w)

    return w2c

def transformsGPS2pix(config, path):
    """
    Transform GPS coordinates to pixel coordinates and save to a new file
    Args:
    - config: Config() object, especially with GEE parameters
    - path: relative path to the .json file with GPS coordinates
            .json structure: {"key1": "transform_matrix", "key2": "transform_matrix", ...}
    """
    path = Path(path)
    assert '.json' in path.suffix, 'The file must be a .json file'

    home_path = getHomePath()
    full_path = Path(home_path, path)

    with open(full_path, 'r') as f:
        transforms = json.load(f)

    # change the GPS coordinates to pixel coordinates
    for frame in transforms.values():
        t_gps = [frame['transform_matrix'][0][3], 
                 frame['transform_matrix'][1][3], 
                 frame['transform_matrix'][2][3]]
        t_pix = geoCoord2Open3Dpx(config, t_gps)
        frame['transform_matrix'][0][3] = t_pix[0]
        frame['transform_matrix'][1][3] = t_pix[1]
        frame['transform_matrix'][2][3] = t_pix[2]

    # save the new file
    new_path = str(full_path).replace('.json', '_pix.json')
    with open(new_path, 'w') as f:
        json.dump(transforms, f, indent=2)

def transformsPix2GPS(config, path):
    """
    Transform pixel coordinates to GPS coordinates and save to a new file
    Args:
    - config: Config() object, especially with GEE parameters
    - path: relative path to the .json file with pixel coordinates
            .json structure: {"key1": "transform_matrix", "key2": "transform_matrix", ...}
    """
    assert path.endswith('.json'), 'The file must be a .json file'

    home_path = getHomePath()
    full_path = Path(home_path, path)

    with open(full_path, 'r') as f:
        transforms = json.load(f)

    # change the pixel coordinates to GPS coordinates
    for frame in transforms.values():
        t_pix = frame['transform_matrix'][:3, 3].tolist()
        t_gps = Open3Dpx2geoCoord(config, t_pix)
        frame['transform_matrix'][:3, 3] = t_gps

    # save the new file
    new_path = full_path.replace('.json', '_gps.json')
    with open(new_path, 'w') as f:
        json.dump(transforms, f, indent=2)

def gen_transformsGPS_imgSplit(config):
  """
  Generate transformation matrices for each camera and each image
  TODO: simplify and new function only for splitting up images
  TODO: new flat transformation matrix architecture
  """
  home_path = getHomePath()
  config.loadCamMetadata(config.data_pathMetadata, home_path=home_path)
  dataCamconfig = config.dataCamconfig

  # loop through all cameras and create transformation matrix: c2w
  transform_matrix_json = {}
  scene_path = Path(home_path, dataCamconfig['pathScene'])
  img_path = Path(scene_path, '360')
  imgFolder = 'images'
  imgFrSave_path = Path(scene_path, imgFolder)
  if not imgFrSave_path.exists():
    imgFrSave_path.mkdir(parents=True)
  frame_nr = 0
  for filename in sorted(img_path.iterdir(), key=lambda path: int(path.stem.rsplit("-")[0])):
    tag4transform_matrix = {}
    cam_id = filename.stem
    imgFile_path = Path(img_path, filename)
    
    # split 360 degree into several frames
    #dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] = 2048
    nbr_frames_pr = dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid'] / dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg']
    nbr_frames = int(nbr_frames_pr + 1) # round to next integer
    overlap = int((nbr_frames - nbr_frames_pr) * dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / (nbr_frames - 1) + 1)
    degPerPix = (dataCamconfig['CamOrigMetadata'][cam_id]['TotFOV_wid[°]'] / dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid'])
    translation = np.zeros(3)
    translation[0] = dataCamconfig['CamOrigMetadata'][cam_id]['longitude[°E]_cor']
    translation[1] = dataCamconfig['CamOrigMetadata'][cam_id]['latitude[°N]_cor']
    translation[2] = dataCamconfig['CamOrigMetadata'][cam_id]['altitude[m]_cor']

    # tilt angle rotation
    rot_tilt_x = dataCamconfig['CamOrigMetadata'][cam_id]['TiltAngle[°]_cor']

    # both image and panorama data
    tag4transform_matrix['tilt_angle'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['TiltAngle[°]_cor'] * np.pi / 180) # Tilt in rad
    tag4transform_matrix['camera_angle_hei'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['FOV_hei[°]'] * np.pi / 180) # FOV in rad, TODO: verringern beim zuschneiden
    tag4transform_matrix['nbr_frames'] = np.float64(nbr_frames)
    tag4transform_matrix['hei'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_hei'])
    # per image data
    tag4transform_matrix['camera_angle_wid_img'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['TotFOV_wid[°]'] / nbr_frames_pr * np.pi / 180) # FOV in rad
    tag4transform_matrix['overlapPix_wid_img'] = np.float64(overlap)
    tag4transform_matrix['degPerPix_wid_img'] = np.float64(degPerPix)
    tag4transform_matrix['cx'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2)
    tag4transform_matrix['cy'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_hei'] / 2 + dataCamconfig['CamOrigMetadata'][cam_id]['delta_center'])
    tag4transform_matrix['wid'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'])
    # panorama data
    tag4transform_matrix['wid_pano'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid'])
    tag4transform_matrix['pixel_north_pano'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_north_cor'])
    tag4transform_matrix['camera_angle_wid_pano'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['TotFOV_wid[°]'] * np.pi / 180) # FOV in rad
    tag4transform_matrix['fphi_pano'] = np.float64(tag4transform_matrix['wid_pano'] / tag4transform_matrix['camera_angle_wid_pano'] ) # w / FOV (rad)
    tag4transform_matrix['fy_pano'] = np.float64(tag4transform_matrix['hei'] / (2 * np.tan(tag4transform_matrix['camera_angle_hei'] / 2))) # h / (2*tan(FOV_y/2))
    tag4transform_matrix['cx_pano'] = np.float64(tag4transform_matrix['wid_pano'] / 2)  # fphi_pano * camera_angle_w_pano / 2
    rotationTilt = rotation(rot_tilt_x, 'x', degrees=True)
    r02 = rotationTilt[0, 2]
    r22 = rotationTilt[2, 2]
    tag4transform_matrix['cy_pano'] = np.float64(tag4transform_matrix['fy_pano'] * np.tan(tag4transform_matrix['camera_angle_hei'] / 2 - np.arccos(np.sqrt(r02**2 + r22**2)))) # fy_pano * tan(camera_angle_h / 2 - arccos(sqrt(r02^2 + r22^2)))

    frame = {}
    middle_x_pix = np.int32(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2)
    rotPix = dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] - overlap
    full_img = cv2.imread(imgFile_path)
    rot_horizontal_y = 0
    
    for i in range(nbr_frames):     # generate rotated frames per camera
      frame = {}
      img_id = str(frame_nr) + '_' + cam_id + '_'+ str(i)
      fname = img_id + '.jpg'
      frame_nr += 1
      frame['file_path'] = Path(imgFrSave_path.relative_to(home_path.as_posix()), fname).as_posix()
      frame['fx'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / (2 * np.tan(tag4transform_matrix['camera_angle_wid_img'] / 2))) # f*pixel_x/sensor_w --> sensor_w cuts out
      frame['fy'] = np.float64(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_hei'] / (2 * np.tan(tag4transform_matrix['camera_angle_hei'] / 2))) # f*pixel_h/sensor_h --> sensor_h cuts out
      frame['k1'] = 0
      frame['k2'] = 0
      frame['p1'] = 0
      frame['p2'] = 0    
      frame['k3'] = 0
      frame['k4'] = 0
      frame['k5'] = 0
      frame['k6'] = 0

      if i == 0:    # first frame
        if dataCamconfig['CamOrigMetadata'][cam_id]['pixel_north_cor'] < dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2: # corrected north pixel
          rot_horizontal_y = -(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2 - dataCamconfig['CamOrigMetadata'][cam_id]['pixel_north_cor']) * degPerPix
        else:
          rot_horizontal_y = (dataCamconfig['CamOrigMetadata'][cam_id]['pixel_north_cor'] - dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2) * degPerPix
        middle_x_pix = np.int32(dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2)
      else:
        if i == (nbr_frames - 1):   # last frame
          spare_pix = dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid'] - (dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] + (nbr_frames-1) * rotPix)
          rot_horizontal_y -= (rotPix + spare_pix) * degPerPix
          middle_x_pix += (rotPix + spare_pix)
        else:
          rot_horizontal_y -= rotPix * degPerPix         
          middle_x_pix += rotPix
      
      tmp_img = full_img[:, np.int32(middle_x_pix - dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2):np.int32(middle_x_pix + dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg'] / 2), :]
      # check image shape correctness
      if tmp_img.shape[1] != dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid_perImg']:
        print(f"Warning: Pixel_w_perImg of {fname} {dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid']} does not match shape full img {full_img.shape}")
      if full_img.shape[1] != dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid']:
        print(f"Warning: Config of {fname} {dataCamconfig['CamOrigMetadata'][cam_id]['pixel_wid']} does not match shape full img {full_img.shape}")
      cv2.imwrite(frame['file_path'], tmp_img)

      rotation_angles = [rot_tilt_x, rot_horizontal_y]
      transMatrix = getCamTransform_np(translation, rotation_angles, rotation_axis='xy')
      rot = transMatrix[:3, :3]
      rot = init_to_opengl(rot)
      transMatrix[:3, :3] = rot
      frame['transform_matrix'] = transMatrix.tolist()

      tag4transform_matrix[img_id] = frame

    transform_matrix_json[cam_id] = tag4transform_matrix

  # save the transformation matrix to json file
  save_path = Path(scene_path, 'transformation_matrices', 'transformation_Roundshot.json').as_posix()
  with open(save_path, 'w') as f:
    json.dump(transform_matrix_json, f, indent=2)
    print('transformation matrix saved to', save_path)

def linear_interpolate_transforms(start, end, steps):
    """
    Linear interpolation between two transformation matrices
    Args:
    - start: start transformation matrix
    - end: end transformation matrix
    - steps: number of steps for interpolation
    Returns:
    - trajectory: Array of shape (steps, 4, 4) representing the interpolated trajectory
    """
    # interpolate translation
    steps = int(steps)
    translation = np.linspace(start[:3, 3], end[:3, 3], steps)

    # slerp rotation interpolation
    rot_matrices = np.stack([start[:3, :3], end[:3, :3]])
    rotations = Rotation.from_matrix(rot_matrices)
    key_times = np.array([0, 1])
    slerp = Slerp(key_times, rotations)

    interp_times = np.linspace(0, 1, steps)
    interp_rots = slerp(interp_times)
    rotation = interp_rots.as_matrix()  # Shape: (steps, 3, 3)
    
    # combine translation and rotation
    trajectory = np.eye(4)[np.newaxis, :, :].repeat(steps, axis=0)
    trajectory[:, :3, 3] = translation
    trajectory[:, :3, :3] = rotation

    return trajectory

def smooth_interpolate_waypoints(waypoints, nbr_points):
    """
    Generate a smooth trajectory through waypoints using cubic splines for translation
    and Slerp for rotation.
    Args:
    - waypoints: Array of shape (N, 4, 4) representing N transformation matrices.
    - nbr_points: Total number of points in the output trajectory.
    Returns:
    - trajectory: Array of shape (nbr_points, 4, 4) representing the interpolated trajectory
    """
    N = len(waypoints)  # Number of waypoints
    if N < 2:
        raise ValueError("At least two waypoints are required for interpolation.")

    # Extract translations and rotations from waypoints
    translations = waypoints[:, :3, 3]  # Shape: (N, 3)
    rotations = Rotation.concatenate([Rotation.from_matrix(waypoints[i, :3, :3]) for i in range(N)])

    # Define key times for waypoints (normalized from 0 to 1)
    key_times = np.linspace(0, 1, N)
    traj_times = np.linspace(0, 1, nbr_points)

    # Cubic spline interpolation for translation
    cs_x = CubicSpline(key_times, translations[:, 0])
    cs_y = CubicSpline(key_times, translations[:, 1])
    cs_z = CubicSpline(key_times, translations[:, 2])
    interp_translations = np.vstack([cs_x(traj_times), cs_y(traj_times), cs_z(traj_times)]).T  # Shape: (nbr_points, 3)

    # Slerp for rotation (smooth interpolation across multiple waypoints)
    slerp = Slerp(key_times, rotations)
    interp_rotations = slerp(traj_times).as_matrix()  # Shape: (nbr_points, 3, 3)

    trajectory = np.eye(4)[None, :, :].repeat(nbr_points, axis=0)
    trajectory[:, :3, 3] = interp_translations
    trajectory[:, :3, :3] = interp_rotations

    return trajectory

def smooth_interpolate_values(values, nbr_points):
    """
    Generiert eine glatte Trajektorie durch gegebene Werte mithilfe von kubischen Splines.
    Args:
    - values: Array von Form (N, D) oder (N,) mit N Werten und D Dimensionen.
    - nbr_points: Gesamtzahl der Punkte in der Ausgabetrajektorie.
    Returns:
    - interp_values: Array von Form (nbr_points, D) oder (nbr_points,) mit interpolierten Werten.
    """
    N = len(values)
    if N < 2:
        raise ValueError("Für die Interpolation sind mindestens zwei Werte erforderlich.")

    key_times = np.linspace(0, 1, N)
    traj_times = np.linspace(0, 1, nbr_points)

    if values.ndim == 1:
        # Interpolation für eindimensionale Werte
        cs = CubicSpline(key_times, values)
        interp_values = cs(traj_times)
    else:
        # Interpolation für mehrdimensionale Werte
        interp_values = np.zeros((nbr_points, values.shape[1]))
        for dim in range(values.shape[1]):
            cs = CubicSpline(key_times, values[:, dim])
            interp_values[:, dim] = cs(traj_times)

    return interp_values