
import json
import numpy as np
from pathlib import Path
from dataVisu_engine.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from utils.general import *
from utils.camFormat import *

POSE_PATH = "data/2024-10-09/12-00-00"
POSE_FILE = "transformation.json"
RANGE_VISU = [[7.6, 8.3], [46.5 , 46.8], [500, 4200]]  # x, y, z, [7.9, 8.1], [46.5 , 46.8], [500, 4200]
FL_SCALE = 0.02
ASPRATIO = 0.5
NEW_TRANS = True
VIS_2D = False


def fastAngleCheck(data):
    '''
    Fast angle check for the camera poses, angle(v1, v2) < (FOV1 / 2 + FOV2 / 2)
    v1/2: view direction of camera 1/2
    FOV1/2: field of view of camera
    '''
    for idx_c1 in range(len(data)):
        for idx_c2 in range(len(data)):
            if idx_c1 == idx_c2:
                continue

            # horizontal FOV
            FOV1 = data[str(idx_c1)]['camera_angle_wid_img']*180/np.pi
            FOV2 = data[str(idx_c2)]['camera_angle_wid_img']*180/np.pi
            fov_sum = (FOV1 / 2 + FOV2 / 2)
            overlapping = []
            for idx_img1 in range(len(data[str(idx_c1)]['frames'])):
                for idx_img2 in range(len(data[str(idx_c2)]['frames'])):
                    # get the view direction of the camera
                    rot_matrix = np.array(data[str(idx_c1)]['frames'][idx_img1]['transform_matrix'])[:3, :3]
                    view_dir1 = get_rotation_angle(rot_matrix, degree=True)
                    rot_matrix = np.array(data[str(idx_c2)]['frames'][idx_img2]['transform_matrix'])[:3, :3]
                    view_dir2 = get_rotation_angle(rot_matrix, degree=True)

                    # calculate the angle
                    angle = np.arccos(np.dot(view_dir1, view_dir2) / (np.linalg.norm(view_dir1) * np.linalg.norm(view_dir2)))
                    if angle < fov_sum:         # overlapping view
                        angle = angle * 180 / np.pi
                        overlapping.append((idx_c1, idx_img1, idx_c2, idx_img2, angle))
                        #print('Camera {} and {}, Image {} and {} overlapping: Angle {}, FOV1: {} FOV2 {}'.format(idx_c1, idx_c2, idx_img1, idx_img2, angle, FOV1, FOV2))
                               
    home_path = getHomePath()
    save_path = Path(home_path, POSE_PATH, 'overlapping.txt')
    with open(save_path, 'w') as f:
        for item in overlapping:
            f.write(f"cam1.img {item[0]}.{item[1]} cam2.img {item[2]}.{item[3]}. Angle {item[4]}\n")

            
if __name__ == '__main__':

    home_path = getHomePath()
    poseF_path = Path(home_path, POSE_PATH, POSE_FILE).as_posix()
    with open(poseF_path, 'r') as f:
        data = json.load(f)
    org_cam_poses_c2w = []
    nbr_cams = len(data)
    for idx in range(nbr_cams):
        poses_frame = data[str(idx)]['frames']
        for i in range(len(poses_frame)):
            pose = np.array(poses_frame[i]['transform_matrix'])
            if NEW_TRANS:
                rotX = R.from_euler('x', 90, degrees=True)
                pose[:3, :3] = rotX.as_matrix() @ pose[:3, :3]
            if VIS_2D:
                pose[2, 3] = 0
                RANGE_VISU[2] = [0, 2]
            pose = opengl_to_opencv(pose)

            org_cam_poses_c2w.append(pose)

    org_cam_poses_c2w = np.array(org_cam_poses_c2w)

    centers = org_cam_poses_c2w[:, :3, 3]

    print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))

    visualizer = CameraPoseVisualizer(RANGE_VISU[0], RANGE_VISU[1], RANGE_VISU[2])

    K  = np.array([[2317.0, 0, 500, 0],[0, 2317.0, 500, 0], [0,0,1, 0],[0,0, 0,1]])
    img_size = np.array([1000,1000]) # W, H

    for index, frame_i in enumerate(org_cam_poses_c2w[::1]):
        visualizer.extrinsic2pyramid(frame_i, plt.cm.rainbow(index / len(org_cam_poses_c2w)), focal_len_scaled=FL_SCALE, aspect_ratio=ASPRATIO)

    fastAngleCheck(data)

    visualizer.show()
    # save visualization to file
    fName = 'camera_poses.png'
    if VIS_2D:
        fName = 'camera_poses_2D.png'
    save_path = Path(home_path, POSE_PATH, fName)
    visualizer.fig.savefig(save_path.as_posix(), dpi=300)

