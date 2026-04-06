#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Thierry Backes <tbackes@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#
# pylint: disable=line-too-long

import time, json, cv2, math, statistics
import numpy as np
import tkinter as tk
from pathlib import Path
from enum import Enum
from PIL import Image, ImageTk
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

from dataVisu_engine.ThreeD_helpers import loadPointCloud_ply
from utils.camFormat import *

def transformInverse(T):
    # inv(T) = [inv(R) | -inv(R)*t]
    inv_T = np.zeros_like(T) # 4x4
    inv_T[3,3] = 1.0
    inv_T[:3,:3] = np.transpose(T[:3,:3])
    inv_T[:3,3] = np.dot(-np.transpose(T[:3,:3]), T[:3,3])
    return inv_T

def hsv2rgb(h, s, v):
    # Code used from online
    """
    HSV to RGB
    :param float h: 0.0 - 360.0
    :param float s: 0.0 - 1.0
    :param float v: 0.0 - 1.0
    :return: rgb
    :rtype: list
    """
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c

    if 0.0 <= h < 60:
        rgb = (c, x, 0)
    elif 0.0 <= h < 120:
        rgb = (x, c, 0)
    elif 0.0 <= h < 180:
        rgb = (0, c, x)
    elif 0.0 <= h < 240:
        rgb = (0, x, c)
    elif 0.0 <= h < 300:
        rgb = (x, 0, c)
    elif 0.0 <= h < 360:
        rgb = (c, 0, x)
    return list(map(lambda n: (n + m) * 255, rgb))


class States(Enum):
    CAMERA_SELECTION = 1
    LIDAR_SELECTION = 2

class ColorStates(Enum):
    COLOR = 1
    RED = 2


class CamLidarTool(object):

    def __init__(self, lidar_path, lidar_fName, camera_path, data_id, intrinsics_file, config, pointMatch_savePath, depth_max=1000):
        self.data_id = data_id
        self.cam_id = data_id.split('_')[1].split('_')[0]
        self.data_num = int(data_id.split('_')[0])
        #self.cam_num = int(data_id.split('_')[1].split('-')[0]) - 1
        #self.frame_num = int(data_id.split('_')[2].split('.')[0])
        self.config = config
        self.depth_max = depth_max
        self.depth_min = 0
        self.intrinsics_file = intrinsics_file
        self.pointMatch_savePath = pointMatch_savePath
        self.point_size = 1

        # Set data directories. The arguments are sanitized by Click, so we don't need to check for an unknown sensor
        camera_data_directory = Path(camera_path)
        self.camera_path = camera_data_directory
        lidar_data_directory = Path(lidar_path)
        lidar_data_file = Path(lidar_data_directory, lidar_fName)

        # Get the paths of the data
        # listdir returns a list of strings. However we need a (relative or absolute) path. Hence,
        # the filename is prepended with the data directory
        camera_image_paths = [
            camera_data_directory.joinpath(f) for f in sorted(camera_data_directory.iterdir(), key=lambda path: int(path.stem.rsplit("_")[0]))
        ]

        # load image and pcl
        pil_img = Image.open(camera_image_paths[int(self.data_num)])
        self.pointcloud = loadPointCloud_ply(lidar_data_file.parent, lidar_data_file.name)

        self.imgpoints = []  # 2d correspondence points
        self.lidar_correspondences = []  # 3d correspondence points
        print('Init point cloud:', lidar_data_file)

        self.root = tk.Tk()
        self.canvas: tk.Canvas = None
        self.main_button: tk.Button = None
        self.depth_max_entry: tk.Entry = None
        self.depth_min_entry: tk.Entry = None
        self.mask_button: tk.Button = None
        self.button_pressed: bool = False
        self.img_ref = None  # image reference of the camera image for the canvas

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        print("Init screen wid, hei:", self.screen_width, self.screen_height)

        # resize image to fit screen
        self.img_scale = int(np.max([pil_img.size[0] / self.screen_width, pil_img.size[1] / self.screen_height]) + 1)
        self.img_width = pil_img.size[0] // self.img_scale
        self.img_height = pil_img.size[1] // self.img_scale
        self.image = np.array(pil_img.resize((self.img_width, self.img_height)))
        print("Init image scale factor:", self.img_scale,"from", pil_img.size, "to", self.image.shape)

        self.nearest_neighbor: NearestNeighbors = None  # nearest neighbor data structure for LiDAR points

        self.state: States = None  # Stores tue current state from the enum States. Either CAMERA or LIDAR selection
        self.colorState: ColorStates = None

        self.state_label: tk.Label = None  # tkinter label which displays the current state
        self.status_label: tk.Label = None  # tkinter label which displays the current status e.g. "0/10 pairs selected"
        self.camera_info: tk.Label = None
        self.selected_pairs: int = 0  # count the number of selected lidar/camera correspondences

        # holds the selected correspondences
        self.selected_camera_points = []
        self.selected_lidar_indices = []
        self.tk_clicked_points = []  # stores the displayed rectangles when a point is selected

        self.lidar_on_canvas = []  # Stores the reference to the LiDAR points that are drawn on the canvas

        # calibration data
        self.distortion = np.array([])
        self.K: np.array = np.array([])  # camera projection matrix

        # our current best guess of the transformation from lidar to camera
        self.translation: np.array = np.array([])  # translation from lidar to camera
        self.rotation: np.array = np.array(
            [])  # rotation from lidar to camera. This is NOT a rotation matrix,
        # but a rotation vector. Use cv2.Rodrigues() to transform between rotation vector and matrix (and vice versa)

        # init camera calibration
        self.init_calibration(intrinsics_file)
        rot_mat, _ = cv2.Rodrigues(self.rotation)
        print('Init rotation:', rot_mat)

    def init_calibration(self, intrinsics_file) -> None:
        """
        Initialised the calibration matrices. This is our current best guess for LiDAR->Camera calibration,
        and the camera projection and distortion data
        @return: None
        """

        # Load Intrinsics
        with open(intrinsics_file.as_posix(), 'r') as f:
            intrinsics = json.load(f)

        # transform GPS to pixel coordinates
        try:
            c2w = np.array(intrinsics[self.data_id]['transform_matrix'])
        except KeyError as e:
            print(f"WARNING: Old transforms file format with {self.cam_id} key")
            c2w = np.array(intrinsics[self.cam_id][self.data_id]['transform_matrix'])
        t_gps = [c2w[0,3], c2w[1,3], c2w[2,3]]
        t_pix = geoCoord2Open3Dpx(self.config, t_gps)
        print("Init translation in pixel/gps:", t_pix, t_gps)

        c2w[:3, 3] = t_pix
        c2w = opengl_to_opencv(c2w)
        w2c = np.linalg.inv(c2w)
        
        # extract rotation, translation from transformation matrix
        rotation_mat = w2c[:3, :3]
        rotation, _ = cv2.Rodrigues(rotation_mat)
        
        # translation, rotation, distortion and intrinsics matrix K
        self.w2c = w2c
        self.translation = w2c[:3, 3]
        self.rotation = rotation
        try:
            self.distortion = np.array([intrinsics[self.data_id]['k1'],
                                        intrinsics[self.data_id]['k2'],
                                        intrinsics[self.data_id]['p1'],
                                        intrinsics[self.data_id]['p2'],
                                        intrinsics[self.data_id]['k3'],
                                        intrinsics[self.data_id]['k4'],
                                        intrinsics[self.data_id]['k5'],
                                        intrinsics[self.data_id]['k6']], dtype=np.float64)
            self.K = np.array([[intrinsics[self.data_id]['fx'] / self.img_scale, 0, intrinsics[self.data_id]['cx'] / self.img_scale],
                                [0, intrinsics[self.data_id]['fy'] / self.img_scale, intrinsics[self.data_id]['cy'] / self.img_scale],
                                [0, 0, 1]], dtype=np.float64)
        except KeyError:
            self.distortion = np.array([intrinsics[self.cam_id][self.data_id]['k1'],
                                        intrinsics[self.cam_id][self.data_id]['k2'],
                                        intrinsics[self.cam_id][self.data_id]['p1'],
                                        intrinsics[self.cam_id][self.data_id]['p2'],
                                        intrinsics[self.cam_id][self.data_id]['k3'],
                                        intrinsics[self.cam_id][self.data_id]['k4'],
                                        intrinsics[self.cam_id][self.data_id]['k5'],
                                        intrinsics[self.cam_id][self.data_id]['k6']], dtype=np.float64)
            self.K = np.array([[intrinsics[self.cam_id][self.data_id]['fx'] / self.img_scale, 0, intrinsics[self.cam_id]['cx'] / self.img_scale],
                                [0, intrinsics[self.cam_id][self.data_id]['fy'] / self.img_scale, intrinsics[self.cam_id]['cy'] / self.img_scale],
                                [0, 0, 1]], dtype=np.float64)
    
    def change_bgImage(self):
        '''
        Change the background image based on data_num: image number
        '''
        # find image that starts with data_num_
        prefix = f"{self.data_num}_"
        matching_files = find_images_with_prefix(self.camera_path, prefix)
            
        # load image and pcl
        pil_img = Image.open(matching_files[0])
        self.image = np.array(pil_img.resize((self.img_width, self.img_height)))

        self.display_image()

        # set cam_id
        self.cam_id = self.data_id.split('_')[1].split('_')[0]

        self.root.title(self.data_id)

    def draw_lidar_to_canvas(self, points3d_xyz : np.ndarray, points_color : np.ndarray, projected_points: np.ndarray, color_str: str = None) -> None:
        """
        Draws the projected LiDAR points to the canvas with a tkinter rectangle
        @param projected_points: nx2 array of LiDAR points projected to the camera
        @param color: tkinter string of the fill color.
                      Check: http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter for colors
        """                
        # important to display all points as selection is with all points
        for (point, color) in zip(projected_points, points_color):
            x, y = point[0], point[1]
            if (0 <= x and x < self.img_width and 0 <= y and y < self.img_height):
                if color_str is None:
                    color_point = "#%02x%02x%02x" % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                else:
                    color_point = color_str
                point = self.canvas.create_rectangle(x-self.point_size//2, y-self.point_size//2, x+self.point_size//2, y+self.point_size//2, fill=color_point, width=0)
                self.lidar_on_canvas.append(point)  # store a reference to the rectangle so that it can be deleted


    def display_image(self) -> None:
        """
        Display the loaded image at full size on the canvas
        """

        # we have to store this variable in self.img_ref because otherwise it's cleared after this function ends
        # and for some reason, tkinter doesn't have access to the data anymore
        alpha = 0
        h, w = self.image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortion, (w, h), alpha,
                                                          (w, h))
        # undist_img = cv2.undistort(self.image, self.K, self.distortion, None, newcameramtx)
        # self.img_ref = ImageTk.PhotoImage(image=Image.fromarray(undist_img))
        self.img_ref = ImageTk.PhotoImage(image=Image.fromarray(self.image))

        self.canvas.create_image(0, 0, anchor='nw', image=self.img_ref)
        self.canvas.pack()
    
    def _buttonMask_press(self, event):
        self.button_pressed = True

        # disable button as data is now calibrated
        self.main_button.config(state=tk.DISABLED)

        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # old min, max depth values
        old_depth_min = self.depth_min
        old_depth_max = self.depth_max

        # get new depth max value from entry
        if self.depth_max_entry.get() == '':
            self.depth_max = 1000
        else:
            self.depth_max = int(self.depth_max_entry.get())
        if self.depth_min_entry.get() == '':
            self.depth_min = 0
        else:
            self.depth_min = int(self.depth_min_entry.get())

        # handle case of adding more depth and remaining points ordering for selected points
        old_points3d_masked = self.lidar_points_3d_masked
        old_points3d_colors_masked = self.lidar_points_colors_masked
        old_projected_points_masked = self.projected_points_masked
        if old_depth_min == self.depth_min and self.depth_max > old_depth_max:
            points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, self.w2c, old_depth_max, self.depth_max, self.lidar_points_colors)
            projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)
            projected_points_masked = np.squeeze(projected_points_masked)

            self.projected_points_masked = np.vstack((old_projected_points_masked, projected_points_masked))
            self.lidar_points_3d_masked = np.vstack((old_points3d_masked, points3d_masked))
            self.lidar_points_colors_masked = np.vstack((old_points3d_colors_masked, colors_masked))
        
        else:   # delete selected points
            self._buttonDelPairs_press(None)
            
            points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, self.w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
            projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)

            projected_points_masked = np.squeeze(projected_points_masked)  # result has a strange structure. Put back into nx2 matrix

            self.projected_points_masked = projected_points_masked
            self.lidar_points_3d_masked = points3d_masked
            self.lidar_points_colors_masked = colors_masked

        self.draw_lidar_to_canvas(self.lidar_points_3d_masked, self.lidar_points_colors_masked, self.projected_points_masked)

        self.nearest_neighbor = NearestNeighbors(n_neighbors=1,
                                                    algorithm='ball_tree').fit(self.projected_points_masked)

        print('Projected 3D points after masking:', self.projected_points_masked.shape, 'depth max', self.depth_max)

    def _buttonSwitchColor_press(self, event):
        self.button_pressed = True

        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # get depth max value from entry
        if self.depth_max_entry.get() == '':
            self.depth_max = 1000
        else:
            self.depth_max = int(self.depth_max_entry.get())
        if self.depth_min_entry.get() == '':
            self.depth_min = 0
        else:
            self.depth_min = int(self.depth_min_entry.get())

        points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, self.w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)

        projected_points_masked = np.squeeze(projected_points_masked)

        if self.colorState == ColorStates.COLOR:
            self.colorState = ColorStates.RED
            self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)
        elif self.colorState == ColorStates.RED:
            self.colorState = ColorStates.COLOR
            self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked, color_str='red')

    def _buttonContinue_press(self, event):
        self.button_pressed = True

        # disable button as data is now calibrated
        self.main_button.config(state=tk.DISABLED)

        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # indices are packed inside a bunch of arrays. Use list and np.squeeze to get a list of lists for the indices
        indices = np.squeeze(np.array(list(index for index in self.selected_lidar_indices)))

        _, r, t = cv2.solvePnP(np.array(self.lidar_points_3d_masked[indices], dtype=np.float32),
                               np.array(self.selected_camera_points, dtype=np.float32), self.K,
                               self.distortion)
        #_, r, t = cv2.solvePnP(np.array(self.lidar_points_3d_masked[indices], dtype=np.float32),
        #                       np.array(self.selected_camera_points, dtype=np.float32), self.K,
        #                       self.distortion, self.rotation, self.translation, useExtrinsicGuess=True)
        # solvePnPRansac

        print('Rotation PnP:', r)
        print('Translation PnP:', t)

        rotationMat, _ = cv2.Rodrigues(r)
        w2c = np.hstack((rotationMat, t))
        w2c = np.vstack((w2c, np.array([0.0, 0.0, 0.0, 1.0])))
        points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, r, t, self.K, self.distortion)
        #projected_points, _ = cv2.projectPoints(self.lidar_points_3d, r, t, self.K, np.array([0.0, 0.0, 0.0, 0.0]).reshape(4,1))

        projected_points_masked = np.squeeze(
            projected_points_masked)  # result has a strange structure. Put back into nx2 matrix

        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)

        rot_mat, _ = cv2.Rodrigues(r)

        tf2cam = np.hstack((rot_mat, t))
        tf2cam = np.vstack((tf2cam, np.array([0.0, 0.0, 0.0, 1.0])))

        tf2world = np.linalg.inv(tf2cam)
        self.w2c = tf2world

        # c2w for print
        t_pix = tf2world[:3, 3]
        t_gps = Open3Dpx2geoCoord(self.config, t_pix)
        tf2world[:3, 3] = t_gps
        print("Translation in GPS:", tf2world[:3,3])
        print("Rotation (euler):", R.from_matrix(tf2world[:3,:3]).as_euler('xyz', degrees=True))
        print("Rotation (quat):", R.from_matrix(tf2world[:3,:3]).as_quat())

        # append the transformation to json file
        save_path = Path(self.camera_path.parent, 'poseRefinement', f'calibration_{self.data_id}.json')
        if not save_path.exists():
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            
            with open(save_path.as_posix(), 'w') as f:
                json.dump({'0': tf2world.tolist()}, f, indent=2)
        else:
            with open(save_path.as_posix(), 'r') as f:
                data = json.load(f)
                data_keysL = list(map(int, list(data.keys())))
                next_key = max(data_keysL) + 1
            with open(save_path.as_posix(), 'w') as f:
                data[str(next_key)] = tf2world.tolist()
                json.dump(data, f, indent=2)

        # reset for next iteration, remove the pressed points on the canvas
        for cPoint in self.tk_clicked_points:
            self.canvas.delete(cPoint)

        self.rotation = r
        self.translation = t
        self.selected_pairs = 0
        self.selected_lidar_indices = []
        self.selected_camera_points = []
        self.nearest_neighbor = NearestNeighbors(n_neighbors=1,
                                                 algorithm='ball_tree').fit(projected_points_masked)
        self.projected_points_masked = projected_points_masked
        self.lidar_points_3d_masked = points3d_masked
        self.lidar_points_colors_masked = colors_masked
        self.status_label.config(text=f'{self.selected_pairs} point pairs selected (minimum 10)')
        text_tmp = f'Camera (pix/gps): {t_pix}/{t_gps}\n rotation euler: {R.from_matrix(tf2world[:3,:3]).as_euler("xyz", degrees=True)}'
        self.camera_info.config(text=text_tmp)

        print('Projected 3D points:', projected_points_masked.shape)
        print('3D points masked:', self.lidar_points_3d_masked.shape)

    def _buttonDelPairs_press(self, event):
        self.button_pressed = True

        # reset for next iteration, remove the pressed points on the canvas
        for cPoint in self.tk_clicked_points:
            self.canvas.delete(cPoint)

        self.selected_pairs = 0
        self.selected_lidar_indices = []
        self.selected_camera_points = []
        self.status_label.config(text=f'{self.selected_pairs} point pairs selected (minimum 10)')

    def _buttonReset_press(self, event):
        self.button_pressed = True

        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # reset for next iteration, remove the pressed points on the canvas
        for cPoint in self.tk_clicked_points:
            self.canvas.delete(cPoint)

        self.selected_pairs = 0
        self.selected_lidar_indices = []
        self.selected_camera_points = []
        self.status_label.config(text=f'{self.selected_pairs} point pairs selected (minimum 10)')

        # reset calibration
        self.init_calibration(self.intrinsics_file)

        # reload sampled points
        self.depth_min = 0
        self.depth_max = 1000
        points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, self.w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)

        projected_points_masked = np.squeeze(projected_points_masked)  # result has a strange structure. Put back into nx2 matrix

        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)

    def _setPointS(self, event):
        self.button_pressed = True
        self.point_size = int(self.pointS_entry.get())

        self._update_canvas()

        print("Updated point size:", self.point_size)

    def _buttonSaveMatches(self, event):
        self.button_pressed = True

        # save the selected points
        pointMatches = {}
        indices = np.squeeze(np.array(list(index for index in self.selected_lidar_indices)))
        points3d = self.lidar_points_3d_masked[indices]
        pointMatches['match3D'] = points3d.tolist()
        scaled_selection = np.array(self.selected_camera_points) * self.img_scale
        pointMatches['match2D'] = scaled_selection.tolist()
        if not self.pointMatch_savePath.exists():
            if not self.pointMatch_savePath.parent.exists():
                self.pointMatch_savePath.parent.mkdir(parents=True)
            
            with open(self.pointMatch_savePath.as_posix(), 'w') as f:
                json.dump({f'{self.data_id}': pointMatches}, f, indent=2)
        else:
            with open(self.pointMatch_savePath.as_posix(), 'r') as f:
                data = json.load(f)

            # add new data
            data[self.data_id] = pointMatches
            with open(self.pointMatch_savePath.as_posix(), 'w') as f:
                json.dump(data, f, indent=2)

        print('Saved matches to:', self.pointMatch_savePath)

    def _previous(self, event):
        # remove old canvas with background image
        self.canvas.delete("all")

        self.data_num -= 1
        
        # no previous image, roll over to the last image
        if self.data_num < 0:
            max_data_num = 0
            for file in self.camera_path.iterdir():
                data_num = int(file.name.split('_')[0])
                if data_num > max_data_num:
                    max_data_num = data_num
            self.data_num = max_data_num
        
        # find new data_id
        prefix = f'{self.data_num}_'
        new_img_path = find_images_with_prefix(self.camera_path, prefix)
        self.data_id = new_img_path[0].name.split('.')[0]
        self.cam_id = self.data_id.split('_')[1].split('_')[0]
        print('New img path:', new_img_path)

        # update calibration
        self.init_calibration(self.intrinsics_file)

        # add new image, lidar points and reinit sliders
        self.change_bgImage()
        self._update_canvas()

    def _next(self, event):
        # remove old canvas with background image
        self.canvas.delete("all")

        self.data_num += 1
        
        # find img path
        prefix = f'{self.data_num}_'
        new_img_path = find_images_with_prefix(self.camera_path, prefix)

        # no next image --> last one --> start from beginning
        if new_img_path == []:
            self.data_num = 0   # start from beginning
            prefix = f'{self.data_num}_'

        # find new data_id
        self.data_id = new_img_path[0].name.split('.')[0]
        self.cam_id = self.data_id.split('_')[1].split('_')[0]
        print('New img path:', new_img_path)

        # update calibration
        self.init_calibration(self.intrinsics_file)

        # add new image, lidar points and reinit sliders
        self.change_bgImage()
        self._update_canvas()
        
    def _update_canvas(self):
        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # get depth max value from entry
        if self.depth_max_entry.get() == '':
            self.depth_max = 1000
        else:
            self.depth_max = int(self.depth_max_entry.get())
        if self.depth_min_entry.get() == '':
            self.depth_min = 0
        else:
            self.depth_min = int(self.depth_min_entry.get())

        points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, self.w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)

        projected_points_masked = np.squeeze(projected_points_masked)

        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)


    def _entryDepth_in(self, event):
        if self.root.focus_get() == self.depth_min_entry:
            self.depth_min_entry.configure(background='#e1ffe3')
        elif self.root.focus_get() == self.depth_max_entry:
            self.depth_max_entry.configure(background='#e1ffe3')

    def _entryDepth_out(self, event):
        if self.root.focus_get() == self.depth_min_entry:
            self.depth_min_entry.configure(background='white')
        elif self.root.focus_get() == self.depth_max_entry:
            self.depth_max_entry.configure(background='white')

    def _button_release(self, event):  # pylint: disable = unused-argument
        """
        Callback function when the tkinter button is released. Only change button state, this is required for
        the _canvas_press function.
        """
        self.button_pressed = False

    def _canvas_press(self, eventorigin):
        """
        Callback function when the tkinter window gets pressed anywhere. This function handles two
        different cases, whether we are in camera point selection mode, or LiDAR point selection mode.
        For both cases, it stores the selected point in an array, switches the state, and updates the text.

        If the user clicks outside the image or clicks the button, the function doesn't register this as a
        point selection. The LiDAR points are selected by nearest neighbor.
        """
        canvas_x = self.canvas.winfo_rootx()
        canvas_y = self.canvas.winfo_rooty()
        
        x = eventorigin.x_root - canvas_x
        y = eventorigin.y_root - canvas_y

        print('Clicked at:', x, y)

        # Check if the click is on any button
        if (self.button_pressed or y > self.img_height or x > self.img_width):
            return

        if self.state == States.CAMERA_SELECTION:
            self.state = States.LIDAR_SELECTION
            self.state_label.config(text="Select a LiDAR point (brown)")
            rec = self.canvas.create_rectangle(x - 2,
                                               y - 2,
                                               x + 2,
                                               y + 2,
                                               fill="white",
                                               width=1)
            self.tk_clicked_points.append(rec)
            self.selected_camera_points.append([x, y])

            # disable because a corresponding LiDAR point has to be selected first
            self.main_button.config(state=tk.DISABLED)

        elif self.state == States.LIDAR_SELECTION:
            self.state = States.CAMERA_SELECTION
            self.state_label.config(text="Select a camera point (white)")
            indices = self.nearest_neighbor.kneighbors(np.array([[x, y]]),
                                                       n_neighbors=1,
                                                       return_distance=False)

            x = np.squeeze(self.projected_points_masked[indices])[0]
            y = np.squeeze(self.projected_points_masked[indices])[1]
            rec = self.canvas.create_rectangle(x - 2, y - 2, x + 2, y + 2, fill="brown", width=1)
            self.tk_clicked_points.append(rec)
            self.selected_pairs += 1
            self.status_label.config(
                text=f'{self.selected_pairs} point pairs selected (minimum 10)')
            self.selected_lidar_indices.append(indices)

            if self.selected_pairs >= 10:
                self.main_button.config(state=tk.NORMAL)

    def run(self):
        img_shape = self.image.shape
        self.canvas = tk.Canvas(self.root, width=img_shape[1] + 10, height=img_shape[0] + 160)
        self.root.title(self.data_id)

        self.root.bind('<Button 1>', self._canvas_press)  # register callback to handle left mouse clicks on the canvas, selecting points

        # init state
        self.state_label = tk.Label(self.canvas, text="Select a camera point", justify='left')
        self.status_label = tk.Label(self.canvas, text="0 point pairs selected (minimum 10)", justify='left')
        self.state_label.place(x=10, y=img_shape[0] + 5, height=20, width=200, anchor='nw')
        self.status_label.place(x=230, y=img_shape[0] + 5, height=20, width=260, anchor='nw')
        self.state = States.CAMERA_SELECTION
        self.colorState = ColorStates.COLOR

        # text info for camera location
        w2c = self.w2c
        c2w = np.linalg.inv(w2c)
        t_pix = c2w[:3, 3]
        rot = c2w[:3, :3]
        t_gps = Open3Dpx2geoCoord(self.config, t_pix)
        t_pix_str = [round(val, 1) for val in t_pix]
        t_gps_str = [round(val, 6) for val in t_gps]
        rot_str = [round(val, 2) for val in R.from_matrix(rot).as_euler('xyz', degrees=True)]
        self.camera_info = tk.Label(self.canvas, text=f"Camera (pix/gps): {t_pix_str} / {t_gps_str}\n" + f"Rotation euler: {rot_str}", justify='left')
        self.camera_info.place(x=10, y=img_shape[0] + 30, height=80, width=450, anchor='nw')

        # mask button and entry for depth max masking
        self.depth_max_entry = tk.Label(self.canvas, text="max")
        self.depth_max_entry.place(x=520, y=img_shape[0] + 5, height=20, width=30, anchor='nw')
        self.depth_max_entry = tk.Entry(self.canvas, width=6)
        self.depth_max_entry.place(x=550, y=img_shape[0] + 5, height=20, width=40, anchor='nw')        
        self.depth_max_entry.bind("<FocusIn>", self._entryDepth_in)
        self.depth_max_entry.bind("<FocusOut>", self._entryDepth_out)
        self.depth_max_entry.insert(0, 1000)
        self.depth_min_entry = tk.Label(self.canvas, text="min")
        self.depth_min_entry.place(x=520, y=img_shape[0] + 30, height=20, width=30, anchor='nw')
        self.depth_min_entry = tk.Entry(self.canvas, width=6)
        self.depth_min_entry.place(x=550, y=img_shape[0] + 30, height=20, width=40, anchor='nw')
        self.depth_min_entry.bind("<FocusIn>", self._entryDepth_in)
        self.depth_min_entry.bind("<FocusOut>", self._entryDepth_out)
        self.depth_min_entry.insert(0, 0)
        self.mask_button = tk.Button(self.canvas, text="Mask depth")
        self.mask_button.bind("<ButtonPress>", self._buttonMask_press)
        self.mask_button.bind("<ButtonRelease>", self._button_release)
        self.mask_button.place(x=600, y=img_shape[0] + 30, height=20, width=100, anchor='nw')
      
        # entry for point size value and update button
        self.pointS_entry = tk.Entry(self.canvas, width=6)
        self.pointS_entry.place(x=550, y=img_shape[0] + 60, height=20, width=40, anchor='nw')
        self.pointS_entry.bind("<FocusIn>", self._entryDepth_in)
        self.pointS_entry.bind("<FocusOut>", self._entryDepth_out)
        self.pointS_entry.insert(0, 1)
        self.pointS_button = tk.Button(self.canvas, text="Set point size")
        self.pointS_button.bind("<ButtonPress>", self._setPointS)
        self.pointS_button.bind("<ButtonRelease>", self._button_release)
        self.pointS_button.place(x=600, y=img_shape[0] + 60, height=20, width=100, anchor='nw')

        # switch color button
        self.switch_color_button = tk.Button(self.canvas, text="Switch color")
        self.switch_color_button.bind("<ButtonPress>", self._buttonSwitchColor_press)
        self.switch_color_button.bind("<ButtonRelease>", self._button_release)
        self.switch_color_button.place(x=img_shape[1] - 120, y=img_shape[0] + 5, height=20, width=100, anchor='ne')

        # delete selected points button
        self.delete_button = tk.Button(self.canvas, text="Del point pairs")
        self.delete_button.bind("<ButtonPress>", self._buttonDelPairs_press)
        self.delete_button.bind("<ButtonRelease>", self._button_release)
        self.delete_button.place(x=img_shape[1] - 120, y=img_shape[0] + 30, height=20, width=100, anchor='ne')

        # reset to default button
        self.reset_button = tk.Button(self.canvas, text="RESET ALL!", fg="#ff423c")
        self.reset_button.bind("<ButtonPress>", self._buttonReset_press)
        self.reset_button.bind("<ButtonRelease>", self._button_release)
        self.reset_button.place(x=img_shape[1] - 120, y=img_shape[0] + 60, height=20, width=100, anchor='ne')

        # continue button
        self.main_button = tk.Button(self.canvas, text="Continue")
        self.main_button.bind("<ButtonPress>", self._buttonContinue_press)
        self.main_button.bind("<ButtonRelease>", self._button_release)
        self.main_button.place(x=img_shape[1] - 10, y=img_shape[0] + 5, height=20, width=100, anchor='ne')
        self.main_button.config(state=tk.DISABLED)

        # save matching points, 2D and 3D
        self.matching_button = tk.Button(self.canvas, text="Save matches")
        self.matching_button.bind("<ButtonPress>", self._buttonSaveMatches)
        self.matching_button.bind("<ButtonRelease>", self._button_release)
        self.matching_button.place(x=img_shape[1] - 10, y=img_shape[0] + 30, height=20, width=100, anchor='ne')

        # previous and next button
        self.prev_button = tk.Button(self.canvas, text="Previous")
        self.prev_button.bind("<ButtonPress>", self._previous)
        self.prev_button.place(x=10, y=img_shape[0] + 90, height=20, width=100, anchor='nw')
        self.next_button = tk.Button(self.canvas, text="Next")
        self.next_button.bind("<ButtonPress>", self._next)
        self.next_button.place(x=img_shape[1] - 10, y=img_shape[0] + 90, height=20, width=100, anchor='ne')


        self.display_image()

        points = np.array(self.pointcloud.points)
        colors = np.array(self.pointcloud.colors)
        
        self.lidar_points_3d = points
        self.lidar_points_colors = colors

        # mask points behind the camera and depth max
        points3d_masked, colors_masked = mask_points_cameraAxis(points, self.w2c, self.depth_min, self.depth_max, colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, self.rotation, self.translation, self.K, self.distortion)

        projected_points_masked = np.squeeze(projected_points_masked)
        self.projected_points_masked = projected_points_masked
        self.lidar_points_3d_masked = points3d_masked
        self.lidar_points_colors_masked = colors_masked

        self.nearest_neighbor = NearestNeighbors(n_neighbors=1,
                                                 algorithm='ball_tree').fit(projected_points_masked)
        
        print('Projected 3D points:', projected_points_masked.shape)
        print('3D points masked:', self.lidar_points_3d_masked.shape)

        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)

        self.root.mainloop()

