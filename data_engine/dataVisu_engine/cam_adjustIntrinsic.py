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


class CamAdjustTool(object):

    def __init__(self, lidar_path, lidar_fName, camera_path, data_id, intrinsics_fileP, config, depth_max=1000):
        self.data_id = data_id
        self.cam_id = data_id.split('_')[1].split('_')[0]
        self.config = config
        self.data_num = int(data_id.split('_')[0])
        #self.cam_num = int(data_id.split('_')[1].split('-')[0]) - 1
        #self.frame_num = int(data_id.split('_')[2].split('.')[0])
        self.depth_max = depth_max
        self.depth_min = 0
        self.point_size = 1

        # Set data directories. The arguments are sanitized by Click, so we don't need to check for an unknown sensor
        camera_data_directory = Path(camera_path)
        self.camera_path = camera_data_directory
        lidar_data_directory = Path(lidar_path)
        lidar_data_file = Path(lidar_data_directory, lidar_fName)
        self.intrinsics_file = Path(intrinsics_fileP)
        self.intrinsics_file_updated = Path(self.intrinsics_file.parent, f"transformation_manCorrected.json")

        if not self.intrinsics_file_updated.exists():
            with open(self.intrinsics_file_updated.as_posix(), 'w') as f:
                json.dump({}, f)

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

        # tkinter variables, rest is initialized in run()
        self.root = tk.Tk()
        self.canvas: tk.Canvas = None
        self.img_ref = None  # image reference of the camera image for the canvas

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        print("Init screen wid, hei:", self.screen_width, self.screen_height)

        # resize image to fit screen
        self.img_scale = np.max([pil_img.size[0] / self.screen_width, pil_img.size[1] / self.screen_height]) + 0.6
        self.img_width = int(pil_img.size[0] // self.img_scale)   # img width on canvas
        self.img_height = int(pil_img.size[1] // self.img_scale)    # img height on canvas
        self.image = np.array(pil_img.resize((self.img_width, self.img_height)))
        print("Init image scale factor:", self.img_scale)

        self.lidar_on_canvas = []  # Stores the reference to the LiDAR points that are drawn on the canvas

        # calibration data
        self.distortion = np.array([])
        self.K: np.array = np.array([])  # camera projection matrix

        # our current best guess of the transformation w2c
        self.c2w = np.array([])  # camera to world transformation matrix: adjustments are done here
        self.w2c = np.array([])  # world to camera transformation matrix

        # init camera calibration
        self.init_calibration(self.intrinsics_file)

    def init_calibration(self, intrinsics_fPath) -> None:
        """
        Initialised the calibration matrices. This is our current best guess for LiDAR->Camera calibration,
        and the camera projection and distortion data
        @return: None
        """

        # Load Intrinsics
        intrinsics_file = Path(intrinsics_fPath)
        with open(intrinsics_file.as_posix(), 'r') as f:
            intrinsics = json.load(f)

        # transform GPS to pixel coordinates
        c2w = np.array(intrinsics[self.data_id]['transform_matrix'])

        t_gps = [c2w[0,3], c2w[1,3], c2w[2,3]]
        t_pix = geoCoord2Open3Dpx(self.config, t_gps)

        c2w[:3, 3] = t_pix
        c2w = opengl_to_opencv(c2w)
        w2c = np.linalg.inv(c2w)
        self.w2c = w2c
        self.c2w = c2w
        
        # translation, rotation, distortion and intrinsics matrix K
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
        
        print("Init c2w:", c2w)
        print("Init w2c:", w2c)
        print("Init c2w translation in pixel/gps:", t_pix, t_gps)

    def reinit_slider(self):
        '''
        Reinitialize the sliders based on the current camera intrinsics and extrinsics
        '''
        # reinit sliders
        self.sl_fx.config(from_=self.K[0,0]-500, to=self.K[0,0]+500)
        self.sl_fx.set(self.K[0,0])
        self.sl_fy.config(from_=self.K[1,1]-500, to=self.K[1,1]+500)
        self.sl_fy.set(self.K[1,1])
        c2w = self.c2w
        t_gps = Open3Dpx2geoCoord(self.config, c2w[:3, 3])
        rot = c2w[:3, :3]
        eulerA = R.from_matrix(rot).as_euler('xyz', degrees=True)
        tilt = -90-eulerA[0]
        horizontal = eulerA[2]
        self.sl_tilt.config(from_=tilt-5, to=tilt+5)
        self.sl_tilt.set(tilt)
        self.sl_horizontal.config(from_=horizontal-30, to=horizontal+30)
        self.sl_horizontal.set(horizontal)
        self.sl_height.config(from_=t_gps[2]-50, to=t_gps[2]+50)
        self.sl_height.set(t_gps[2])

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

    def remove_lidar_from_canvas(self) -> None:
        """
        Removes the LiDAR points from the canvas
        """
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)
        
    def display_image(self) -> None:
        """
        Display the loaded image at full size on the canvas
        """
        # we have to store this variable in self.img_ref because otherwise it's cleared after this function ends
        # and for some reason, tkinter doesn't have access to the data anymore
        #alpha = 0
        #h, w = self.image.shape[:2]
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortion, (w, h), alpha,
        #                                                  (w, h))
        # undist_img = cv2.undistort(self.image, self.K, self.distortion, None, newcameramtx)
        # self.img_ref = ImageTk.PhotoImage(image=Image.fromarray(undist_img))
        self.img_ref = ImageTk.PhotoImage(image=Image.fromarray(self.image))

        self.canvas.create_image(0, 0, anchor='nw', image=self.img_ref)
        self.canvas.pack()
    
    def _update_canvas(self):
        # remove the previously drawn LiDAR points
        self.remove_lidar_from_canvas()

        # get depth max value from entry
        if self.depth_max_entry.get() == '':
            self.depth_max = 1000
        else:
            self.depth_max = int(self.depth_max_entry.get())
        if self.depth_min_entry.get() == '':
            self.depth_min = 0
        else:
            self.depth_min = int(self.depth_min_entry.get())

        # mask points
        w2c = self.w2c
        w2c_t = w2c[:3, 3]
        w2c_rotRod, _ = cv2.Rodrigues(w2c[:3, :3])
        points3d_masked, colors_masked = mask_points_cameraAxis(self.lidar_points_3d, w2c, self.depth_min, self.depth_max, self.lidar_points_colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, w2c_rotRod, w2c_t, self.K, self.distortion)
        projected_points_masked = np.squeeze(projected_points_masked)

        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)

        # set masked points
        self.projected_points_masked = projected_points_masked
        self.lidar_points_3d_masked = points3d_masked
        print('Projected 3D points after masking:', projected_points_masked.shape, 'depth max', self.depth_max)

        # update camera info
        c2w = self.c2w
        eulerA = R.from_matrix(c2w[:3,:3]).as_euler('xyz', degrees=True)
        t_pix = c2w[:3, 3]
        t_gps = Open3Dpx2geoCoord(self.config, t_pix)
        formatted_gps = ",".join([f"{x:.6f}" for x in t_gps])
        formatted_eulerA = ",".join([f"{x:.2f}" for x in eulerA])
        self.camera_info.config(text=f"GPS: {formatted_gps}\n" +
                                    f"rotation euler: {formatted_eulerA}\n" +
                                    f"focal (fx/fy): {self.K[0, 0]:.1f}/{self.K[1, 1]:1f}\n" +
                                    f"principal point (cx/cy): {self.K[0, 2]:.1f}/{self.K[1, 2]:.1f}\n")
        
    def _update_cam(self, event):
        self.point_size = int(self.pointS.get())

        # remove the previously drawn LiDAR points
        self.remove_lidar_from_canvas()

        # update camera intrinsics
        self.K[0, 0] = self.sl_fx.get()
        self.K[1, 1] = self.sl_fy.get()
        tilt = self.sl_tilt.get()
        horizontal = self.sl_horizontal.get()
        height = self.sl_height.get()

        # update translation
        c2w = self.c2w
        t_gps = Open3Dpx2geoCoord(self.config, c2w[:3, 3])
        t_gps[2] = height
        t_pix = geoCoord2Open3Dpx(self.config, t_gps)
        c2w[:3, 3] = t_pix

        # update rotation
        rot = c2w[:3, :3]
        eulerA = R.from_matrix(rot).as_euler('xyz', degrees=True)
        #eulerA[0] = tilt
        eulerA[0] = -90-tilt
        eulerA[2] = horizontal
        rot = (Rotation.from_euler('xyz', eulerA, degrees=True)).as_matrix()
        c2w[:3, :3] = rot
        self.c2w = c2w

        # also update w2c from changed c2w
        w2c = np.linalg.inv(c2w)
        self.w2c = w2c

        # update scene
        self._update_canvas()

        print('Updated c2w:', c2w)
        print('Updated w2c:', w2c)

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
        self.reinit_slider()
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
        self.reinit_slider()
        self._update_canvas()

    def _selectImgNum(self, event):
        # remove old canvas with background image
        self.canvas.delete("all")
        
        self.data_num = int(self.change_entry.get())
        
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
        self.reinit_slider()
        self._update_canvas()

    def _save(self, event):
        # Load Intrinsics new and old
        with open(self.intrinsics_file_updated.as_posix(), 'r') as f:
            intrinsics_updated = json.load(f)
        with open(self.intrinsics_file.as_posix(), 'r') as f:
            intrinsics_old = json.load(f)
        
        # prepare the updated intrinsics
        c2w_local = copy.deepcopy(self.c2w)
        c2w_local = opencv_to_opengl(c2w_local)
        t_pix = c2w_local[:3, 3]
        t_gps = Open3Dpx2geoCoord(self.config, t_pix)
        c2w_local[:3, 3] = t_gps
        img_name = self.data_id + '.jpg'
        cam_id = self.data_id.split('_')[1]

        # get tilt and horizontal angle including before adjustment for pixel north
        rot = c2w_local[:3, :3]
        eulerA = R.from_matrix(rot).as_euler('xyz', degrees=True)
        tilt = round((eulerA[0]-90), 1)
        tilt = tilt * np.pi / 180
        horizontal = eulerA[2]
        c2w_old = np.array(intrinsics_old[self.data_id]['transform_matrix'])
        c2w_old = opengl_to_opencv(c2w_old)
        rot_old = c2w_old[:3, :3]
        eulerA_old = R.from_matrix(rot_old).as_euler('xyz', degrees=True)
        horizontal_old = eulerA_old[2]
        pixel_north_new = int(intrinsics_old[self.data_id]['pixel_north_pano'] - ((horizontal - horizontal_old) % 360) / intrinsics_old[self.data_id]['degPerPix_wid_img'])
        if pixel_north_new < 0:
            pixel_north_new = intrinsics_old[self.data_id]['wid'] + pixel_north_new


        # search if cam_num exists
        if self.data_id in intrinsics_updated:
            intrinsics_updated[self.data_id]['tilt_angle'] = np.float64(tilt)
            intrinsics_updated[self.data_id]['pixel_north_pano'] = np.float64(pixel_north_new)
            intrinsics_updated[self.data_id]['file_path'] = Path(self.camera_path, img_name).as_posix()
            intrinsics_updated[self.data_id]['width'] = np.float64(int(self.img_width * self.img_scale + 1))
            intrinsics_updated[self.data_id]['height'] = np.float64(int(self.img_height * self.img_scale + 1))
            intrinsics_updated[self.data_id]['fx'] = round(self.K[0, 0] * self.img_scale, 1)
            intrinsics_updated[self.data_id]['fy'] = round(self.K[1, 1] * self.img_scale, 1)
            intrinsics_updated[self.data_id]['k1'] = self.distortion[0]
            intrinsics_updated[self.data_id]['k2'] = self.distortion[1]
            intrinsics_updated[self.data_id]['p1'] = self.distortion[2]
            intrinsics_updated[self.data_id]['p2'] = self.distortion[3]
            intrinsics_updated[self.data_id]['k3'] = self.distortion[4]
            intrinsics_updated[self.data_id]['k4'] = self.distortion[5]
            intrinsics_updated[self.data_id]['k5'] = self.distortion[6]
            intrinsics_updated[self.data_id]['k6'] = self.distortion[7]
            intrinsics_updated[self.data_id]['transform_matrix'] = c2w_local.tolist()
            
        # add new cam_num
        else:
            intrinsics_updated[self.data_id] = {'cam_id': cam_id,
                                                     'tilt_angle': tilt,
                                                     'camera_angle_hei': intrinsics_old[self.data_id]['camera_angle_hei'],
                                                     'nbr_frames': intrinsics_old[self.data_id]['nbr_frames'],
                                                     'hei': intrinsics_old[self.data_id]['hei'],
                                                     'camera_angle_wid_img': intrinsics_old[self.data_id]['camera_angle_wid_img'],
                                                     'overlapPix_wid_img': intrinsics_old[self.data_id]['overlapPix_wid_img'],
                                                     'degPerPix_wid_img': intrinsics_old[self.data_id]['degPerPix_wid_img'],
                                                     'cx': np.float64(int(self.K[0, 2] * self.img_scale)),
                                                     'cy': np.float64(int(self.K[1, 2] * self.img_scale)),
                                                     'wid': intrinsics_old[self.data_id]['wid'],
                                                     'wid_pano': intrinsics_old[self.data_id]['wid_pano'],
                                                     'pixel_north_pano': np.float64(pixel_north_new),
                                                     'camera_angle_wid_pano': intrinsics_old[self.data_id]['camera_angle_wid_pano'],
                                                     'fphi_pano': intrinsics_old[self.data_id]['fphi_pano'],
                                                     'fy_pano': intrinsics_old[self.data_id]['fy_pano'],
                                                     'cx_pano': intrinsics_old[self.data_id]['cx_pano'],
                                                     'cy_pano': intrinsics_old[self.data_id]['cy_pano'],
                                                     'file_path': Path(self.camera_path, img_name).as_posix(),
                                                     'width': np.float64(int(self.img_width * self.img_scale + 1)),
                                                     'height': np.float64(int(self.img_height * self.img_scale + 1)),
                                                     'fx': round(self.K[0, 0] * self.img_scale, 1),
                                                     'fy': round(self.K[1, 1] * self.img_scale, 1),
                                                     'k1': self.distortion[0],
                                                     'k2': self.distortion[1],
                                                     'p1': self.distortion[2],
                                                     'p2': self.distortion[3],
                                                     'k3': self.distortion[4],
                                                     'k4': self.distortion[5],
                                                     'k5': self.distortion[6],
                                                     'k6': self.distortion[7],
                                                     'transform_matrix': c2w_local.tolist()}                                                    
            
        # save the updated intrinsics
        with open(self.intrinsics_file_updated.as_posix(), 'w') as f:
            json.dump(intrinsics_updated, f, indent=2)

        print('Saved updated intrinsics')

    def _load_correctedCam(self, event):
        # remove lidar points from canvas
        self.remove_lidar_from_canvas()

        # load corrected camera intrinsics
        self.init_calibration(self.intrinsics_file_updated)

        # add lidar points, reinit sliders
        self.reinit_slider()
        self._update_canvas()    

        print('Loaded corrected camera intrinsics') 


    def run(self):
        img_shape = self.image.shape
        self.canvas = tk.Canvas(self.root, width=img_shape[1] + 10, height=img_shape[0] + 160)
        self.root.title(self.data_id)

        # entry for depth min, max masking, point size
        self.depth_max_entry = tk.Label(self.canvas, text="max")
        self.depth_max_entry.place(x=img_shape[1] - 65, y=img_shape[0] + 5, height=20, width=50, anchor='ne')
        self.depth_max_entry = tk.Entry(self.canvas, width=6)
        self.depth_max_entry.place(x=img_shape[1] - 10, y=img_shape[0] + 5, height=20, width=50, anchor='ne')        
        self.depth_max_entry.insert(0, 1000)
        self.depth_min_entry = tk.Label(self.canvas, text="min")
        self.depth_min_entry.place(x=img_shape[1] - 65, y=img_shape[0] + 25, height=20, width=50, anchor='ne')
        self.depth_min_entry = tk.Entry(self.canvas, width=6)
        self.depth_min_entry.place(x=img_shape[1] - 10, y=img_shape[0] + 25, height=20, width=50, anchor='ne')
        self.depth_min_entry.insert(0, 0)
        self.pointS = tk.Label(self.canvas, text="Point size")
        self.pointS.place(x=img_shape[1] - 65, y=img_shape[0] + 45, height=20, width=80, anchor='ne')
        self.pointS = tk.Entry(self.canvas, width=6)
        self.pointS.place(x=img_shape[1] - 10, y=img_shape[0] + 45, height=20, width=50, anchor='ne')
        self.pointS.insert(0, 1)

        # data location in gps, tilt, horizontal
        c2w = self.c2w
        t_gps = Open3Dpx2geoCoord(self.config, c2w[:3, 3])

        rot = c2w[:3, :3]
        eulerA = R.from_matrix(rot).as_euler('xyz', degrees=True)
        #tilt = eulerA[0]
        tilt = -90-eulerA[0]   # -89 deg --> -1 deg
        horizontal = eulerA[2]

        # Slider for fx, fy, tilt, horizontal, height
        self.fx = tk.Label(self.canvas, text="fx", justify='right')
        self.fx.place(x=415, y=img_shape[0] + 1, height=20, width=65, anchor='nw')
        self.sl_fx = tk.Scale(self.canvas, from_=self.K[0,0]-500, to=self.K[0,0]+500, resolution=0.1, orient="horizontal", length=200)
        self.sl_fx.set(self.K[0,0])
        self.sl_fx.place(x=480, y=img_shape[0] + 1, height=27, width=200, anchor='nw')
        self.fy = tk.Label(self.canvas, text="fy", justify='right')
        self.fy.place(x=415, y=img_shape[0] + 28, height=20, width=65, anchor='nw')
        self.sl_fy = tk.Scale(self.canvas, from_=self.K[1,1]-500, to=self.K[1,1]+500, resolution=0.1, orient="horizontal", length=200)
        self.sl_fy.set(self.K[1,1])
        self.sl_fy.place(x=480, y=img_shape[0] + 28, height=27, width=200, anchor='nw')
        self.tilt = tk.Label(self.canvas, text="tilt °", justify='right')
        self.tilt.place(x=415, y=img_shape[0] + 55, height=20, width=65, anchor='nw')
        self.sl_tilt = tk.Scale(self.canvas, from_=tilt-5, to=tilt+5, resolution=0.1, orient="horizontal", length=200)
        self.sl_tilt.set(tilt)
        self.sl_tilt.place(x=480, y=img_shape[0] + 55, height=27, width=200, anchor='nw')
        self.horizontal = tk.Label(self.canvas, text="horizontal °", justify='right')
        self.horizontal.place(x=415, y=img_shape[0] + 82, height=20, width=65, anchor='nw')
        self.sl_horizontal = tk.Scale(self.canvas, from_=horizontal-30, to=horizontal+30, resolution=0.1, orient="horizontal", length=200)
        self.sl_horizontal.set(horizontal)
        self.sl_horizontal.place(x=480, y=img_shape[0] + 82, height=27, width=200, anchor='nw')
        self.height = tk.Label(self.canvas, text="height [m]", justify='right')
        self.height.place(x=415, y=img_shape[0] + 109, height=20, width=65, anchor='nw')
        self.sl_height = tk.Scale(self.canvas, from_=t_gps[2]-50, to=t_gps[2]+50, resolution=0.1, orient="horizontal", length=200)
        self.sl_height.set(t_gps[2])
        self.sl_height.place(x=480, y=img_shape[0] + 109, height=27, width=200, anchor='nw')

        # Button for updating canvas based on all sliders and entries
        self.update_button = tk.Button(self.canvas, text="Update cam")
        self.update_button.bind("<ButtonPress>", self._update_cam)
        self.update_button.place(x=img_shape[1] - 10, y=img_shape[0] + 70, height=20, width=100, anchor='ne')
        

        # text info for camera location
        formatted_gps = ",".join([f"{x:.6f}" for x in t_gps])
        formatted_eulerA = ",".join([f"{x:.2f}" for x in eulerA])
        self.camera_info = tk.Label(self.canvas, 
                                    text=(f"GPS: {formatted_gps}\n" +
                                        f"rotation euler: {formatted_eulerA}\n" +
                                        f"focal (fx/fy): {self.K[0, 0]:.1f}/{self.K[1, 1]:.1f}\n" +
                                        f"principal point (cx/cy): {self.K[0, 2]:.1f}/{self.K[1, 2]:.1f}\n"),
                                    justify="left")
        self.camera_info.place(x=10, y=img_shape[0] + 5, height=80, width=400, anchor='nw')

        # previous and next button
        self.prev_button = tk.Button(self.canvas, text="Previous")
        self.prev_button.bind("<ButtonPress>", self._previous)
        self.prev_button.place(x=10, y=img_shape[0] + 115, height=20, width=100, anchor='nw')
        self.next_button = tk.Button(self.canvas, text="Next")
        self.next_button.bind("<ButtonPress>", self._next)
        self.next_button.place(x=img_shape[1] - 10, y=img_shape[0] + 115, height=20, width=100, anchor='ne')

        # select specific image number
        self.change_button = tk.Button(self.canvas, text="Change image")
        self.change_button.bind("<ButtonPress>", self._selectImgNum)
        self.change_button.place(x=175, y=img_shape[0] + 115, height=20, width=100, anchor='nw')
        self.change_entry = tk.Entry(self.canvas, width=6)
        self.change_entry.place(x=120, y=img_shape[0] + 115, height=20, width=50, anchor='nw')
        self.change_entry.insert(0, self.data_num)

        # load corrected lidar points
        self.load_corrected_button = tk.Button(self.canvas, text="Load corrected")
        self.load_corrected_button.bind("<ButtonPress>", self._load_correctedCam)
        self.load_corrected_button.place(x=175, y=img_shape[0] + 90, height=20, width=100, anchor='nw')

        # save button
        self.save_button = tk.Button(self.canvas, text="Save")
        self.save_button.bind("<ButtonPress>", self._save)
        self.save_button.place(x=img_shape[1] - 10, y=img_shape[0] + 90, height=20, width=100, anchor='ne')

        self.display_image()

        # 3D points and color
        points = np.array(self.pointcloud.points)
        colors = np.array(self.pointcloud.colors)
        self.lidar_points_3d = points
        self.lidar_points_colors = colors

        # mask points behind the camera and depth max
        w2c = self.w2c
        w2c_t = w2c[:3, 3]
        rotMat = w2c[:3, :3]
        w2c_rotRod, _ = cv2.Rodrigues(rotMat)
        points3d_masked, colors_masked = mask_points_cameraAxis(points, self.w2c, self.depth_min, self.depth_max, colors)
        projected_points_masked, _ = cv2.projectPoints(points3d_masked, w2c_rotRod, w2c_t, self.K, self.distortion)
        projected_points_masked = np.squeeze(projected_points_masked)

        # store the masked points for later use
        self.projected_points_masked = projected_points_masked
        self.lidar_points_3d_masked = points3d_masked
        
        print('Projected 3D points:', projected_points_masked.shape)
        print('3D points masked:', self.lidar_points_3d_masked.shape)


        self.draw_lidar_to_canvas(points3d_masked, colors_masked, projected_points_masked)

        self.root.mainloop()
