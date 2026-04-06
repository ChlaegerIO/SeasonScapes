import os
import argparse
from pathlib import Path

import torch
import tqdm
import torch.optim as optim
import kaolin as kal
import time
import logging
from PIL import Image
from torch_scatter import scatter_max
from contextlib import contextmanager


from utils.camFormat import *
from dataVisu_engine.ThreeD_helpers import *
from dataPose_optimization.model import *
from torch.utils.tensorboard import SummaryWriter

from dataVisu_engine.ThreeD_helpers import *
from utils.config import *

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class Cam_Optimization():
    def __init__(self, transformMatrices_path, gee_config_path=None, log_name="tst"):
        # read transformation matrices
        self.log_name = log_name
        self.home_path = getHomePath()
        self.transform_matrices = openTFMatrix(transformMatrices_path)
        self.config = Config()
        gee_config_path = Path(gee_config_path)
        if gee_config_path.exists():
            self.config.loadGEE(gee_config_path)
        else:
            self.config.loadGEE(Path(self.home_path, 'configs', 'EarthEngine', 'Scale60_-8512.json'))

        # mesh
        hash = gee_config_path.stem.split('_')[-1].split('.')[0]
        mesh_path = Path(self.home_path, f"data/EarthEngine/mesh_{hash}.ply")
        self.mesh = loadTriMesh_ply(mesh_path.parent, mesh_path.name)
        # self.pcd = loadPointCloud_ply(mesh_path.parent, mesh_path.name)

        # segmentation model
        model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        self.segment = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.segProcessor = SegformerImageProcessor.from_pretrained(model_name, size={"height":1024, "width": 1536})

        # create logger
        self.log_path = Path(self.home_path, 'data_engine', 'Logs', 'current_auto_poseOptim')
        pyLog_fName = Path(self.log_path, f'log_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{self.log_name}.log')
        # create folder for logs
        if not Path(self.log_path).exists():
            Path(self.log_path).mkdir(parents=True)

        self.logger_py = logging.getLogger(__name__)
        logging.basicConfig(filename=pyLog_fName.as_posix(),level=logging.INFO)

        # device init
        is_cuda = (torch.cuda.is_available())
        self.device = torch.device("cuda" if is_cuda else "cpu")
        print(f"Device: {self.device}")

        # training parameters
        self.NBR_SKYLINE_POINTS = 150
        self.POINT_SIZE = 8
        self.NBR_EPOCHS = 251
        self.PRINT_EVERY = 5
        self.FAC_PRINT_IMG = 2
        self.BACKUP_EVERY = 250
        self.LR_POSE = 0.001
        self.LR_FOCAL = 0.1
        self.LR_DISTORT = 1e-10
        self.LEARN_F = True
        self.LEARN_R = True
        self.LEARN_T = True
        self.LEARN_DISTORT = False
        self.FX_ONLY = False
        self.IDENTICAL_T = True  # if True, all cameras have the same translation vector
        self.L_MATCH = 1
        self.L_DELTA_T = 10000
        self.L_INIT_T = 1
        self.L_INIT_DIST = 100000
        self.L_INIT_RY = 10
        self.DIST_PARAMS = 4

        # loss function
        self.criterion = torch.nn.L1Loss()      # optionally: torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()      # binary cross entropy loss for segmentation    

        # normalize vertices and faces
        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=self.device, requires_grad=True)
        self.faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=self.device)
        # self.points3d = torch.tensor(self.pcd.points, dtype=torch.float32, requires_grad=True)
        # self.center = self.vertices.mean(dim=0)
        # vertices_centered = self.vertices - self.center
        # max_extent = torch.max(torch.abs(vertices_centered))
        # self.scale_factor = 1.0 / max_extent
        # self.vertices = vertices_centered * self.scale_factor

    def __call__(self):
        self.Rshot_skyline()
        cam_ids = []
        for data_id in self.transform_matrices.keys():
            cam_id = data_id.split('_')[1]  # e.g. '1-HEISCH' from '0_1-HEISCH_0'
            if cam_id not in cam_ids:
                cam_ids.append(cam_id)
            else:
                continue  # skip if camera id already optimized
            self.logger_py.info(f"Training camera {cam_id}")
            print(f"Training camera {cam_id}")
            self.train_Rshot_cam(cam_id)

    @contextmanager
    def timer(self, name):
        start = time.time()
        yield
        print(f"{name} took {time.time() - start:.2f} seconds")


    def get_points_from_skyline_img(self, skyline_img : torch.tensor, nbr_points=40) -> torch.tensor:
        """
        Get points from skyline segmentation image
        Args:
            skyline_img (torch.tensor): skyline segmentation image
            nbr_points (int): number of points to return
        Returns:
            2D points (torch.tensor) of skyline as list of [row, col]
        """
        img = skyline_img.squeeze()

        points = torch.zeros((nbr_points+1, 2), dtype=torch.float32)
        width = img.shape[1]
        step = (width // nbr_points) + 1

        p = 0
        for i in range(0, width, step):
            column = img[:, i]
            # Find the first row where the pixel is greater than 0
            row_indices = torch.where(column > 0)[0]
            if len(row_indices) > 0:
                points[p, 0] = row_indices[0]
                points[p, 1] = i
            else: # no row found
                points[p, 0] = points[p-1, 0] if p > 0 else 0
                points[p, 1] = i
            p += 1

        return points
    
    def skyline_to_mask(self, skyline_img, H, W):
        """ 
        Convert skyline image to a binary mask, 1 for pixels below the skyline and 0 above.
        Args:
            skyline_img (torch.tensor): skyline segmentation image
            H (int): height of the image
            W (int): width of the image
        """
        mask = torch.zeros((H, W), device=self.device)
        # mask is 1 for pixels below the skyline and 0 above
        pixel_y = 0
        pixel_y_old = 0
        for x in range(W):
            non_zero = skyline_img[:, x].nonzero()[0]
            if len(non_zero) > 0:
                pixel_y = non_zero[0]  # Get the first non-zero pixel in the column
                pixel_y_old = pixel_y
            else:
                pixel_y = pixel_y_old
            mask[int(pixel_y):, x] = 1  # Fill pixels below the skyline
        return mask

    def soft_mask_from_depth(self, depth_map, threshold=1e-4, temperature=1e-2):
        """
        Computes a soft mask where depth > 0 is approximated smoothly.
        
        Args:
            depth_map: Tensor of shape (H, W), with depth values
            threshold: A small positive value to separate foreground/background
            temperature: Controls steepness of the transition

        Returns:
            soft_mask: Tensor of shape (H, W), values in (0, 1)
        """
        # depth < threshold -> background = 0, smooth with sigmoid
        return torch.sigmoid((depth_map - threshold) / temperature)

    
    def get_points_from_skyline_img_diff(self, skyline_img: torch.tensor, nbr_points=40) -> torch.tensor:
        """
        Get points from skyline segmentation image

        Args:
            skyline_img (torch.tensor): skyline segmentation image
            nbr_points (int): number of points to return

        Returns:
            2D points (torch.tensor) of skyline as list of [row, col]
        """
        img = skyline_img.squeeze()

        points = torch.zeros((nbr_points + 1, 2), dtype=torch.float32)
        width = img.shape[1]
        step = (width // nbr_points) + 1

        p = 0
        for i in range(0, width, step):
            column = img[:, i]

            # Use softmax to find the row with the highest probability
            row_indices = torch.softmax(column, dim=0)
            row_index = torch.argmax(row_indices).float()

            if row_index > 0:
                points[p, 0] = row_index
                points[p, 1] = torch.tensor(i, dtype=torch.float32)
            elif p > 0:
                points[p, 0] = points[p - 1, 0]
                points[p, 1] = torch.tensor(i, dtype=torch.float32)
            p += 1

        return points


    def get_skyline_from_points(self, points2D : torch.tensor, downscale=4.0, width=3072, height=2048, bin_width=4, nbr_points=40):
        """    Get skyline from 3D points projected to 2D
        Args:
            points2D (torch.tensor): 2D points projected from 3D points
            downscale (float): downscale factor
            nbr_points (int): number of points to return
        Returns:
            skyline_image (torch.tensor): skyline image with shape (height, width, 1) with values 0 or 1
            selected skyline points (torch.tensor): shape (nbr_points, 2) with [row, col] coordinates
        """
        low_height = int(height / downscale)
        low_width = int(width / downscale)
        # fill skyline_image with highest points
        highest_point_per_col = torch.zeros(low_width, dtype=torch.float32, device=self.device)
        # remove not used points --> more efficient
        points2D = points2D[(points2D[:, 0] >= 0) & (points2D[:, 0] < int(height)) & (points2D[:, 1] >= 0) & (points2D[:, 1] < int(width))]
        for point in points2D:
            row = int(point[0] / downscale)
            col = int(point[1] / downscale)
            if 0 <= row < low_height and 0 <= col < low_width and row > highest_point_per_col[col]:
                highest_point_per_col[col] = row
        
        # TODO: filter highest_point_per_col
            
        skyline_image = torch.zeros((int(height), int(width), 1), dtype=torch.float32, device=self.device)
        previous_height = 0
        for idx, height in enumerate(highest_point_per_col):
            if idx == 0:
                skyline_image[int(height), idx, 0] = 1.0
                previous_height = int(height)
            elif height > 0:
                for sub in range(int(downscale)-1): # interpolate subpixels
                    sub_height = int(previous_height + (height - previous_height) * (sub + 1) / int(downscale))
                    skyline_image[sub_height, idx*int(downscale) - int(downscale) + 1 + sub, 0] = 1.0
                skyline_image[int(height), idx, 0] = 1.0
                previous_height = int(height)
            else:   # fill with previous height
                for sub in range(int(downscale)-1):
                    skyline_image[previous_height, idx*int(downscale) - int(downscale) + 1 + sub, 0] = 1.0
                skyline_image[previous_height, idx, 0] = 1.0

        # select skyline points
        selected_points = self.get_points_from_skyline_img(skyline_image, nbr_points=nbr_points)

        return skyline_image, selected_points
    
    def get_skyline_from_points_diff(self, points2D: torch.tensor, downscale=4.0, width=3072, height=2048, bin_width=4, nbr_points=40):
        """Get skyline from 3D points projected to 2D

        Args:
            points2D (torch.tensor): 2D points projected from 3D points
            downscale (float): downscale factor
            nbr_points (int): number of points to return

        Returns:
            skyline_image (torch.tensor): skyline image with shape (height, width, 1) with values 0 or 1
            selected skyline points (torch.tensor): shape (nbr_points, 2) with [row, col] coordinates
        """
        # low_height = int(height / downscale)
        low_width = int(width / downscale)

        # Initialize highest_point_per_col with differentiable operations
        # highest_point_per_col = torch.zeros(low_width, dtype=torch.float32)

        # Filter points within bounds using differentiable operations
        mask = (points2D[:, 0] >= 0) & (points2D[:, 0] < height) & (points2D[:, 1] >= 0) & (points2D[:, 1] < width)
        points2D = points2D[mask]

        highest_point_per_col = points2D.new_zeros(low_width, dtype=torch.int64)

        # Use differentiable operations to update highest_point_per_col
        points2D = torch.floor(points2D / downscale).long()
        # rows = points2D[:, 0]
        cols_int = points2D[:, 1]

        # Use scatter_max to find the highest point per column
        highest_point_per_col, _ = scatter_max(points2D, cols_int, dim=0, out=highest_point_per_col)

        # Initialize skyline_image
        skyline_image = torch.zeros((int(height), int(width), 1), dtype=torch.float32)

        # Interpolate subpixels using differentiable operations
        for idx in range(low_width):
            current_height = highest_point_per_col[idx]
            if idx == 0:
                skyline_image[int(current_height), idx, 0] = 1.0
                previous_height = current_height
            else:
                for sub in range(int(downscale) - 1):
                    sub_height = previous_height + (current_height - previous_height) * (sub + 1) / downscale
                    skyline_image[int(sub_height), idx * int(downscale) - int(downscale) + 1 + sub, 0] = 1.0
                skyline_image[int(current_height), idx, 0] = 1.0
                previous_height = current_height

        # Select skyline points using differentiable operations
        selected_points = self.get_points_from_skyline_img_diff(skyline_image, nbr_points=nbr_points)

        return skyline_image, selected_points


    def Rshot_skyline(self):
        """
        Rshot skyline segmentation
        Returns:
            2D target points of skyline
        """
        # loop through transform matrices (1. load image, 2. segment, 3. skyline detection, 4. save skyline)
        tqdm_bar = tqdm(self.transform_matrices.items(), total=len(self.transform_matrices), desc="Skyline Segmentation")
        for _, cam in tqdm_bar:
            # checks
            img_path = Path(self.home_path, cam['file_path'])
            skyline_path = img_path.parent.parent / 'skyline' / img_path.name.replace('.jpg', '.png')
            if skyline_path.exists():
                self.logger_py.info(f"Skyline already exists {skyline_path}, skipping")
                continue
            if not img_path.exists():
                self.logger_py.warning(f"No image found {img_path}, skipping")
                continue

            # load image
            img = Image.open(img_path)

            # segment image
            inputs = self.segProcessor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.segment(**inputs)
            logits = outputs.logits.cpu()
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=img.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]

            # skyline detection
            pred_skyline = pred_seg.clone()
            pred_skyline = pred_skyline.squeeze().numpy()
            pred_skyline[pred_seg != 2] = 0
            pred_skyline[pred_seg == 2] = 1
            pred_skyline = cv2.Canny(pred_skyline.astype('uint8') * 255, 100, 200)

            # save skyline
            if not skyline_path.parent.exists():
                skyline_path.parent.mkdir(parents=True)
            cv2.imwrite(skyline_path.as_posix(), pred_skyline)

    def train_Rshot_cam(self, cam_id):
        """
        Train camera parameters for Rshot camera for one cam_id
        Note that the cam_id can be from different timestamps --> batched training
        Args:
            cam_id (str): camera id to train, e.g. '1-HEISCH'
        """  
        fName = f'cam_{cam_id}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        tBoard_fName = Path(self.log_path, fName)
        logger = SummaryWriter(tBoard_fName.as_posix())

        # init camera parameters
        cam = {}
        for data_id, transform in self.transform_matrices.items():
            if cam_id not in data_id:
                continue
            cam[data_id] = transform
        frame_num = len(cam)

        # init camera parameters
        c2w_init = torch.zeros([frame_num, 4, 4], dtype=torch.float32)
        focal_init = torch.zeros([frame_num, 2], dtype=torch.float32)
        dist_init = torch.zeros([frame_num, self.DIST_PARAMS], dtype=torch.float32)
        for i, data_id in enumerate(cam.keys()):
            focal_init[i, 0] = cam[data_id]['fx']
            focal_init[i, 1] = cam[data_id]['fy']
            dist_init[i, 0] = cam[data_id]['k1']
            dist_init[i, 1] = cam[data_id]['k2']
            dist_init[i, 2] = cam[data_id]['p1']
            dist_init[i, 3] = cam[data_id]['p2']
            if self.DIST_PARAMS >= 5:
                dist_init[i, 4] = cam[data_id]['k3']
            if self.DIST_PARAMS >= 6:
                dist_init[i, 5] = cam[data_id]['k4']
                dist_init[i, 6] = cam[data_id]['k5']
                dist_init[i, 7] = cam[data_id]['k6']
            c2w = torch.tensor(cam[data_id]['transform_matrix'])
            t_gps = c2w[:3, 3].tolist()
            t_pix = geoCoord2Open3Dpx(self.config, t_gps)
            c2w[:3, 3] = torch.tensor(t_pix)
            c2w_init[i] = c2w
            width = cam[data_id]['wid']
            height = cam[data_id]['hei']
            cx = cam[data_id]['cx']
            cy = cam[data_id]['cy']

        # send all devices to GPU
        c2w_init = c2w_init.to(self.device)
        focal_init = focal_init.to(self.device)
        dist_init = dist_init.to(self.device)

        print(f"\nTraining for camera: {cam_id} with {frame_num} views")
        print(f"Initial focal length: {focal_init}")
        print(f"Initial distortion parameters: {dist_init}")
        print(f"Initial c2w: {c2w_init}")

        # focal net
        intrinsic_net = LearnFocal(frame_num, self.LEARN_F, width, height, cx, cy, mode='norm', init_focal=focal_init.tolist()).to(device=self.device)
        optimizer_intrinsic = optim.Adam(intrinsic_net.parameters(), lr=self.LR_FOCAL)
        scheduler_intrinsic = torch.optim.lr_scheduler.MultiStepLR(optimizer_intrinsic, milestones=[self.NBR_EPOCHS/4, self.NBR_EPOCHS/2, self.NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)
        # pose net
        pose_net = LearnPose(frame_num, self.LEARN_R, self.LEARN_T, width, height, mode='norm', init_c2w=c2w_init, retDegree=True, identical_t=self.IDENTICAL_T).to(device=self.device)
        optimizer_pose = optim.Adam(pose_net.parameters(), lr=self.LR_POSE)
        scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=[self.NBR_EPOCHS/4, self.NBR_EPOCHS/2, self.NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)
        # distortion parameters
        distort_net = Learn_Distortion(frame_num, self.LEARN_DISTORT, nbr_params=self.DIST_PARAMS, init_dist=dist_init).to(device=self.device)
        optimizer_distort = optim.Adam(distort_net.parameters(), lr=self.LR_DISTORT)
        scheduler_distort = torch.optim.lr_scheduler.MultiStepLR(optimizer_distort, milestones=[self.NBR_EPOCHS/4, self.NBR_EPOCHS/2, self.NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)

        # set to training mode
        intrinsic_net.train()
        pose_net.train()
        distort_net.train()

        # Print model
        self.logger_py.info(f"\nTraining for camera: {cam_id} with {frame_num} views")
        self.logger_py.info(f"Learns focal length: {self.LEARN_F}\nLearns rotation: {self.LEARN_R}\nLearns translation: {self.LEARN_T}\nLearns distortion: {self.LEARN_DISTORT}")
        self.logger_py.info(f"loss factors: L_MATCH={self.L_MATCH}, L_DELTA_T={self.L_DELTA_T}, L_INIT_T={self.L_INIT_T}, L_INIT_DIST={self.L_INIT_DIST}, L_INIT_RY={self.L_INIT_RY}")
        self.logger_py.info(f"Learning rates: LR_POSE={self.LR_POSE}, LR_FOCAL={self.LR_FOCAL}, LR_DISTORT={self.LR_DISTORT}")
        self.logger_py.info(f"Distortion parameters used: {self.DIST_PARAMS}")
        self.logger_py.info(f"Initial focal length: {focal_init}")
        self.logger_py.info(f"Initial distortion parameters: {dist_init}")
        self.logger_py.info(f"Initial c2w: {c2w_init}")
        t0b = time.time()

        ry_init = pose_net.get_r().detach().cpu()[:, 1]

        # get target points from skylines
        img_gt_list = []
        img_gt_backg = []
        img_gt_skyline_list = []
        target_Rshot_points = {}
        mask_target_skyline_dict = {}
        for data_id in cam.keys():
            try:
                fPath = Path(self.home_path, cam[data_id]['file_path'])
            except:
                self.logger_py.warning(f"No file path for {fPath}, use default path data/2024-10-09/12-00-00/imagesPlane/{data_id}.jpg")
                fPath = Path(self.home_path, 'data', '2024-10-09', '12-00-00', 'imagesPlane', f'{data_id}.jpg')

            img_target_skyline = loadRGBImg(fPath.parent.parent / 'skyline' / fPath.name.replace('.jpg', '.png'))
            img = loadRGBImg(fPath)
            img = torch.tensor(img, dtype=torch.float32, requires_grad=False)
            image_size = img.shape
            # target points visualization
            target_Rshot_points[data_id] = self.get_points_from_skyline_img(torch.tensor(img_target_skyline), nbr_points=self.NBR_SKYLINE_POINTS)
            mask_target_skyline_dict[data_id] = self.skyline_to_mask(img_target_skyline, image_size[0], image_size[1]).detach()
            img_target_point = torch.zeros_like(img)
            for p in range(len(target_Rshot_points[data_id])):
                row = int(target_Rshot_points[data_id][p, 0].item())
                col = int(target_Rshot_points[data_id][p, 1].item())
                img_target_point[row-self.POINT_SIZE//2:row+self.POINT_SIZE//2, 
                                 col-self.POINT_SIZE//2:col+self.POINT_SIZE//2] = torch.tensor([0, 255, 0], dtype=torch.float32)
                img_target_point[row, col] = torch.tensor([255, 255, 255], dtype=torch.float32)  # target = green with black middle point
            img_gt_list.append(img_target_point.detach())
            img_gt_skyline_list.append(img_target_skyline)
            img_gt_backg.append(img.detach())

        # training loop
        pbar = tqdm(range(self.NBR_EPOCHS))
        for epoch in pbar:
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # zero gradients
            optimizer_intrinsic.zero_grad()
            optimizer_pose.zero_grad()
            optimizer_distort.zero_grad()

            # forward pass
            intrinsic_lNet = intrinsic_net()
            c2w_pNet = pose_net()
            distort_lNet = distort_net()

            # loss
            loss_dict = {}
            losses = []
            img_infer_list = []
            rendered_mesh_skyline_list = []
            depth_list = []
            nbr_img_per_webcam = 0
            for i, data_id in enumerate(cam.keys()):
                intrinsic = intrinsic_lNet[i]
                distortion = distort_lNet[i]
                c2w = c2w_pNet[i]
                
                # openGL to openCV, point projection
                # c2w = opengl_to_opencv(c2w)
                # w2c = torch.linalg.inv(c2w)

                ######################################### skyline from mesh #########################################
                # openGL to openCV, mesh projection
                # c2w[:3, 3] = (c2w[:3, 3].to(self.device) - self.center) * self.scale_factor

                # filtered_vertices, filtered_faces= get_differentiable_visibility_scores(c2w, self.vertices, self.faces)
                
                c2w = opengl_to_kao(c2w)
                w2c = torch.linalg.inv(c2w)

                intrinsics_kal = kal.render.camera.PinholeIntrinsics.from_focal(width=int(cam[data_id]['wid']), height=int(cam[data_id]['hei']), 
                                                                                      focal_x=intrinsic[0,0], focal_y=intrinsic[1,1], device=self.device)
                # faces must be in64 and thus have no gradient, not possible
                # face_vertices_camera, face_vertices_image = prepare_vertices_ndc(filtered_vertices, filtered_faces, intrinsics=intrinsics_kal, camera_transform=w2c.to(self.device), img_size=image_size)                
                face_vertices_camera, face_vertices_image = prepare_vertices_ndc(self.vertices, self.faces, intrinsics=intrinsics_kal, camera_transform=w2c.to(self.device), img_size=image_size)   
                
                # filter face_vertices_camera and face_vertices_image that are outside of the image size
                face_vertices_image_idx = (face_vertices_image[..., 0] > -cam[data_id]['hei']/500) & (face_vertices_image[..., 0] < cam[data_id]['hei']/500) & \
                                           (face_vertices_image[..., 1] > -cam[data_id]['wid']/500) & (face_vertices_image[..., 1] < cam[data_id]['wid']/500)
                face_vertices_image_idx = face_vertices_image_idx[0,:,0]
                face_vertices_camera = face_vertices_camera[:, face_vertices_image_idx]
                face_vertices_image = face_vertices_image[:, face_vertices_image_idx]

                # with self.timer("render depth"):    # this takes by far the most time! - reason larger dataset of the whole area!
                depth, _ = render_depth_plus_pytorch3d(face_vertices_image, face_vertices_camera[:, :, :, -1], cam[data_id]['hei'], cam[data_id]['wid'], device=self.device)
                
                # with self.timer("Rest Visu prepare garbage"):     # this is fast
                inference_mask = self.soft_mask_from_depth(depth, threshold=1e-4, temperature=1e-2).squeeze(0).squeeze(-1)
                rendered_mesh_skyline = depth.clone()
                # rendered_mesh_skyline[rendered_mesh_skyline > 0] = 1.0    # in-line operation is not good for gradient flow
                rendered_mesh_skyline = torch.where(rendered_mesh_skyline > 0, torch.ones_like(rendered_mesh_skyline), rendered_mesh_skyline)
                sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                rendered_mesh_skyline = rendered_mesh_skyline.squeeze(-1)
                grad_x = F.conv2d(rendered_mesh_skyline, sobel_x, padding=1)
                grad_y = F.conv2d(rendered_mesh_skyline, sobel_y, padding=1)
                rendered_mesh_skyline = torch.sqrt(grad_x**2 + grad_y**2)
                # rendered_mesh_skyline[rendered_mesh_skyline < 0.4] = 0.0  # threshold to get skyline
                # rendered_mesh_skyline[rendered_mesh_skyline >= 0.4] = 1.0
                rendered_mesh_skyline = torch.where(rendered_mesh_skyline < 0.4, torch.zeros_like(rendered_mesh_skyline), rendered_mesh_skyline)
                rendered_mesh_skyline = torch.where(rendered_mesh_skyline >= 0.4, torch.ones_like(rendered_mesh_skyline), rendered_mesh_skyline)
                rendered_mesh_skyline = (rendered_mesh_skyline * 255).to(torch.uint8)
                rendered_mesh_skyline_list.append(rendered_mesh_skyline)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # normalize depth to 0-255
                depth = depth.to(torch.uint8)  # scale depth to 0-255
                depth_list.append(depth.squeeze(-1).detach())
                inference = self.get_points_from_skyline_img(rendered_mesh_skyline, nbr_points=self.NBR_SKYLINE_POINTS)
                ############################################################################################################

                ######################################### skyline from point cloud #########################################
                # downscale = 4.0
                # points2D = render_3d_w2c_opencv_torch(self.points3d, w2c, intrinsic, distortion)
                # # inference = points2D[0:self.NBR_SKYLINE_POINTS, :2]
                # # infer_skyline = inference.clone()
                # # infer_skyline, inference = self.get_skyline_from_points(points2D, downscale, width, height, self.NBR_SKYLINE_POINTS)
                # infer_skyline, inference = self.get_skyline_from_points_diff(points2D, downscale, width, height, self.NBR_SKYLINE_POINTS)
                # rendered_mesh_skyline_list.append(infer_skyline)
                ############################################################################################################

                target = torch.tensor(target_Rshot_points[data_id], dtype=torch.float32, requires_grad=True)
                target_mask = torch.tensor(mask_target_skyline_dict[data_id], dtype=torch.float32, requires_grad=True)
                # self.logger_py.info(f"View {i}, data_id {data_id}, target points: {target.shape}, inference points: {inference.shape}")

                # visualize inference points
                if self.PRINT_EVERY > 0 and (epoch % (self.PRINT_EVERY*self.FAC_PRINT_IMG)) == 0 or self.BACKUP_EVERY > 0 and (epoch % self.BACKUP_EVERY) == 0:
                    img_infer = torch.zeros_like(img_gt_list[0])
                    for p in range(inference.shape[0]):
                        if inference[p, 1] < 0 or inference[p, 1] >= width or inference[p, 0] < 0 or inference[p, 0] >= height:
                            continue
                        img_infer[int(inference[p,0])-self.POINT_SIZE//2:int(inference[p,0])+self.POINT_SIZE//2, 
                                int(inference[p,1])-self.POINT_SIZE//2:int(inference[p,1])+self.POINT_SIZE//2] = torch.tensor([255, 0, 0], dtype=torch.float32)
                        img_infer[int(inference[p,0]), int(inference[p,1])] = torch.tensor([255, 255, 255], dtype=torch.float32)  # infered = red with black middle point
                    img_infer_list.append(img_infer)

                # num_points = len(target)
                # if num_points != 0 and len(target) == len(inference):
                #     nbr_img_per_webcam += 1
                #     matching_loss += self.criterion(inference, target)
                # else:
                #     self.logger_py.warning(f"Mask sum is zero for view {i} or target len {len(target)} and inference len {len(inference)} different")

                # calculate bce loss
                nbr_img_per_webcam += 1
                losses.append(self.bce_loss(inference_mask, target_mask))
            # loss calculation
            matching_loss = torch.stack(losses).mean()

            # get variables
            t_init_loc = c2w_init[0, :3, 3].clone().detach().cpu()
            t_poses_loc = c2w_pNet[:, :3, 3].clone().detach().cpu()
            dist_init_loc = dist_init.clone().detach().cpu()
            distort_cur_loc = distort_lNet.clone().detach().cpu()
            # normalize
            normalize_t = torch.tensor([width, height, 70], dtype=torch.float32)
            t_poses_loc = t_poses_loc / normalize_t.repeat(frame_num, 1)
            t_init_loc = t_init_loc / normalize_t
            # loss terms
            if len(t_poses_loc) == 1:
                loss_deltaT = torch.tensor(0.0, dtype=torch.float32)
            else:
                loss_deltaT = torch.var(t_poses_loc, axis=0).sum()
            loss_deltaTInit = torch.norm(t_poses_loc - t_init_loc, p=2)
            loss_distInit = torch.norm(distort_cur_loc - dist_init_loc, p=2)
            loss_ryInit = torch.norm(pose_net.get_r().detach().cpu()[:, 1] - ry_init, p=1)
            # total loss with factors
            loss = self.L_MATCH*matching_loss + self.L_DELTA_T*loss_deltaT + self.L_INIT_T*loss_deltaTInit + self.L_INIT_DIST*loss_distInit + self.L_INIT_RY*loss_ryInit
            if loss.isnan() or loss.isinf():
                loss = self.L_MATCH*matching_loss
                self.logger_py.warning(f"Loss is NaN of Inf, using only matching loss")

            # loss logging
            loss_dict['matching_loss'] = matching_loss.detach().item()
            loss_dict['loss_deltaT'] = loss_deltaT.detach().item()
            loss_dict['loss_deltaTInit'] = loss_deltaTInit.detach().item()
            loss_dict['loss_distInit'] = loss_distInit.detach().item()
            loss_dict['loss_ryInit'] = loss_ryInit.detach().item()
            loss_dict['total_loss'] = loss.detach().item()

            # backward pass
            loss.backward(retain_graph=True)  # retain_graph=True to keep the graph for the next iteration (works, but more memory)
            # loss.backward()

            # update parameters
            optimizer_intrinsic.step()
            optimizer_pose.step()
            optimizer_distort.step()

            # print loss
            # with self.timer("print loss"):   # takes also 7.5s with image visualization
            if self.PRINT_EVERY > 0 and (epoch % self.PRINT_EVERY) == 0 or epoch == 0:
                tqdm.write(f'Epoch {epoch}, loss={loss}, l_match={matching_loss}, l_dist={loss_distInit}, L_INIT_RY={loss_ryInit}, l_t={loss_deltaT}, l_tInit={loss_deltaTInit}, time={time.time()-t0b}')
                self.logger_py.info(f'Epoch {epoch}, loss={loss}, time={time.time()-t0b}')
                t0b = time.time()

                logger.add_scalar('lr_rate/intrinsic', optimizer_intrinsic.param_groups[0]['lr'], epoch)
                logger.add_scalar('lr_rate/pose', optimizer_pose.param_groups[0]['lr'], epoch)
                logger.add_scalar('lr_rate/distort', optimizer_distort.param_groups[0]['lr'], epoch)
                logger.add_histogram('params/fx0', intrinsic_net.fx[0].detach().cpu(), epoch)
                logger.add_histogram('params/fy0', intrinsic_net.fy[0].detach().cpu(), epoch)
                logger.add_histogram('params/rx0', pose_net.get_r().detach().cpu()[:, 0], epoch)    # rotation axis of all views
                logger.add_histogram('params/ry0', pose_net.get_r().detach().cpu()[:, 1], epoch)
                logger.add_histogram('params/rz0', pose_net.get_r().detach().cpu()[:, 2], epoch)
                logger.add_histogram('params/tx0', pose_net.get_t().detach().cpu()[:, 0], epoch)
                logger.add_histogram('params/ty0', pose_net.get_t().detach().cpu()[:, 1], epoch)
                logger.add_histogram('params/tz0', pose_net.get_t().detach().cpu()[:, 2], epoch)
                logger.add_histogram('params/k1_0', distort_net.get_distortionAll()[:, 0].detach().cpu(), epoch)
                logger.add_histogram('params/k2_0', distort_net.get_distortionAll()[:, 1].detach().cpu(), epoch)
                logger.add_histogram('params/p1_0', distort_net.get_distortionAll()[:, 2].detach().cpu(), epoch)
                logger.add_histogram('params/p2_0', distort_net.get_distortionAll()[:, 3].detach().cpu(), epoch)
                if self.DIST_PARAMS >= 5:
                    logger.add_histogram('params/k3_0', distort_net.get_distortionAll()[:, 4].detach().cpu(), epoch)
                if self.DIST_PARAMS >= 6:
                    logger.add_histogram('params/k4_0', distort_net.get_distortionAll()[:, 5].detach().cpu(), epoch)
                    logger.add_histogram('params/k5_0', distort_net.get_distortionAll()[:, 6].detach().cpu(), epoch)
                    logger.add_histogram('params/k6_0', distort_net.get_distortionAll()[:, 7].detach().cpu(), epoch)

                # only visualize every other epoch
                if epoch % (self.PRINT_EVERY*self.FAC_PRINT_IMG) == 0:
                    for idx in range(frame_num):
                        ovlImg = cv2.addWeighted(np.array(img_gt_backg[idx]), 0.8, np.array(img_gt_list[idx]), 0.6, 0)
                        ovlImg = cv2.addWeighted(ovlImg, 0.8, np.array(img_infer_list[idx]), 0.6, 0)
                        # add skyline lines
                        colored_skyline = img_gt_skyline_list[idx].astype(np.float32)
                        colored_skyline[:, :, 0] = colored_skyline[:, :, 0]*0.0
                        colored_skyline[:, :, 2] = colored_skyline[:, :, 2]*0.0
                        ovlImg = cv2.addWeighted(ovlImg, 0.8, colored_skyline, 0.6, 0)
                        _, binary = cv2.threshold(rendered_mesh_skyline_list[idx].permute(1, 2, 0).detach().cpu().numpy(), 1, 255, cv2.THRESH_BINARY)
                        colored_mesh_skyline = np.zeros((rendered_mesh_skyline_list[idx].detach().cpu().numpy().shape[1], rendered_mesh_skyline_list[idx].detach().cpu().numpy().shape[2], 3), dtype=np.float32)
                        colored_mesh_skyline[binary > 0] = [255.0, 0, 0]  # red color
                        ovlImg = cv2.addWeighted(ovlImg, 0.8, colored_mesh_skyline, 0.6, 0)
                        ovlImg = np.transpose(ovlImg, (2, 0, 1))
                        ovlImg = ovlImg / 255.0
                        logger.add_image(f'skyline points image {idx}', ovlImg, epoch)
                        logger.add_image(f'mesh depth {idx}', depth_list[idx].cpu().numpy(), epoch)
                        # self.logger_py.info(f"Epoch {epoch}, depth min/max: {depth_list[idx].cpu().numpy().min()}/{depth_list[idx].cpu().numpy().max()}")

            for l, num in loss_dict.items():
                if l == 'total_loss':
                    logger.add_scalar(f'train_{self.L_MATCH}*l_match + {self.L_DELTA_T}*l_deltaT + {self.L_INIT_T}*l_InitT + {self.L_INIT_DIST}*l_InitDist + {self.L_INIT_RY}*L_INIT_RY/{l}', num, epoch)
                else:
                    logger.add_scalar(f'train/{l}', num, epoch)
            self.logger_py.info(f'Epoch {epoch}, loss={loss}, l_match={matching_loss}, l_dist={loss_distInit}, L_INIT_RY={loss_ryInit}, l_t={loss_deltaT}, l_tInit={loss_deltaTInit}, time={time.time()-t0b}')

            # update learning rate
            scheduler_intrinsic.step()
            scheduler_pose.step()
            scheduler_distort.step()

            # save checkpoints
            if self.BACKUP_EVERY > 0 and (epoch % self.BACKUP_EVERY) == 0:
                transMatrix_path = Path(fPath.parent.parent, f'transformation_{epoch}epoch.json')
                if transMatrix_path.exists():
                    with open(transMatrix_path.as_posix()) as f:
                        transform_ckpt = json.load(f)
                else:
                    transform_ckpt = self.transform_matrices

                for idx, data_id in enumerate(cam.keys()):
                    c2w = pose_net.get_c2w(idx).cpu().detach().numpy()
                    trans_gps = Open3Dpx2geoCoord(self.config, c2w[:3, 3])
                    c2w[:3, 3] = trans_gps

                    transform_ckpt[data_id]['transform_matrix'] = c2w.tolist()
                    transform_ckpt[data_id]['fx'] = intrinsic_lNet[idx, 0, 0].detach().cpu().item()
                    transform_ckpt[data_id]['fy'] = intrinsic_lNet[idx, 1, 1].detach().cpu().item()
                    transform_ckpt[data_id]['k1'] = distort_lNet[idx, 0].detach().cpu().item()
                    transform_ckpt[data_id]['k2'] = distort_lNet[idx, 1].detach().cpu().item()
                    transform_ckpt[data_id]['p1'] = distort_lNet[idx, 2].detach().cpu().item()
                    transform_ckpt[data_id]['p2'] = distort_lNet[idx, 3].detach().cpu().item()
                    transform_ckpt[data_id]['k3'] = 0.0
                    transform_ckpt[data_id]['k4'] = 0.0
                    transform_ckpt[data_id]['k5'] = 0.0
                    transform_ckpt[data_id]['k6'] = 0.0
                    if self.DIST_PARAMS >= 5:
                        transform_ckpt[data_id]['k3'] = distort_lNet[idx, 4].detach().cpu().item()
                    if self.DIST_PARAMS >= 6:
                        transform_ckpt[data_id]['k4'] = distort_lNet[idx, 5].detach().cpu().item()
                        transform_ckpt[data_id]['k5'] = distort_lNet[idx, 6].detach().cpu().item()
                        transform_ckpt[data_id]['k6'] = distort_lNet[idx, 7].detach().cpu().item()

                    self.logger_py.info(f'Backup epoch {epoch}: {transform_ckpt[data_id]}')

                # write to file
                with open(transMatrix_path.as_posix(), 'w') as f:
                    json.dump(transform_ckpt, f, indent=2)
                f.close()

                self.logger_py.info(f'Backup epoch {epoch}: {time.strftime("%Y-%m-%d_%H-%M-%S")} to {transMatrix_path}')

            # profile timing of training epoch
            # self.logger_py.info(prof.key_averages().table(sort_by="self_cpu_time_total"))

            pbar.set_description(f'Epoch {epoch}, loss={loss}')
