import torch
import torch.nn as nn
from kornia.geometry.camera.perspective import project_points
import torch.nn.functional as F
import kaolin as kal
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.structures import Meshes

from utils.camFormat import *


class LearnFocal(nn.Module):
    """
    Learn focal length
    In model call the forward function returns list of focal lengths [fx, fy]

    :param num_cam:  number of cameras/frames
    :param req_grad:  True/False
    :param mode:     '1', '2', 'norm' (with normaize factor)
    :param init_focal:  None or list [fx, fy]
    :param width, height:     width, height of the image
    :param cx, cy:       x,y focal point
    """
    def __init__(self, num_cam, req_grad, width, height, cx, cy, mode='norm', init_focal=1):
        super(LearnFocal, self).__init__()

        self.mode = mode  # check our supplementary section.
        self.normalize_x = width
        self.normalize_y = height
        self.cx = torch.tensor(cx, dtype=torch.float32, requires_grad=False)
        self.cy = torch.tensor(cy, dtype=torch.float32, requires_grad=False)
        eye = torch.eye(3, dtype=torch.float32)
        eye = eye.reshape((1, 3, 3))
        self.intrinsic = eye.repeat(num_cam, 1, 1)        # intrinsic (N, 3, 3)

        if isinstance(init_focal, list):
            init_focal = torch.tensor(init_focal, dtype=torch.float32)
            if self.mode == '2':
                # a**2 * W = fx  --->  a**2 = fx / W
                coe_x = torch.ones(size=(num_cam, 1), dtype=torch.float32) * torch.sqrt(init_focal[:, 0])
                coe_y = torch.ones(size=(num_cam, 1), dtype=torch.float32) * torch.sqrt(init_focal[:, 1])
            elif self.mode == '1':
                # a * W = fx  --->  a = fx / W
                coe_x = torch.ones(size=(num_cam, 1), dtype=torch.float32) * init_focal[:, 0]
                coe_y = torch.ones(size=(num_cam, 1), dtype=torch.float32) * init_focal[:, 1]
            elif self.mode == 'norm':
                # fx learn = fx / normalize_factor
                coe_x = torch.mul(torch.ones(size=(num_cam, 1), dtype=torch.float32), init_focal[:, 0]) / self.normalize_x
                coe_y = torch.mul(torch.ones(size=(num_cam, 1), dtype=torch.float32), init_focal[:, 1]) / self.normalize_y
            else:
                coe_x = torch.rand(size=(num_cam, 1), dtype=torch.float32)
                coe_y = torch.rand(size=(num_cam, 1), dtype=torch.float32)
                print('WARNING: Mode need to be "1" or "2" or "norm" - random init')
            self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (N, 1)
            self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (N, 1)
        else:
            print('WARNING: focal not a list')

    def forward(self):  # the i=None is just to enable multi-gpu training
        '''
        Returns the intrinsic matrix
        '''
        if self.mode == '1':
            fx = self.fx
            fy = self.fy
        elif self.mode == '2':
            fx = self.fx**2
            fy = self.fy**2
        elif self.mode == 'norm':
            fx = self.fx * self.normalize_x
            fy = self.fy * self.normalize_y
        else:
            fx = self.fx
            fy = self.fy
            print('WARNING: Mode need to be "1" or "2" or "norm"')
        
        self.intrinsic[:, 0, 0] = fx[0,:]
        self.intrinsic[:, 1, 1] = fy[0,:]
        self.intrinsic[:, 0, 2] = self.cx
        self.intrinsic[:, 1, 2] = self.cy
        return self.intrinsic
    
    def get_fx(self):
        if self.mode == '1':
            return self.fx
        elif self.mode == '2':
            return self.fx ** 2
        elif self.mode == 'norm':
            return self.fx * self.normalize_x
        else:
            print('WARNING: Mode need to be "1" or "2" or "norm"')
            return self.fx

    def get_fy(self):
        if self.mode == '1':
            return self.fy
        elif self.mode == '2':
            return self.fy ** 2
        elif self.mode == 'norm':
            return self.fy * self.normalize_y
        else:
            print('WARNING: Mode need to be "1" or "2" or "norm"')
            return self.fy
    

class Learn_Distortion(nn.Module):
    def __init__(self, num_cams, require_grad, nbr_params, init_dist):
        """
        Depth distortion parameters
        In model call the forward function returns list of distortion parameters
        [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]

        Args:
            num_cams (int): number of cameras/frames
            require_grad (bool): True/False for learning
            nbr_params (int): 4, 5, 8, 12, 14 distortion parameters
            init_dist (tensor): (num_cams, nbr_params)
                                [k1, k2, p1, p2, 
                                    k3, 
                                        k4, k5, k6
                                            s1, s2, s3, s4
                                                taux, tauy]
        """
        super(Learn_Distortion, self).__init__()
        self.nbr_params = nbr_params
        init_dist = init_dist.detach().cpu()  # (nbr_params, num_cams)
        self.distortion = init_dist.clone() # (N, 14)

        if self.nbr_params == 4:
            self.k1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 0]), requires_grad=require_grad)
            self.k2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 1]), requires_grad=require_grad)
            self.p1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 2]), requires_grad=require_grad)
            self.p2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 3]), requires_grad=require_grad)
            self.k3 = None
            self.k4 = None
            self.k5 = None
            self.k6 = None
            self.s1 = None
            self.s2 = None
            self.s3 = None
            self.s4 = None
            self.taux = None
            self.tauy = None
        elif self.nbr_params == 5:
            self.k1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 0]), requires_grad=require_grad)
            self.k2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 1]), requires_grad=require_grad)
            self.p1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 2]), requires_grad=require_grad)
            self.p2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 3]), requires_grad=require_grad)
            self.k3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 4]), requires_grad=require_grad)
            self.k4 = None
            self.k5 = None
            self.k6 = None
            self.s1 = None
            self.s2 = None
            self.s3 = None
            self.s4 = None
            self.taux = None
            self.tauy = None
        elif self.nbr_params == 8:
            self.k1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 0]), requires_grad=require_grad)  # (N, 1)
            self.k2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 1]), requires_grad=require_grad)
            self.p1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 2]), requires_grad=require_grad)
            self.p2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 3]), requires_grad=require_grad)
            self.k3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 4]), requires_grad=require_grad)
            self.k4 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 5]), requires_grad=require_grad)
            self.k5 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 6]), requires_grad=require_grad)
            self.k6 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 7]), requires_grad=require_grad)
            self.s1 = None
            self.s2 = None
            self.s3 = None
            self.s4 = None
            self.taux = None
            self.tauy = None
        elif self.nbr_params == 12:
            self.k1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 0]), requires_grad=require_grad)
            self.k2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 1]), requires_grad=require_grad)
            self.p1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 2]), requires_grad=require_grad)
            self.p2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 3]), requires_grad=require_grad)
            self.k3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 4]), requires_grad=require_grad)
            self.k4 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 5]), requires_grad=require_grad)
            self.k5 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 6]), requires_grad=require_grad)
            self.k6 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 7]), requires_grad=require_grad)
            self.s1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 8]), requires_grad=require_grad)
            self.s2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 9]), requires_grad=require_grad)
            self.s3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 10]), requires_grad=require_grad)
            self.s4 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 11]), requires_grad=require_grad)
            self.taux = None
            self.tauy = None
        elif self.nbr_params == 14:
            self.k1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 0]), requires_grad=require_grad)
            self.k2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 1]), requires_grad=require_grad)
            self.p1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 2]), requires_grad=require_grad)
            self.p2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 3]), requires_grad=require_grad)
            self.k3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 4]), requires_grad=require_grad)
            self.k4 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 5]), requires_grad=require_grad)
            self.k5 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 6]), requires_grad=require_grad)
            self.k6 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 7]), requires_grad=require_grad)
            self.s1 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 8]), requires_grad=require_grad)
            self.s2 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 9]), requires_grad=require_grad)
            self.s3 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 10]), requires_grad=require_grad)
            self.s4 = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 11]), requires_grad=require_grad)
            self.taux = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 12]), requires_grad=require_grad)
            self.tauy = nn.Parameter(torch.mul(torch.ones(size=(num_cams,), dtype=torch.float32), init_dist[:, 13]), requires_grad=require_grad)

    def forward(self):
        if self.nbr_params == 4:
            self.distortion[:, 0] = self.k1
            self.distortion[:, 1] = self.k2
            self.distortion[:, 2] = self.p1
            self.distortion[:, 3] = self.p2
            return self.distortion
        elif self.nbr_params == 5:
            self.distortion[:, 0] = self.k1
            self.distortion[:, 1] = self.k2
            self.distortion[:, 2] = self.p1
            self.distortion[:, 3] = self.p2
            self.distortion[:, 4] = self.k3
            return self.distortion
        elif self.nbr_params == 8:
            self.distortion[:, 0] = self.k1
            self.distortion[:, 1] = self.k2
            self.distortion[:, 2] = self.p1
            self.distortion[:, 3] = self.p2
            self.distortion[:, 4] = self.k3
            self.distortion[:, 5] = self.k4
            self.distortion[:, 6] = self.k5
            self.distortion[:, 7] = self.k6
            return self.distortion
        elif self.nbr_params == 12:
            self.distortion[:, 0] = self.k1
            self.distortion[:, 1] = self.k2
            self.distortion[:, 2] = self.p1
            self.distortion[:, 3] = self.p2
            self.distortion[:, 4] = self.k3
            self.distortion[:, 5] = self.k4
            self.distortion[:, 6] = self.k5
            self.distortion[:, 7] = self.k6
            self.distortion[:, 8] = self.s1
            self.distortion[:, 9] = self.s2
            self.distortion[:, 10] = self.s3
            self.distortion[:, 11] = self.s4
            return self.distortion
        elif self.nbr_params == 14:
            self.distortion[:, 0] = self.k1
            self.distortion[:, 1] = self.k2
            self.distortion[:, 2] = self.p1
            self.distortion[:, 3] = self.p2
            self.distortion[:, 4] = self.k3
            self.distortion[:, 5] = self.k4
            self.distortion[:, 6] = self.k5
            self.distortion[:, 7] = self.k6
            self.distortion[:, 8] = self.s1
            self.distortion[:, 9] = self.s2
            self.distortion[:, 10] = self.s3
            self.distortion[:, 11] = self.s4
            self.distortion[:, 12] = self.taux
            self.distortion[:, 13] = self.tauy
            return self.distortion
    
    def get_distortionCam(self, frame_num):
        if self.nbr_params == 4:
            return self.distortion[frame_num, 0:4]
        elif self.nbr_params == 5:
            return self.distortion[frame_num, 0:5]
        elif self.nbr_params == 8:
            return self.distortion[frame_num, 0:8]
        elif self.nbr_params == 12:
            return self.distortion[frame_num, 0:12]
        elif self.nbr_params == 14:
            return self.distortion[frame_num, 0:14]
        
    def get_distortionAll(self):
        if self.nbr_params == 4:
            return self.distortion[:, 0:4]
        elif self.nbr_params == 5:
            return self.distortion[:, 0:5]
        elif self.nbr_params == 8:
            return self.distortion[:, 0:8]
        elif self.nbr_params == 12:
            return self.distortion[:, 0:12]
        elif self.nbr_params == 14:
            return self.distortion[:, 0:14]

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, width, height, mode='norm', init_c2w=None, retDegree=False, identical_t=False):
        """
        Parameters learn the difference between initial pose and target pose
        In a model call the forward function includes the initial pose [N, 4, 4] with c2w

        :param num_cams: number of cameras/frames
        :param learn_R:  True/False --> in rad
        :param learn_t:  True/False --> in pixel/gps
        :param mode:     '1', 'norm' (with normaize factor)
        :param init_c2w: (N, 4, 4) torch tensor
        :param normalize_factor:  for translation, 1000
        :param retDegree: return degree, False
        :param identical_t: if True, all cameras share the same translation
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.normalize_x = width
        self.normalize_y = height
        self.normalize = torch.tensor([self.normalize_x, self.normalize_y, 1], dtype=torch.float32)
        self.mode = mode
        self.retDegree = retDegree
        self.identical_t = identical_t

        init_c2w = init_c2w.clone().detach().cpu()
        if init_c2w is not None:
            rotM = init_c2w[:, :3, :3]  # (N, 3, 3)
            r_angle = torch.tensor(get_rotation_angle(rotM, rotSeq='xyz', degree=False), dtype=torch.float32) # (N, 3)

            if mode == '1':
                self.r = nn.Parameter(torch.mul(torch.ones(size=(num_cams, 3)), r_angle), requires_grad=learn_R)  # (N, 3), euler angles in rad
                if self.identical_t:
                    t_val = init_c2w[0, :3, 3].unsqueeze(0)  # (1, 3)
                    self.t = nn.Parameter(t_val, requires_grad=learn_t)
                else:
                    self.t = nn.Parameter(torch.mul(torch.ones(size=(num_cams, 3)) , init_c2w[:, :3, 3]), requires_grad=learn_t)  # (N, 3), translation absolut in pixel
            elif mode == 'norm':
                self.r = nn.Parameter(torch.mul(torch.ones(size=(num_cams, 3)), r_angle), requires_grad=learn_R)
                # normalize translation
                t = torch.mul(torch.ones(size=(num_cams, 3)), init_c2w[:, :3, 3])
                normalize = 1 / self.normalize
                normalize = normalize.reshape((1, 3))
                normalize = normalize.repeat(num_cams, 1)     # (N, 3)
                t = torch.mul(t, normalize)
                if self.identical_t:
                    t_val = t[0].unsqueeze(0)  # (1, 3)
                    self.t = nn.Parameter(t_val, requires_grad=learn_t)
                else:
                    self.t = nn.Parameter(t, requires_grad=learn_t)
        else:
            self.r = nn.Parameter(torch.rand(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3), euler angles in rad
            if self.identical_t:
                self.t = nn.Parameter(torch.rand(size=(1, 3), dtype=torch.float32), requires_grad=learn_t)  # (1, 3)
            else:
                self.t = nn.Parameter(torch.rand(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3), translation absolut in pixel
            print('WARNING: Mode is not 1 or norm - random init')

        self.c2w_all = init_c2w.clone()  # (N, 4, 4)

    def forward(self):
        # returns c2w including the initial pose
        r = self.r
        if self.retDegree:
            r = r * torch.tensor(180) / torch.pi
        rotM = rotationBatch_torch(r, 'xyz', self.c2w_all, degrees=self.retDegree)  # (N, 3, 3)
        self.c2w_all[:, :3, :3] = rotM
        
        # translation
        if self.mode == '1':
            if self.identical_t:
                self.c2w_all[:, :3, 3] = self.t.expand(self.num_cams, 3)
            else:
                self.c2w_all[:, :3, 3] = self.t
        elif self.mode == 'norm':
            normalize = self.normalize
            normalize = normalize.reshape((1, 3))
            normalize = normalize.repeat(self.num_cams, 1)
            normalize = normalize.to(self.t.device)
            if self.identical_t:
                self.c2w_all[:, :3, 3] = (self.t * normalize).expand(self.num_cams, 3)
            else:
                self.c2w_all[:, :3, 3] = self.t * normalize
        else:
            if self.identical_t:
                self.c2w_all[:, :3, 3] = self.t.expand(self.num_cams, 3)
            else:
                self.c2w_all[:, :3, 3] = self.t
            print('WARNING: Mode need to be "1" or "norm"')

        return self.c2w_all
    
    def get_t(self):
        if self.mode == '1':
            if self.identical_t:
                return self.t.expand(self.num_cams, 3)
            else:
                return self.t
        elif self.mode == 'norm':
            normalize = self.normalize
            normalize = normalize.reshape((1, 3))
            normalize = normalize.repeat(self.num_cams, 1)
            normalize = normalize.to(self.t.device)
            if self.identical_t:
                return (self.t * normalize).expand(self.num_cams, 3)
            else:
                return self.t * normalize
        else:
            print('WARNING: Mode need to be "1" or "norm"')
            if self.identical_t:
                return self.t.expand(self.num_cams, 3)
            else:
                return self.t
    
    def get_r(self):
        if self.retDegree:
            return self.r * torch.tensor(180) / torch.pi
        else:
            return self.r    # rotation angle (N, 3)
    
    def get_c2w(self, frame_num):
        frame_num = int(frame_num)
        c2w = torch.zeros(size=(4, 4), dtype=torch.float32)
        c2w[3, 3] = 1.0

        r = self.r[frame_num].detach().cpu()
        if self.retDegree:
            r = r * torch.tensor(180) / torch.pi
        rotM = rotation(r, 'xyz', degrees=self.retDegree)  # (3, 3)
        c2w[:3, :3] = torch.tensor(rotM, dtype=torch.float32)

        # translation
        if self.mode == '1':
            if self.identical_t:
                c2w[:3, 3] = self.t[0].detach().cpu()
            else:
                c2w[:3, 3] = self.t[frame_num].detach().cpu()
        elif self.mode == 'norm':
            normalize = self.normalize
            normalize = normalize.reshape((1, 3))
            normalize = normalize.repeat(1, 1)
            normalize = normalize.to(self.t.device)
            if self.identical_t:
                c2w[:3, 3] = (self.t[0] * normalize[0]).detach().cpu()
            else:
                c2w[:3, 3] = (self.t[frame_num] * normalize[0]).detach().cpu()
        else:
            if self.identical_t:
                c2w[:3, 3] = self.t[0].detach().cpu()
            else:
                c2w[:3, 3] = self.t[frame_num].detach().cpu()
            print('WARNING: Mode need to be "1" or "norm"')

        return c2w
    
def rotationBatch_torch(rotation_angles, rotation_axes, c2w, degrees=True):
    '''
    Create rotation matrices for multiple axes across batches
    Input:
    - rotation_angles: 2D tensor of shape (N, number_of_angles) where N is batch size
    - rotation_axes: list of rotation axes ('x', 'y', or 'z') for each angle
    - degrees: True if the angles are in degrees, False if in radians
    '''
    # Check if number of columns in angles matches number of axes
    if rotation_angles.shape[1] != len(rotation_axes):
        raise ValueError("The number of angles in each batch must match the number of axes.")

    # Convert angles to radians if they are in degrees
    if degrees:
        angles = torch.deg2rad(rotation_angles)
    else:
        angles = rotation_angles

    rot = c2w[:, :3, :3].clone()

    for b, batch_angles in enumerate(angles):
        # Start with identity matrix for each batch
        batch_rotation = c2w[b, :3, :3].clone()
        batch_rotation[:,:] = 0
        batch_rotation[0,0] = 1
        batch_rotation[1,1] = 1
        batch_rotation[2,2] = 1

        rotEye = batch_rotation.clone()
        
        for angle, axis in zip(batch_angles, rotation_axes):
            rot_matrix = rotEye.clone()
            if axis == 'x':
                rot_matrix[1, 1] = torch.cos(angle)
                rot_matrix[1, 2] = -torch.sin(angle)
                rot_matrix[2, 1] = torch.sin(angle)
                rot_matrix[2, 2] = torch.cos(angle)
            elif axis == 'y':
                rot_matrix[0, 0] = torch.cos(angle)
                rot_matrix[0, 2] = torch.sin(angle)
                rot_matrix[2, 0] = -torch.sin(angle)
                rot_matrix[2, 2] = torch.cos(angle)
            elif axis == 'z':
                rot_matrix[0, 0] = torch.cos(angle)
                rot_matrix[0, 1] = -torch.sin(angle)
                rot_matrix[1, 0] = torch.sin(angle)
                rot_matrix[1, 1] = torch.cos(angle)
            else:
                raise ValueError(f"Invalid axis '{axis}'. Axis must be 'x', 'y', or 'z'.")
            
            # Accumulate rotations for this batch
            batch_rotation = torch.matmul(batch_rotation, rot_matrix)
        
        # swap and change sign to match opencv
        tmp_rotation = batch_rotation.clone()
        batch_rotation[0, 1] = -tmp_rotation[1, 0]
        batch_rotation[0, 2] = tmp_rotation[2, 0]
        batch_rotation[1, 0] = -tmp_rotation[0, 1]
        batch_rotation[1, 2] = -tmp_rotation[2, 1]
        batch_rotation[2, 0] = -tmp_rotation[0, 2]
        batch_rotation[2, 1] = -tmp_rotation[1, 2]
        rot[b, :, :] = batch_rotation

    # Combine all batch rotations into a 3D tensor
    return rot

def project_points_my(points3d_xyz, rMat, tvec, intrinsics, dist_coeffs=None):
    # Transform points from world to camera coordinates
    points_cam = torch.matmul(points3d_xyz, rMat.T) + tvec

    # Normalize points by dividing by z (convert to homogeneous coordinates)
    points_cam_hom = points_cam / points_cam[:, 2:3]

    # Apply camera intrinsics
    points_proj = torch.matmul(points_cam_hom, intrinsics.T)

    # Distortion correction
    points_proj = distort_my(points_proj, dist_coeffs)

    # Return projected points (in image coordinates)
    return points_proj

def distort_my(points2d, dist_coeffs):
    if dist_coeffs is not None:
        dist_coeffs_loc = dist_coeffs.clone()
        points2d_loc = points2d.clone()
        x = points2d_loc[:, 0]
        y = points2d_loc[:, 1]
        
        # Radial distortion coefficients
        k1 = dist_coeffs_loc[0]
        k2 = dist_coeffs_loc[1]
        if dist_coeffs_loc.size(0) > 4:
            k3 = dist_coeffs_loc[4]
        else:
            k3 = torch.tensor(0.0)
        if dist_coeffs_loc.size(0) > 5:
            k4 = dist_coeffs_loc[5]
            k5 = dist_coeffs_loc[6]
            k6 = dist_coeffs_loc[7]
        else:
            k4 = torch.tensor(0.0)
            k5 = torch.tensor(0.0)
            k6 = torch.tensor(0.0)
        if dist_coeffs_loc.size(0) > 11:
            k7 = dist_coeffs_loc[12]
            k8 = dist_coeffs_loc[13]
        else:
            k7 = torch.tensor(0.0)
            k8 = torch.tensor(0.0)
        
        # Tangential distortion coefficients
        p1 = dist_coeffs_loc[2]
        p2 = dist_coeffs_loc[3]
        
        # Thin prism distortion coefficients
        if dist_coeffs_loc.size(0) > 13:
            s1 = dist_coeffs_loc[8]
            s2 = dist_coeffs_loc[9]
            s3 = dist_coeffs_loc[10]
            s4 = dist_coeffs_loc[11]
        else:
            s1 = torch.tensor(0.0)
            s2 = torch.tensor(0.0)
            s3 = torch.tensor(0.0)
            s4 = torch.tensor(0.0)
        
        # Tilt coefficients
        if dist_coeffs_loc.size(0) > 15:
            tx = dist_coeffs_loc[14]
            ty = dist_coeffs_loc[15]
        else:
            tx = torch.tensor(0.0)
            ty = torch.tensor(0.0)
        
        # Compute r^2
        r2_tmp = x**2 + y**2

        # Radial distortion
        r2 = torch.clamp(r2_tmp, min=1e-6, max=1e3)        # numerical stability
        xd = x * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4 + k5 * r2**5 + k6 * r2**6 + k7 * r2**7 + k8 * r2**8)
        yd = y * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4 + k5 * r2**5 + k6 * r2**6 + k7 * r2**7 + k8 * r2**8)

        # Tangential distortion
        xd = xd + (2 * p1 * x * y + p2 * (r2 + 2 * x**2))
        yd = yd + (p1 * (r2 + 2 * y**2) + 2 * p2 * x * y)

        # Thin prism distortion
        xd = xd + (s1 * r2 + s2 * r2**2)
        yd = yd + (s3 * r2 + s4 * r2**2)

        # Tilt model (rarely used)
        xd = xd + (tx * r2 + ty * r2**2)
        yd = yd + (tx * r2 + ty * r2**2)

        # points2d_loc[:, 0] = xd     # Fails because of this inlined operation, point2d_loc._version = 2 after both operations
        # points2d_loc[:, 1] = yd

        points2d_loc = torch.stack([xd, yd], dim=1) # solves issue :-)

    return points2d_loc


def render_3d_w2c_opencv_torch(points3d, w2c, intrinsics, dist_coeffs = None, info=False):
    # dist_coeffs (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 element
    w2c = w2c[:3, :]
    points3d_xyz = points3d.T  # 3xN
    points3d_xyz = torch.vstack((points3d_xyz, torch.ones((1, points3d_xyz.shape[1]))))   # 4xN
    # step2 change to camera coordinate
    points3d_xyz = w2c @ points3d_xyz  # 3xN
    points3d_xyz = points3d_xyz[:3, :] # 3xN

    # select based on depth > 0 and < depth_max
    points3d_xyz = points3d_xyz.T  # Nx3

    # prevent empty points error
    if len(points3d_xyz) == 0:
        print("No points in the camera view")
        return None
    
    # rMat = torch.eye(3)
    # tvec = torch.zeros(3)

    # Project 3D points to 2D
    #points2d = project_points_my(points3d_xyz, rMat, tvec, intrinsics, dist_coeffs)      # does not solve the problem, here it has to be numpy array no gradient anymore!!!
    points2d = project_points(points3d_xyz, intrinsics)
    points2d = distort_my(points2d, dist_coeffs)    # fails in backward pass, because of inlined operation

    if info:
        print("points in camera view from", len(points2d))    
        
    return points2d    # without filtering

def prepare_vertices_ndc(vertices, faces, intrinsics, camera_transform, img_size):
    """
    Transforms vertices to clip space and then NDC, another method than projection with intrinsic matrix, used to scale with custom 3*focal_x

    Args:
        vertices: 3D vertices in world space.
        faces: Face indices.
        intrinsics: The camera's projection matrix (4x4).
        camera_transform: The camera's transformation matrix (3x4).
        img_size: Size of the image (height, width, C).

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
    cx = img_size[1] / 2 + intrinsics.x0
    cy = img_size[0] / 2 + intrinsics.y0
    # 3:2 aspect ratio for focal length
    projection_matrix = torch.tensor([[3*intrinsics.focal_x/img_size[1], 0, 2*cx/img_size[1]-1, 0],
                                        [0, 2*intrinsics.focal_y/img_size[0], 2*cy/img_size[0]-1, 0],
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
    # face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

    return face_vertices_camera, face_vertices_ndc[:, :, :, :2]

def render_depth_plus_pytorch3d(face_vertices_image, face_vertices_z, view_target_h, view_target_w, device, render_scale=1.0):
    """
    Rasterize a depth map, face index map, and UV map using PyTorch3D for speed.

    Args:
    - face_vertices_image: Tensor (1, num_faces, 3, 2) with 2D vertices (NDC XY) of each face. Assumes +Y is up.
    - face_vertices_z: Tensor (1, num_faces, 3) with view-space z-values (positive depth) of each face vertex.
    - view_target_h: Height of the target image
    - view_target_w: Width of the target image
    - device: Device to use
    - render_scale: Scale factor for rendering resolution (default 1.0).

    Returns:
    - depth_image: Tensor (1, view_target_h, view_target_w, 1)
    - face_idx_buffer: Tensor (1, view_target_h, view_target_w)
    """
    batch_size = 1 # This function processes one mesh at a time
    num_faces = face_vertices_image.shape[1]
    render_h = int(view_target_h * render_scale)
    render_w = int(view_target_w * render_scale)

    face_vertices_image = face_vertices_image.to(device)
    face_vertices_z = face_vertices_z.to(device)

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

    if render_scale != 1.0:        # Handle potential upsampling if render_scale was not 1.0
        depth_image = F.interpolate(depth_image.permute(0, 3, 1, 2), size=(view_target_h, view_target_w), mode='nearest').permute(0, 2, 3, 1)
        face_idx_buffer = F.interpolate(face_idx_buffer.unsqueeze(1).float(), size=(view_target_h, view_target_w), mode='nearest').long().squeeze(1)

    return depth_image, face_idx_buffer

def get_differentiable_visibility_scores(c2w, vertices, faces, visible_threshold=0.2):
    """
    Calculates a differentiable visibility score for each vertex.

    Instead of making a hard decision to cull a vertex, this function assigns
    a continuous score between 0.0 and 1.0, indicating how "visible" a vertex
    is based on whether it's in front of the camera. This is achieved by
    passing the dot product through a sigmoid function. This soft score is
    differentiable and suitable for use in learning pipelines.

    Args:
        c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).
        vertices (torch.Tensor): 3D vertices of shape (N, 3).
        faces (torch.Tensor): Face indices of shape (F, 3).
        visible_threshold (float): Threshold for visibility score. Vertices with scores above this threshold are considered visible.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) with visible vertices
        torch.Tensor: A tensor of shape (F, 3) with visible faces
    """
    camera_rotation = c2w[:3, :3].to(vertices.device)
    camera_position = c2w[:3, 3].to(vertices.device)

    camera_forward_vector = -camera_rotation[:, 2]
    vectors_to_vertices = vertices - camera_position
    dot_products = torch.matmul(vectors_to_vertices, camera_forward_vector)
    vertex_score_per_face = dot_products[faces]
    face_scores = vertex_score_per_face.mean(dim=1)

    visible_vertices = dot_products > visible_threshold
    visible_faces = face_scores > visible_threshold
    filtered_vertices = vertices[visible_vertices]
    filtered_faces = faces[visible_faces]

    return filtered_vertices, filtered_faces