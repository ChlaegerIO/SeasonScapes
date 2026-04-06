import torch
import tqdm
import torch.optim as optim
import time
import logging

from utils.camFormat import *
from dataVisu_engine.ThreeD_helpers import *
from dataPose_optimization.model import *
from torch.utils.tensorboard import SummaryWriter

from dataVisu_engine.ThreeD_helpers import *
from utils.config import *

######################## CONFIG ########################
HASH = "-8512"

POINTMATCH_PATH = "data/pointMatch/pointMatch_v4.json"             # Path to the point matching file
TRANSFORM_MATRIX_PATH = "data/transformation_matrices/Rshot/transformation_final_v3_flat.json" # Path to the transformation matrix file

WHICH_TASK = 'matchImg2Pc_1'   # matchImg2Pc_1, matchImag2Pc
CAM_ID = '85_20-WENGBELL_1'     # exact cam_id, i.e. key in transformMatrix
CAM_COMPLETE = True     # True: All cameras at one location, False: not all cameras at one location (no deltaT loss between cameras)

POINT_SIZE = 8

# training parameters
NBR_EPOCHS = 251
PRINT_EVERY = 5
FAC_PRINT_IMG = 20
BACKUP_EVERY = 250

LR_POSE = 0.01
LR_FOCAL = 0.02
LR_DISTORT = 1e-10

# init model and optimizer
LEARN_F = True
LEARN_R = True
LEARN_T = False
LEARN_DISTORT = False
FX_ONLY = False

# loss factors
L_MATCH = 1
L_DELTA_T = 10000
L_INIT_T = 100
L_INIT_DIST = 100000
L_RY = 2

DIST_PARAMS = 4

######################## PROGRAM ########################

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
print(f"Device: {device}")

# initializations
home_path = getHomePath()
pointMatch_path = Path(home_path, POINTMATCH_PATH)
config_fPath = Path(home_path, 'configs', 'EarthEngine', f"Scale60_{HASH}.json")
data_eng_path = Path(home_path, 'data_engine')
scene_path = Path(home_path, Path(TRANSFORM_MATRIX_PATH).parent)
transform_path = Path(home_path, TRANSFORM_MATRIX_PATH)

# configuration
transformMatrix = openTFMatrix(TRANSFORM_MATRIX_PATH)
config = Config()
config.loadGEE(config_fPath)

# create logger
log_path = Path(data_eng_path, 'Logs', 'current')
pyLog_fName = Path(log_path, f'log_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log')
# create folder for checkpoints, logs
if not Path(data_eng_path, 'checkpoints').exists():
    Path(data_eng_path, 'checkpoints').mkdir(parents=True)
if not Path(log_path).exists():
    Path(log_path).mkdir(parents=True)

logger_py = logging.getLogger(__name__)
logging.basicConfig(filename=pyLog_fName.as_posix(),level=logging.INFO)

# loss function
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()


# training function for one image to learn camera pose, focal length and distortion parameters to fit the point cloud
# NOTE: translation in pixel, rotation in rad
def fitImage2Pc(cam_id):
    # init information e.g. cam
    cam = {}
    for data_id in transformMatrix.keys():
        if cam_id in data_id:
            cam[data_id] = transformMatrix[data_id]
            break
    frame_num = len(cam)

    fName = f'cam_{cam_id}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    tBoard_fName = Path(log_path, fName)
    logger = SummaryWriter(tBoard_fName.as_posix())

    # load matching frames
    with open(pointMatch_path.as_posix()) as f:
        matchingFrames = json.load(f)

    # init camera parameters and points
    c2w_init = torch.zeros([frame_num, 4, 4], dtype=torch.float32)
    focal_init = torch.zeros([frame_num, 2], dtype=torch.float32)
    dist_init = torch.zeros([frame_num, DIST_PARAMS], dtype=torch.float32)
    for i, data_id in enumerate(cam.keys()):
        focal_init[i, 0] = cam[data_id]['fx']
        focal_init[i, 1] = cam[data_id]['fy']
        dist_init[i, 0] = cam[data_id]['k1']
        dist_init[i, 1] = cam[data_id]['k2']
        dist_init[i, 2] = cam[data_id]['p1']
        dist_init[i, 3] = cam[data_id]['p2']
        if DIST_PARAMS >= 5:
            dist_init[i, 4] = cam[data_id]['k3']
        if DIST_PARAMS >= 6:
            dist_init[i, 5] = cam[data_id]['k4']
            dist_init[i, 6] = cam[data_id]['k5']
            dist_init[i, 7] = cam[data_id]['k6']
        # cx_init[i, 0] = cam[data_id]['cx']
        # cy_init[i, 1] = cam[data_id]['cy']
        c2w = torch.tensor(cam[data_id]['transform_matrix'])
        t_gps = c2w[:3, 3].tolist()
        t_pix = geoCoord2Open3Dpx(config, t_gps)
        c2w[:3, 3] = torch.tensor(t_pix)
        c2w_init[i] = c2w
        width = cam[data_id]['wid']
        height = cam[data_id]['hei']
        cx = cam[data_id]['cx']
        cy = cam[data_id]['cy']

    # send all devices to GPU
    c2w_init = c2w_init.to(device)
    focal_init = focal_init.to(device)
    dist_init = dist_init.to(device)

    print(f"\nTraining for camera: {cam_id} with {frame_num} views")
    print(f"Initial focal length: {focal_init}")
    print(f"Initial distortion parameters: {dist_init}")
    print(f"Initial c2w: {c2w_init}")

    # focal net
    intrinsic_net = LearnFocal(frame_num, LEARN_F, width, height, cx, cy, mode='norm', init_focal=focal_init.tolist()).to(device=device)
    optimizer_intrinsic = optim.Adam(intrinsic_net.parameters(), lr=LR_FOCAL)
    scheduler_intrinsic = torch.optim.lr_scheduler.MultiStepLR(optimizer_intrinsic, milestones=[NBR_EPOCHS/4, NBR_EPOCHS/2, NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)
    # pose net
    pose_net = LearnPose(frame_num, LEARN_R, LEARN_T, width, height, mode='norm', init_c2w=c2w_init, retDegree=True).to(device=device)
    optimizer_pose = optim.Adam(pose_net.parameters(), lr=LR_POSE)
    scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=[NBR_EPOCHS/4, NBR_EPOCHS/2, NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)
    # distortion parameters
    distort_net = Learn_Distortion(frame_num, LEARN_DISTORT, nbr_params=DIST_PARAMS, init_dist=dist_init).to(device=device)
    optimizer_distort = optim.Adam(distort_net.parameters(), lr=LR_DISTORT)
    scheduler_distort = torch.optim.lr_scheduler.MultiStepLR(optimizer_distort, milestones=[NBR_EPOCHS/4, NBR_EPOCHS/2, NBR_EPOCHS/4*3], gamma=0.1, last_epoch=-1)

    # set to training mode
    intrinsic_net.train()
    pose_net.train()
    distort_net.train()

    # Print model
    logger_py.info(f"\nTraining for camera: {cam_id} with {frame_num} views")
    logger_py.info(f"Learns focal length: {LEARN_F}\nLearns rotation: {LEARN_R}\nLearns translation: {LEARN_T}\nLearns distortion: {LEARN_DISTORT}")
    logger_py.info(f"loss factors: L_MATCH={L_MATCH}, L_DELTA_T={L_DELTA_T}, L_INIT_T={L_INIT_T}, L_INIT_DIST={L_INIT_DIST}")
    logger_py.info(f"Learning rates: LR_POSE={LR_POSE}, LR_FOCAL={LR_FOCAL}, LR_DISTORT={LR_DISTORT}")
    logger_py.info(f"Distortion parameters used: {DIST_PARAMS}")
    logger_py.info(f"Initial focal length: {focal_init}")
    logger_py.info(f"Initial distortion parameters: {dist_init}")
    logger_py.info(f"Initial c2w: {c2w_init}")
    t0b = time.time()

    # load ground truth selected 2D points to device
    img_gt_list = []
    img_gt_backg = []
    for data_id in cam.keys():
        try:
            fPath = Path(home_path, cam[data_id]['file_path'])
        except:
            logger_py.warning(f"No file path for, use default path data/2024-10-09/12-00-00/imagesPlane/{data_id}.jpg")
            fPath = Path(home_path, 'data', '2024-10-09', '12-00-00', 'imagesPlane', f'{data_id}.jpg')
        
        # check if data_id in pointMatch
        if data_id not in matchingFrames.keys():
            logger_py.warning(f"No camera {data_id}, skipping")
            continue

        img = loadRGBImg(fPath)
        img = torch.tensor(img, dtype=torch.float32, requires_grad=False)
        # white img
        img_point = torch.zeros_like(img)
        # add points
        for p in range(len(matchingFrames[data_id]['match2D'])):
            img_point[matchingFrames[data_id]['match2D'][p][1]-POINT_SIZE//2:matchingFrames[data_id]['match2D'][p][1]+POINT_SIZE//2, 
                matchingFrames[data_id]['match2D'][p][0]-POINT_SIZE//2:matchingFrames[data_id]['match2D'][p][0]+POINT_SIZE//2] = torch.tensor([0, 255, 0], dtype=torch.float32)  # target = green with black middle point
            img_point[matchingFrames[data_id]['match2D'][p][1], matchingFrames[data_id]['match2D'][p][0]] = torch.tensor([255, 255, 255], dtype=torch.float32)
        img_gt_list.append(img_point)
        img_gt_backg.append(img)

    pbar = tqdm(range(NBR_EPOCHS))
    for epoch in pbar:
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
        matching_loss = torch.tensor(0.0, dtype=torch.float32)
        img_infer_list = []
        nbr_img_per_webcam = 0
        for i, data_id in enumerate(cam.keys()):
            # check if data_id in pointMatch
            if data_id not in matchingFrames.keys():
                continue

            intrinsic = intrinsic_lNet[i]
            distortion = distort_lNet[i]

            c2w = c2w_pNet[i]
            # openGL to openCV
            c2w = opengl_to_opencv(c2w)
            w2c = torch.linalg.inv(c2w)

            points3d = torch.tensor(matchingFrames[data_id]['match3D'], dtype=torch.float32, requires_grad=True)
            # project points to 2D, torch version
            inference = render_3d_w2c_opencv_torch(points3d, w2c, intrinsic, distortion)
            target = torch.tensor(matchingFrames[data_id]['match2D'], dtype=torch.float32, requires_grad=True)

            # visualize inference points
            if PRINT_EVERY > 0 and (epoch % (PRINT_EVERY*FAC_PRINT_IMG)) == 0 or BACKUP_EVERY > 0 and (epoch % BACKUP_EVERY) == 0:  # for visualization
                img_infer = torch.zeros_like(img_gt_list[0])
                for p in range(inference.shape[0]):
                    if inference[p, 0] < 0 or inference[p, 0] >= width or inference[p, 1] < 0 or inference[p, 1] >= height:
                        continue
                    img_infer[int(inference[p,1])-POINT_SIZE//2:int(inference[p,1])+POINT_SIZE//2, 
                              int(inference[p,0])-POINT_SIZE//2:int(inference[p,0])+POINT_SIZE//2] = torch.tensor([255, 0, 0], dtype=torch.float32)  # infered = red with black middle point
                    img_infer[int(inference[p,1]), int(inference[p,0])] = torch.tensor([255, 255, 255], dtype=torch.float32)
                img_infer_list.append(img_infer)

            num_points = len(target)
            if num_points != 0 and len(target) == len(inference):
                # do not normalize for pixel loss
                target = target
                inference = inference
                nbr_img_per_webcam += 1
                #matching_loss += mse_loss(inference, target)
                matching_loss += l1_loss(inference, target)
            else:
                logger_py.warning(f"Mask sum is zero for view {i} or target len {len(target)} and inference len {len(inference)} different")        

        # loss calculation
        matching_loss = matching_loss / nbr_img_per_webcam

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
        loss_ry = torch.norm(pose_net.get_r().detach().cpu()[:, 1], p=1)

        # total loss with factors
        if CAM_COMPLETE:
            loss = L_MATCH*matching_loss + L_DELTA_T*loss_deltaT + L_INIT_T*loss_deltaTInit + L_INIT_DIST*loss_distInit + L_RY*loss_ry
        else:
            loss = L_MATCH*matching_loss + L_INIT_T*loss_deltaTInit + L_INIT_DIST*loss_distInit + L_RY*loss_ry

        # manually set gradients
        c2w_pNet.grad = torch.full(c2w_pNet.shape, -loss.item())

        # loss logging
        loss_dict['matching_loss'] = matching_loss
        loss_dict['deltaT_loss'] = loss_deltaT
        loss_dict['deltaT_init_loss'] = loss_deltaTInit
        loss_dict['distortion_init_loss'] = loss_distInit
        loss_dict['ry_loss'] = loss_ry
        loss_dict['loss'] = loss

        # backward pass
        loss.backward(retain_graph=True)
        
        # update parameters
        optimizer_intrinsic.step()
        optimizer_pose.step()
        optimizer_distort.step()
        
        # print loss
        if PRINT_EVERY > 0 and (epoch % PRINT_EVERY) == 0 or epoch == 0:
            tqdm.write(f'Epoch {epoch}, loss={loss}, l_match={matching_loss}, l_dist={loss_distInit}, l_ry={loss_ry}, l_t={loss_deltaT}, l_tInit={loss_deltaTInit}, time={time.time()-t0b}')
            logger_py.info(f'Epoch {epoch}, loss={loss}, time={time.time()-t0b}')
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
            if DIST_PARAMS >= 5:
                logger.add_histogram('params/k3_0', distort_net.get_distortionAll()[:, 4].detach().cpu(), epoch)
            if DIST_PARAMS >= 6:
                logger.add_histogram('params/k4_0', distort_net.get_distortionAll()[:, 5].detach().cpu(), epoch)
                logger.add_histogram('params/k5_0', distort_net.get_distortionAll()[:, 6].detach().cpu(), epoch)
                logger.add_histogram('params/k6_0', distort_net.get_distortionAll()[:, 7].detach().cpu(), epoch)

            # only visualize every other epoch
            if epoch % (PRINT_EVERY*FAC_PRINT_IMG) == 0:
                for idx in range(frame_num):
                    ovlImg = cv2.addWeighted(np.array(img_gt_backg[idx]), 0.8, np.array(img_gt_list[idx]), 0.6, 0)
                    ovlImg = cv2.addWeighted(ovlImg, 0.8, np.array(img_infer_list[idx]), 0.6, 0)
                    ovlImg = np.transpose(ovlImg, (2, 0, 1))
                    ovlImg = ovlImg / 255.0
                    logger.add_image(f'image {idx}', ovlImg, epoch)

        for l, num in loss_dict.items():
            if l == 'loss':
                logger.add_scalar(f'train_{L_MATCH}*l_match + {L_DELTA_T}*l_deltaT + {L_INIT_T}*l_InitT + {L_INIT_DIST}*l_InitDist + {L_RY}*l_ry/{l}', num.detach().cpu(), epoch)
            else:
                logger.add_scalar(f'train/{l}', num.detach().cpu(), epoch)

        # update learning rate
        scheduler_intrinsic.step()
        scheduler_pose.step()
        scheduler_distort.step()

        # save checkpoints and write transformation_matris.json
        if BACKUP_EVERY > 0 and (epoch % BACKUP_EVERY) == 0:
            # save point matched images
            img_path = Path(scene_path, 'poseOptim_img')
            if not img_path.exists():
                img_path.mkdir(parents=True)

            for idx in range(frame_num):
                ovlImg = cv2.addWeighted(np.array(img_gt_backg[idx]), 0.8, np.array(img_gt_list[idx]), 0.6, 0)
                ovlImg = cv2.addWeighted(ovlImg, 0.8, np.array(img_infer_list[idx]), 0.6, 0)
                ovlImg = np.transpose(ovlImg, (2, 0, 1))
                ovlImg = ovlImg / 255.0

                # save image
                img_fName = Path(img_path, f'img_{cam_id}_{epoch}_{idx}.png')
                cv2.imwrite(img_fName.as_posix(), ovlImg)

            # append write transformation matrix, open start file
            transMatrix_path = Path(data_eng_path, 'checkpoints', f'transformation_{epoch}.json')
            if transMatrix_path.exists():
                with open(transMatrix_path.as_posix()) as f:
                    transform_ckpt = json.load(f)
            else:
                with open(transform_path.as_posix()) as f:
                    transform_ckpt = json.load(f)

            for idx, data_id in enumerate(cam.keys()):
                c2w = pose_net.get_c2w(idx).cpu().detach().numpy()
                trans_gps = Open3Dpx2geoCoord(config, c2w[:3, 3])
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
                if DIST_PARAMS >= 5:
                    transform_ckpt[data_id]['k3'] = distort_lNet[idx, 4].detach().cpu().item()
                if DIST_PARAMS >= 6:
                    transform_ckpt[data_id]['k4'] = distort_lNet[idx, 5].detach().cpu().item()
                    transform_ckpt[data_id]['k5'] = distort_lNet[idx, 6].detach().cpu().item()
                    transform_ckpt[data_id]['k6'] = distort_lNet[idx, 7].detach().cpu().item()
                
                logger_py.info(f'Backup epoch {epoch}: {transform_ckpt[data_id]}')

            # write to file
            with open(transMatrix_path.as_posix(), 'w') as f:
                json.dump(transform_ckpt, f, indent=2)
            f.close()
            
            logger_py.info(f'Backup epoch {epoch}: {time.strftime("%Y-%m-%d_%H-%M-%S")} to {transMatrix_path}')

        # torch.cuda.empty_cache()
   
    pbar.set_description(f'Epoch {epoch}, loss={loss}')


# training loop
if WHICH_TASK == 'matchImg2Pc_1':
    fitImage2Pc(CAM_ID)
elif WHICH_TASK == 'matchImg2Pc':
    trained_cam_ids = []
    for data_id in transformMatrix.keys():
        cam_id = data_id.split('_')[1]
        if cam_id in trained_cam_ids:
            continue
        trained_cam_ids.append(cam_id)
        fitImage2Pc(cam_id)
