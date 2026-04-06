import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from scipy.stats import gaussian_kde

from utils.camFormat import *
from utils.general import *


def plotImg(img, title='', show=True):
    plt.imshow(img)
    plt.title(title)
    if show:
        plt.show()

    return plt

def plot2Imgs(img1, img2, title='', show=True):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.title(title)
    if show:
        plt.show()

    return fig

def plot2ImgsGray(img1, img2, title='', show=True):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1, cmap='gray')
    axs[1].imshow(img2, cmap='gray')
    plt.title(title)
    if show:
        plt.show()

    return fig

def loadGrayImg(img_path):
    # 8-bit grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Error: No such file {img_path}')
    return img

def loadRGBImg(rgb_path):
    # 8-bit RGB image
    img = cv2.imread(rgb_path)
    if img is None:
        print(f'Error: No such file {rgb_path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def loadUnchangedImg(img_path):
    # As it is, RGBA image, no conversion
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'Error: No such file {img_path}')
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

def compareRealvsSampled(real_path, sampled_path, save_path, cam_id=None, warp=False, show=False):
    '''
    Compare real images with sampled images in same plot
    '''
    if not save_path.exists():
        save_path.mkdir(parents=True)
    parent_path = real_path.parent
    # loop through all images of the real path and compare with same sampled images
    for filename in tqdm(sorted(real_path.iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        if cam_id is None or cam_id in filename.stem:
            real_img = loadRGBImg(filename)
            fNameWoExt = filename.stem
            if warp:
                fNameWoExt = fNameWoExt + '_warped'
            try:
                sampled_img = cv2.cvtColor(cv2.imread(Path(sampled_path, fNameWoExt + '_color.png')), cv2.COLOR_BGR2RGB)
            except:
                print(f'Error: No such file {Path(sampled_path, fNameWoExt + "_color.png")}')
                continue
            if sampled_img is None:
                print(f'No sampled image for {fNameWoExt}')
                continue
            
            # 4x faster than matplotlib
            final_img = cv2.hconcat([real_img, sampled_img])
            plotImg(final_img, title=cam_id, show=show)
            if not save_path.exists():
                save_path.mkdir(parents=True)
            
            # save image
            cv2.imwrite(Path(save_path, 'cmp' + fNameWoExt + '.png').as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    print('Finished comparing real and sampled images')


def overlayRealvsSampled(real_path, sampled_path, save_path, cam_id=None, warp=False, show=False):
    '''
    Overlap real images with sampled images in same plot
    '''
    parent_path = real_path.parent
    # loop through all images of the real path and compare with same sampled images
    for filename in tqdm(sorted(real_path.iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        if cam_id is None or cam_id in filename.stem:
            real_img = loadRGBImg(filename)
            fNameWoExt = filename.stem
            if warp:
                fNameWoExt = fNameWoExt + '_warped'
            sampled_img = cv2.cvtColor(cv2.imread(Path(sampled_path, fNameWoExt + '_color.png')), cv2.COLOR_BGR2RGB)
            if sampled_img is None:
                print(f'No sampled image for {fNameWoExt}')
                continue

            # overlap the images
            overl_img = cv2.addWeighted(real_img, 0.8, sampled_img, 0.6, 0)

            plotImg(overl_img, title=fNameWoExt, show=show)

            if not save_path.exists():
                save_path.mkdir(parents=True)

            # save image
            cv2.imwrite(Path(save_path, 'ovl' + fNameWoExt + '.png').as_posix(), cv2.cvtColor(overl_img, cv2.COLOR_RGB2BGR))

    print('Finished overlapping real and sampled images')


def compareWarping(scene_path, camMetadata,  task, show=False):
    ov_path = Path(scene_path, 'CompareReVsIm')
    if not ov_path.exists():
        ov_path.mkdir(parents=True)

    if task == 'compare':
        for filename in tqdm(sorted(Path(scene_path, 'images').iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
            fNameWoExt = filename.stem
            cam_id = fNameWoExt.split('_')[1]
            fNameWoExt_warped = fNameWoExt + '_warped'
            img = cv2.cvtColor(cv2.imread(Path(scene_path, 'SimImages', fNameWoExt + '_color.png')), cv2.COLOR_BGR2RGB)
            img_warped = cv2.cvtColor(cv2.imread(Path(scene_path, 'SimImages', fNameWoExt_warped + '_color.png')), cv2.COLOR_BGR2RGB)
            if img is None or img_warped is None:
                print(f'No image for {fNameWoExt} or {fNameWoExt_warped}')
                continue

            final_img = cv2.hconcat([img, img_warped])
            
            title_str = f'camera {cam_id} generation {camMetadata[cam_id]["camGeneration"]}'
            plotImg(final_img, title=title_str, show=show)

            # save image
            fGen = f'_gen{camMetadata[cam_id]["camGeneration"]}'
            cv2.imwrite(Path(ov_path, 'cmpImgWarp' + fNameWoExt + fGen + '.png').as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    elif task == 'overlay':
        for filename in tqdm(sorted(Path(scene_path, 'images').iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
            fNameWoExt = filename.stem
            fNameWoExt_warped = fNameWoExt + '_warped'
            cam_id = fNameWoExt.split('_')[1]
            img = cv2.cvtColor(cv2.imread(Path(scene_path, 'CompareReVsIm', 'ovl' + fNameWoExt + '.png')), cv2.COLOR_BGR2RGB)
            img_warped = cv2.cvtColor(cv2.imread(Path(scene_path, 'CompareReVsIm', 'ovl' + fNameWoExt_warped + '.png')), cv2.COLOR_BGR2RGB)
            if img is None or img_warped is None:
                print(f'No image for {fNameWoExt} or {fNameWoExt_warped}')
                continue

            final_img = cv2.hconcat([img, img_warped])
            title_str = f'camera {cam_id} generation {camMetadata[cam_id]["camGeneration"]}'
            plotImg(final_img, title=title_str, show=show)

            # save image
            fGen = f'_gen{camMetadata[cam_id]["camGeneration"]}'
            cv2.imwrite(Path(ov_path, 'cmpOvlImgWarp' + fNameWoExt + fGen + '.png').as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    
    print('Finished comparing warped images')


def overlaySatCloud(sat_fPath, satCloud_fPath, satCloudTemp_fPath, satRoadNet_fPath, satCloudcfg_dict, show=False):
    '''
    Overlay satellite image with cloud satellite image
    Input:
        sat_fPath: Path to satellite image
        satCloud_fPath: Path to cloud satellite image
        satRoadNet_fPath: Path to roadNet satellite image
        satCloudcfg_dict: Dictionary with cloud satellite image configuration
        show: Show images
    '''
    cropLon_min = satCloudcfg_dict['cropLon_min']
    cropLon_max = satCloudcfg_dict['cropLon_max']
    cropLat_min = satCloudcfg_dict['cropLat_min']
    cropLat_max = satCloudcfg_dict['cropLat_max']

    # load images
    sat_img = loadRGBImg(sat_fPath)
    satCloud_img = loadGrayImg(satCloud_fPath)
    satCloudTemp_img = loadGrayImg(satCloudTemp_fPath)
    satRoadNet_img = loadUnchangedImg(satRoadNet_fPath)
    satRoadNet_img = satRoadNet_img[:satCloud_img.shape[0], :satCloud_img.shape[1], :]

    # crop and resize cloud image
    if cropLon_max == 0:
        cropLon_max = sat_img.shape[1]
    if cropLat_max == 0:
        cropLat_max = sat_img.shape[0]
    
    plotImg(sat_img, title='Satellite image', show=show)
    plotImg(satCloud_img, title='Cloud satellite image', show=show)
    plotImg(satCloudTemp_img, title='Temperature cloud satellite image', show=show)
    plotImg(satRoadNet_img, title='RoadNet satellite image', show=show)

    cropedCloud = satCloud_img[cropLat_min:cropLat_max, cropLon_min:cropLon_max]
    cropedCloudTemp = satCloudTemp_img[cropLat_min:cropLat_max, cropLon_min:cropLon_max]
    cropedRoadNet = satRoadNet_img[cropLat_min:cropLat_max, cropLon_min:cropLon_max]
    satCloud_img = cv2.resize(cropedCloud, (sat_img.shape[1], sat_img.shape[0]))
    satCloudTemp_img = cv2.resize(cropedCloudTemp, (sat_img.shape[1], sat_img.shape[0]))
    satRoadNet_img = cv2.resize(cropedRoadNet, (sat_img.shape[1], sat_img.shape[0]))

    # overlay roudNet on sat_img with numpy for speed
    satRoadNet_alpha = satRoadNet_img[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
    satRoadNet_rgb = satRoadNet_img[:, :, :3]
    alpha_mask = satRoadNet_alpha[:, :, np.newaxis]
    background_subsection = sat_img

    # combine the background with the overlay image weighted by alpha
    sat_img = (background_subsection * (1 - alpha_mask) + satRoadNet_rgb * alpha_mask).astype(np.uint8)

    # overlay cloud image on sat image
    # TODO: mask out snow regions
    satCloud_alpha = (satCloudTemp_img > satCloudcfg_dict['cloudThresh']).astype(np.float16)*0.8
    alpha_mask = satCloud_alpha[:, :, np.newaxis]

    overl_img = (sat_img * (1 - alpha_mask) + satCloudTemp_img[:, :, np.newaxis] * alpha_mask).astype(np.uint8)

    plotImg(overl_img, title='Satellite and Cloud overlay', show=show)
    
    # save preprocessed satCloud
    fName = satCloud_fPath.stem
    split_fName = fName.split('_')
    fName = split_fName[0] + '_' + split_fName[1] + '_' + split_fName[2] + '_' + split_fName[3]
    save_fPath = Path(satCloud_fPath.parent, fName + '_overlay.png')
    overl_img = cv2.cvtColor(overl_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_fPath.as_posix(), overl_img)

    save_fPath = Path(satCloud_fPath.parent, fName + '_preprocess.jpg')
    cv2.imwrite(save_fPath.as_posix(), satCloudTemp_img)


def camCutout(parent_path):
    parent_path = Path(parent_path)
    for folder in parent_path.iterdir():
        if folder.is_dir():
            # open file in folder
            for file in folder.iterdir():
                if file.suffix == '.jpg':
                    img = loadRGBImg(file)

                    # find center all
                    all_max = img.shape[0]
                    all_center = all_max // 2

                    # find cut_min
                    cut_min = 0
                    cut_minOld = 0
                    Offset = 220
                    for pix_y in range(1, all_max):
                        for pix_x in range(Offset, img.shape[1]-Offset):
                            if img[pix_y, pix_x, 0] < 50:
                                cut_min = pix_y
                                break
                        if cut_min == cut_minOld:
                            break
                        cut_minOld = cut_min

                    # find cut_max
                    cut_max = all_max
                    cut_maxOld = all_max
                    for pix_y in range(all_max-1, 0, -1):
                        for pix_x in range(Offset, img.shape[1]-Offset):
                            if img[pix_y, pix_x, 0] < 50:
                                cut_max = pix_y
                                break
                        if cut_max == cut_maxOld:
                            break
                        cut_maxOld = cut_max

                    # find center cut
                    cut_center = (cut_max + cut_min) // 2

                    delta_center = cut_center - all_center

                    # make red line in image at cut_min, cut_max
                    red = [0, 0, 255]
                    for band in range(-8, 8):
                        img[cut_min+band, :] = red
                        img[cut_max+band, :] = red

                    # save image
                    save_path = Path(folder, 'cutoutResult.png')
                    cv2.imwrite(save_path.as_posix(), img)

                    # write results to file json
                    save_path = Path(folder, 'cutoutResult.json')
                    file_dict = {
                        'img_height': all_max,
                        'cut_min': cut_min,
                        'cut_max': cut_max,
                        'cut_center': cut_center,
                        'delta_center': delta_center
                    }
                    with open(save_path, 'w') as f:
                        json.dump(file_dict, f, indent=2)
                        f.close()

def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given rgb image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    try:
        img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    except: # no color image 
        print('No color image')
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def overlay1Pano(scene_path, camMetadata, cam_id, show=False):
    parent_path = Path(scene_path, 'CompareReVsIm')
    if not parent_path.exists():
        parent_path.mkdir(parents=True)

    try:
        panoReal = loadRGBImg(Path(scene_path, '360', f'{cam_id}.jpg').as_posix())
        panoSample = loadRGBImg(Path(scene_path, 'SimPano', f'{cam_id}_color.png').as_posix())
    except:
        print('No panorama images found')
        return
    
    # resize panoSample to panoReal and overlay
    if panoReal.shape[0] != panoSample.shape[0] or panoReal.shape[1] != panoSample.shape[1]:
        panoSample = cv2.resize(panoSample, (panoReal.shape[1], panoReal.shape[0]))
    overl_img = cv2.addWeighted(panoReal, 0.8, panoSample, 0.5, 0)

    plotImg(overl_img, title=cam_id, show=show)

    # save image
    fGen = f'_gen{camMetadata[cam_id]["camGeneration"]}'
    cv2.imwrite(Path(parent_path, f'ovlPano_{cam_id}' + fGen + '.png').as_posix(), cv2.cvtColor(overl_img, cv2.COLOR_RGB2BGR))


def overlayPanos(scene_path, camMetadata):
    for filename in tqdm(sorted(Path(scene_path, '360').iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        cam_id = filename.stem.split('.')[0]
        overlay1Pano(scene_path, camMetadata, cam_id)

    print('Finished overlaying panoramas')

def compare1Pano(scene_path, camMetadata, cam_id, show=False):
    parent_path = Path(scene_path, 'CompareReVsIm')
    if not parent_path.exists():
        parent_path.mkdir(parents=True)

    try:
        panoReal = loadRGBImg(Path(scene_path, '360', f'{cam_id}.jpg').as_posix())
        panoSample = loadRGBImg(Path(scene_path, 'SimPano', f'{cam_id}_color.png').as_posix())
    except:
        print('No panorama images found')
        return
    
    # resize panoSample to panoReal and compare
    if panoReal.shape[0] != panoSample.shape[0] or panoReal.shape[1] != panoSample.shape[1]:
        panoSample = cv2.resize(panoSample, (panoReal.shape[1], panoReal.shape[0]))
    final_img = cv2.vconcat([panoReal, panoSample])

    plotImg(final_img, title=cam_id, show=show)
    
    # save image
    fGen = f'_gen{camMetadata[cam_id]["camGeneration"]}'
    cv2.imwrite(Path(parent_path, f'cmpPano_{cam_id}' + fGen + '.png').as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

def comparePanos(scene_path, camMetadata):
    for filename in tqdm(sorted(Path(scene_path, '360').iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        cam_id = filename.stem.split('.')[0]
        compare1Pano(scene_path, camMetadata, cam_id)

def pano2plane_1(scene_path, transformMatrix, cam_id, show=False):
    save_path = Path(scene_path, 'imagesPlane')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    try:
        panoReal = loadRGBImg(Path(scene_path, '360', f'{cam_id}.jpg').as_posix())
    except:
        print('No panorama images found')
        return
    
    # get camera parameters
    target_fov_wid = transformMatrix[cam_id]['camera_angle_wid_img'] * 180 / np.pi
    target_fov_hei = transformMatrix[cam_id]['camera_angle_hei'] * 180 / np.pi
    fov_wid_overl = (transformMatrix[cam_id]['overlapPix_wid_img'] * transformMatrix[cam_id]['degPerPix_wid_img'])
    output_size = (int(transformMatrix[cam_id]['wid']), int(transformMatrix[cam_id]['hei']))
    pano_fov_wid = transformMatrix[cam_id]['camera_angle_wid_pano'] * 180 / np.pi
    pano_fov_hei = transformMatrix[cam_id]['camera_angle_hei'] * 180 / np.pi
    pitch = transformMatrix[cam_id]['tilt_angle']
    
    # loop through all frames of the panorama
    fr_num = 0
    for data_id in transformMatrix[cam_id].keys():
        if cam_id in data_id:
            fileName = transformMatrix[cam_id][data_id]['file_path'].split('/')[-1]
            save_fName = Path(save_path, fileName)
            yaw = target_fov_wid/2 + fr_num * (target_fov_wid - fov_wid_overl)
            plane_img, map_x, map_y = cylindrical_to_plane_projection(panoReal, target_fov_wid, target_fov_hei, yaw, pitch, output_size, pano_hor_fov=pano_fov_wid, pano_vert_fov=pano_fov_hei)
            fr_num += 1

            cv2.imwrite(save_fName.as_posix(), cv2.cvtColor(plane_img, cv2.COLOR_RGB2BGR))
   
    if show:
        plot2Imgs(plane_img, panoReal, title=cam_id)
        plot2ImgsGray(map_x, map_y, title='Map_x:l, Map_y:r')

def pano2plane(scene_path, transformMatrix):
    '''
    Split up 360° images into several planar images
    Args:
        - scene_path: Path to scene folder: full path
        - transformMatrix: Dictionary with transformation matrix
    '''
    for filename in tqdm(sorted(Path(scene_path, '360').iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        cam_id = filename.stem.split('.')[0]
        pano2plane_1(scene_path, transformMatrix, cam_id)


def compareSplittedPano(imgCyl_path, imgPlane_path, save_path, cam_id=None, show=False):
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for filename in tqdm(sorted(imgCyl_path.iterdir(), key=lambda path: int(path.stem.rsplit("-")[0]))):
        if cam_id is None or cam_id in filename.stem:
            imgCyl = loadRGBImg(filename)
            imgPlane = loadRGBImg(Path(imgPlane_path, filename.name))

            final_img = cv2.hconcat([imgCyl, imgPlane])
            plotImg(final_img, title=filename.stem, show=show)

            # save image
            cv2.imwrite(Path(save_path, 'cmpImg' + filename.name).as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

def compareCycVsPlane(scene_path, imgCylCmp_path, imgPlaneCmp_path, save_path, task='compare'):
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    if task == 'compare':
        for filename in tqdm(Path(imgCylCmp_path).iterdir()):
            if 'cmp' in filename.stem:
                imgCyl = loadRGBImg(filename)
                imgPlane = loadRGBImg(Path(imgPlaneCmp_path, filename.name))

                final_img = cv2.vconcat([imgCyl, imgPlane])

                cv2.imwrite(Path(save_path, 'cmp' + filename.name).as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    elif task == 'overlay':
        for filename in tqdm(Path(imgCylCmp_path).iterdir()):
            if 'ovl' in filename.stem:
                imgCyl = loadRGBImg(filename)
                imgPlane = loadRGBImg(Path(imgPlaneCmp_path, filename.name))

                overl_img = cv2.hconcat([imgCyl, imgPlane])

                cv2.imwrite(Path(save_path, 'ovl' + filename.name).as_posix(), cv2.cvtColor(overl_img, cv2.COLOR_RGB2BGR))
    

def comparePoseAtOtherTime(img_path, img_path2):
    for filename in tqdm(sorted(Path(img_path).iterdir())):
        cam_id = filename.stem.split('_')[1].split('_')[0]
        comparePoseAtOtherTime_1(img_path, img_path2, cam_id)

def comparePoseAtOtherTime_1(img_path, img_path2, cam_id, show=False):
    save_path = Path(Path(img_path).parent, 'PoseTimeDiff')
    scene_path = Path(img_path).parent
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for file in Path(img_path).iterdir():
        if cam_id in file.stem:
            img_id = file.stem.split('.')[0]
            img_id_local = img_id.split('_')[1] + '_' + img_id.split('_')[2]

            # check if same frame exist in other folder
            foundFile = False
            otherFile = None
            for file2 in Path(img_path2).iterdir():
                if img_id_local in file2.stem:
                    foundFile = True
                    otherFile = file2
                    break
            
            if not foundFile:
                continue

            img_filePath = file
            img_filePath2 = file2

            img1 = loadRGBImg(img_filePath.as_posix())
            img2 = loadRGBImg(img_filePath2.as_posix())

            # load 2D matching points and add to image
            pointMatch_path = Path(scene_path, 'pointMatch', 'pointMatch.json')
            pointMatches = getPointMatches(pointMatch_path)

            points2d = pointMatches[cam_id][img_id]["match2D"]
            for point in points2d:
                cv2.circle(img1, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
                cv2.circle(img1, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)
                cv2.circle(img2, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
                cv2.circle(img2, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)

            # combine images
            final_img = cv2.hconcat([img1, img2])

            # save image
            save_file = Path(save_path, f'{img_id}.png')
            cv2.imwrite(save_file.as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            if show:
                plotImg(final_img, title=img_id)

def comparePoseTraining(realPlane_path, transfMatrix_before, transfMatrix_after, config):
    for filename in tqdm(sorted(Path(realPlane_path).iterdir())):
        cam_id = filename.stem.split('_')[1].split('_')[0]
        comparePoseTraining_1(realPlane_path, transfMatrix_before, transfMatrix_after, cam_id, config)

def comparePoseTraining_1(realPlane_path, transfMatrix_before, transfMatrix_after, cam_id, config, show=False):
    save_path = Path(Path(realPlane_path).parent, 'PoseTraining')
    scene_path = Path(realPlane_path).parent
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for file in Path(realPlane_path).iterdir():
        if cam_id in file.stem:
            img_id = file.stem.split('.')[0]

            # load image
            img = loadRGBImg(file.as_posix())
            img_after = img.copy()

            # load point matching
            pointMatch_path = Path(scene_path, 'pointMatch', 'pointMatch.json')
            pointMatches = getPointMatches(pointMatch_path)
            points2d = pointMatches[cam_id][img_id]["match2D"]
            points3d = np.array(pointMatches[cam_id][img_id]["match3D"])

            # project points to plane before and after
            intrinsics = getIntrinsics(transfMatrix_before, cam_id, img_id)
            distortion = getDistortion(transfMatrix_before, cam_id, img_id)
            w2c = getW2c(transfMatrix_before, cam_id, img_id)

            rot, _ = cv2.Rodrigues(w2c[:3, :3])
            translation = w2c[:3, 3]

            #points3d_masked, _ = mask_points_cameraAxis(points3d, w2c, 0, 1000)
            projected_2d_before, _ = cv2.projectPoints(points3d, rot, translation, intrinsics, distortion)

            intrinsics = getIntrinsics(transfMatrix_after, cam_id, img_id)
            distortion = getDistortion(transfMatrix_after, cam_id, img_id)
            w2c = getW2c(transfMatrix_after, cam_id, img_id, config=config, inPix=True)

            rot, _ = cv2.Rodrigues(w2c[:3, :3])
            translation = w2c[:3, 3]

            #points3d_masked, _ = mask_points_cameraAxis(points3d, w2c, 0, 1000)
            projected_2d_after, _ = cv2.projectPoints(points3d, rot, translation, intrinsics, distortion)

            # draw target points and trained points on image
            for point in points2d:  # target points
                cv2.rectangle(img, (int(point[0]-5), int(point[1]-5)), (int(point[0]+5), int(point[1]+5)), (0, 255, 0), -1)
                cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)
            for point in projected_2d_before:   # trained points before
                cv2.rectangle(img, (int(point[0][0]-5), int(point[0][1]-5)), (int(point[0][0]+5), int(point[0][1]+5)), (255, 0, 0), -1)
                cv2.circle(img, (int(point[0][0]), int(point[0][1])), 1, (0, 0, 0), -1)

            for point in points2d:  # target points
                cv2.rectangle(img_after, (int(point[0]-5), int(point[1]-5)), (int(point[0]+5), int(point[1]+5)), (0, 255, 0), -1)
                cv2.circle(img_after, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)
            for point in projected_2d_after:    # trained points after
                cv2.rectangle(img_after, (int(point[0][0]-5), int(point[0][1]-5)), (int(point[0][0]+5), int(point[0][1]+5)), (255, 0, 0), -1)

            # combine images
            final_img = cv2.hconcat([img, img_after])

            # save image
            save_file = Path(save_path, f'{img_id}.png')
            cv2.imwrite(save_file.as_posix(), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            if show:
                plotImg(final_img, title=img_id)
    
def invert_depth_image(depth_image):
    """
    Inverts the depth values of an image from far to near.
    Args:
        depth_image (np.array): The depth image to invert.
    """
    # Find the maximum depth value in the image
    max_depth = np.max(depth_image)
    
    # Invert the depth image by subtracting each pixel from the max depth
    inverted_depth = max_depth - depth_image
    
    return inverted_depth

def analyze_fxFy(transfMatrix, file_name, show=False):
    fx_list = []
    fy_list = []
    for cam_key, cam in transfMatrix.items():
        for img_key, img in cam.items():
            if cam_key in img_key:
                fx_list.append(img['fx'])
                fy_list.append(img['fy'])

    # plot mean, std, min, max of fx and fy
    fx_list = np.array(fx_list)
    fy_list = np.array(fy_list)
    fx_mean = np.mean(fx_list)
    fy_mean = np.mean(fy_list)
    fx_std = np.std(fx_list)
    fy_std = np.std(fy_list)
    fx_min = np.min(fx_list)
    fy_min = np.min(fy_list)
    fx_max = np.max(fx_list)
    fy_max = np.max(fy_list)

    print(f'fx mean: {fx_mean}, std: {fx_std}, min: {fx_min}, max: {fx_max}')
    print(f'fy mean: {fy_mean}, std: {fy_std}, min: {fy_min}, max: {fy_max}')

    if show:
        fig, ax = plt.subplots(figsize=(10, 6))  # One subplot for combined histogram
    
        # Plotting fx and fy together
        ax.hist([fx_list, fy_list], bins=50, alpha=0.5, label=['fx', 'fy'], 
                color=['#1f77b4', '#ff7f0e'], edgecolor='black')
        
        ax.set_title('Distribution of fx and fy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Values', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=10)
        fig.tight_layout()

        # save histogram
        file_name = Path(file_name)
        save_path = Path(file_name.parent, f'fxFy_hist_{file_name.stem}.png')
        plt.savefig(save_path.as_posix())        
        plt.show()

        # boxplot
        plt.title('Distribution of fx and fy')
        plt.boxplot(np.array([fx_list, fy_list]).T, labels=['fx', 'fy'], showmeans=True)
        save_path = Path(file_name.parent, f'fxFy_box_{file_name.stem}.png')
        plt.savefig(save_path.as_posix())
        plt.show()

def plot_camDensity(transformMatrix : dict, save_path, show=False):
    '''
    Plot camera density on a 2D map
    Args:
        transformMatrix (dict): Dictionary containing the transformation matrices
        save_path (str): Path to save the plot
        show (bool): Show the plot
    '''
    camPoints = []

    for cam_key, cam in transformMatrix.items():
        for img_key, img in cam.items():
            if cam_key in img_key:
                log = img['transform_matrix'][0][3]
                lat = img['transform_matrix'][1][3]
                camPoints.append([log, lat])
                break
    
    camPoints = np.array(camPoints)
    print("camPoints shape: ", camPoints.shape)

    # plot cam density
    fig = plt.figure(figsize=(10, 6))
    
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = gaussian_kde(camPoints.T)
    x, y = camPoints.T
    nbins = 20
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.scatter(x, y, c='r', marker='o')    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.title('Camera density')
    plt.grid(True)
    plt.tight_layout()

    # save plot
    save_path = Path(save_path)
    save_file = Path(save_path, 'camDensity.png')
    plt.savefig(save_file.as_posix())

    if show:
        plt.show()

