import open3d as o3d
import trimesh
import pyrender
import numpy as np
import cv2
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import plotly.graph_objects as go

from utils.general import *
from utils.camFormat import *
from dataVisu_engine.TwoD_helpers import *



def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def loadPointCloud_ply(ply_path, name_ply, show=False):
    """
    Visualize a ply file
    Input:
        ply_path: Path to the ply file
        name_ply: Name of the ply file
        show: Boolean to show the ply file
    Return:
        pcd: Open3D point cloud object
    """
    ply_path = Path(ply_path, name_ply)
    pcd = o3d.io.read_point_cloud(ply_path.as_posix())
    if show:
        o3d.visualization.draw_geometries([pcd])

    return pcd

def loadMesh_ply(ply_path, name_ply, show=False):
    """
    Visualize a ply file
    Input:
        ply_path: Path to the ply file
        name_ply: Name of the ply file
        show: Boolean to show the ply file
    Return:
        mesh: Open3D mesh object
    """
    ply_path = Path(ply_path, name_ply)
    mesh = o3d.io.read_triangle_mesh(ply_path.as_posix(), enable_post_processing=True)
    mesh.compute_vertex_normals()
    if show:
        o3d.visualization.draw_geometries([mesh])

    return mesh

def loadTriMesh_ply(ply_path, name_ply, show=False):
    '''
    load ply file as trimesh object
    Args:
        ply_path: Path to the ply file
        name_ply: Name of the ply file
        show: Boolean to show the ply file
    Return:
        mesh: trimesh object
    '''
    ply_path = Path(ply_path, name_ply)
    # Load the PLY file
    mesh = trimesh.load(ply_path.as_posix())

    if show:
        mesh.show()

    return  mesh

def genPYLfromRGBdepth(config, rgb_Fpath, depth_Fpath, show=False):
    '''
    Create ply point cloud from rgb and depth image
    Input:
        rgb_Fpath: Path to the RGB image
        depth_Fpath: Path to the Depth image
    Return:
        pcd: Open3D point cloud object
    '''
    # Read RGB and Depth images without open3d
    rgb_raw = loadRGBImg(rgb_Fpath.as_posix()) 
    depth_raw = loadUnchangedImg(depth_Fpath.as_posix())

    rgb = np.array(rgb_raw).astype(np.float64)
    depth = np.array(depth_raw).astype(np.float64)
    rgb = rgb / np.max(rgb)
    rgb = resize(rgb, (depth.shape[0], depth.shape[1], 3), anti_aliasing=True).astype(np.float64)

    # scale height according to long, lat scaling
    if config.GEE_dem_save_mode == 'I;16':
        depth /= config.GEE_pxScale

    # generate point cloud list
    points = []
    color = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            z = depth[i, j, 0]
            r = i
            c = j
            points.append([r, c, z])
            color.append(rgb[i, j, :])

    print("Finished point cloud list")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)

    if show:
        o3d.visualization.draw_geometries([pcd])

    return pcd

def showPointCloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def showMesh(mesh):
    o3d.visualization.draw_geometries([mesh])

def meshFromPointCloud(pcd, radius=0.01, max_nn=40, width=4, show=False):
    '''
    Create a mesh from a point cloud
    Input:
        pcd: Open3D point cloud object
        radius (0.01): radius for normal estimation --> higher, more points included --> smoother, less detailed
        max_nn (40): max number of nearest neighbors --> limits number of neighbors --> lower faster
        width (4): width of one mesh pixel for poisson reconstruction --> lower --> more detailed
    Return:
        mesh: Open3D mesh object
    '''
    # calculate mesh from point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, width=width)
    mesh.compute_vertex_normals()

    # cut the mesh outside the bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    if show:
        o3d.visualization.draw_geometries([mesh])
    return mesh

def savePointCloud_ply(pcd, save_path, fName="pointcloud.ply", write_ascii=False):
    '''
    Save the point cloud to a ply file
    Input:
        pcd: Open3D point cloud object
        save_path: Path to save the point cloud
    '''
    save_path = Path(save_path, fName)
    o3d.io.write_point_cloud(save_path.as_posix(), pcd, write_ascii=write_ascii)
    print(f"Point cloud saved to {save_path}")

def saveMesh_ply(mesh : o3d, save_path, fName="mesh.ply", write_ascii=False):
    '''
    Save the mesh to a ply file
    Input:
        mesh: Open3D mesh object
        save_path: Path to save the mesh
    '''
    save_path = Path(save_path, fName)
    o3d.io.write_triangle_mesh(save_path.as_posix(), mesh, write_ascii=write_ascii)
    print(f"Mesh saved to {save_path}")

def saveTriMesh_ply(mesh : trimesh, save_path, fName="mesh.ply"):
    '''
    Save the mesh to a ply file
    Input:
        mesh: trimesh object
        save_path: Path to save the mesh
    '''
    save_path = Path(save_path, fName)
    bytes = trimesh.exchange.ply.export_ply(mesh, encoding='ascii', vertex_normal=True, include_attributes=True)
    with open(save_path.as_posix(), 'wb') as f:
        f.write(bytes)
    print(f"Mesh saved to {save_path}")

def addCam2Pcd_Rshot(config, pcd, transformMatrix, scale_line=1, gps_coords=True):
    '''
    Add camera poses to the point cloud
    Args:
        config: Config() object
        pcd: Open3D point cloud object
        transformMatrix: dict with camera poses to be added
                        in Roundshot format {cam_id: {data_id: {...}}}
        scale_line: scale of the orientation line
    '''
    currentPoints = pcd.points
    currentColors = pcd.colors

    print("Max/Min currentPoints", np.max(currentPoints), np.min(currentPoints))

    # loop through all poses
    for cam_id, cam in tqdm(transformMatrix.items()):
        coord = [cam['transform_matrix'][0][3],
                 cam['transform_matrix'][1][3],
                 cam['transform_matrix'][2][3]]
        if gps_coords:
            trans_px = geoCoord2Open3Dpx(config, coord)
        else:
            trans_px = [coord[0], coord[1], coord[2]]

        currentPoints.append(trans_px)
        currentColors.append([1, 0, 0])  # red

        # draw line for orientation
        vec = np.array([0, 0, -scale_line])
        rot = np.array(cam['transform_matrix'])[:3, :3]
        rotVec = np.dot(rot, vec)
        for i in range(1, 20):
            pArrow = np.array(trans_px) + rotVec * i
            currentPoints.append(pArrow.tolist())
            currentColors.append([0, 0, 1])  # blue

    pcd.points = o3d.utility.Vector3dVector(currentPoints)
    pcd.colors = o3d.utility.Vector3dVector(currentColors)

def addCam2Pcd(config, pcd, transformMatrix, scale_line=1):
    '''
    Add camera poses to the point cloud
    Args:
        config: Config() object
        pcd: Open3D point cloud object
        transformMatrix: dict with camera poses to be added, {cam_id: {data_id: {...}}}
        scale_line: scale of the orientation line
    '''
    currentPoints = pcd.points
    currentColors = pcd.colors

    print("Max/Min currentPoints", np.max(currentPoints), np.min(currentPoints))

    # loop through all poses
    for cam in tqdm(transformMatrix.values()):
        coord = [cam['transform_matrix'][0][3], 
                cam['transform_matrix'][1][3], 
                cam['transform_matrix'][2][3]]

        trans_px = geoCoord2Open3Dpx(config, coord)

        currentPoints.append(trans_px)
        currentColors.append([1, 0, 0])     # red
    
        # draw line for orientation
        vec = np.array([0, 0, -scale_line])
        rot = np.array(cam['transform_matrix'])[:3,:3]
        rotVec = np.dot(rot, vec)
        for i in range(1, 20):
            pArrow = np.array(trans_px) + rotVec * i
            pArrow = pArrow.tolist()
            currentPoints.append(pArrow)
            currentColors.append([0, 0, 1])    # blue

    pcd.points = o3d.utility.Vector3dVector(currentPoints)
    pcd.colors = o3d.utility.Vector3dVector(currentColors)

def addCloud2Ply(config, pcd, satCloud_img, height_img):
    '''
    Add cloud to the point cloud
    Input:
        config: Config() object
        pcd: Open3D point cloud object
        satCloud_img: Cloud image
    '''
    delta_vTemp = config.SatCloud_delta_vTemp
    null_vTemp = config.SatCloud_null_vTemp
    delta_tempH = config.SatCloud_delta_tempH
    null_tempH = config.SatCloud_null_tempH
    currentPoints = pcd.points
    currentColors = pcd.colors
    currentNormals = pcd.normals
    height_img = height_img[:,:,0]

    # calculate cloud height and mask cloud
    maskCloud = satCloud_img > config.SatCloud_cloudThresh 
    temp = delta_vTemp * satCloud_img + null_vTemp
    cloudHeight = delta_tempH * temp + null_tempH

    # scale cloud image for approximating the cloud height
    cloudHeight_px = cloudHeight / config.GEE_pxScale
    height_img_px = height_img
    if config.GEE_dem_save_mode == 'I;16':
        height_img_px = (height_img_px / config.GEE_pxScale).astype(np.float64)

    # add cloud points to the point cloud
    for i in tqdm(range(satCloud_img.shape[0])):
        for j in range(satCloud_img.shape[1]):
            if not maskCloud[i,j]:  # check if cloud is present
                continue

            # check if cloud is present and > 100m above ground
            land_h_px = height_img_px[i,j]
            cloud_h_px = cloudHeight_px[i,j]
            if (cloud_h_px > land_h_px + (200 / config.GEE_pxScale)):
                # random heights
                ran_i = np.random.randint(-500, 500) / config.GEE_pxScale
                ran_j = np.random.randint(-500, 500) / config.GEE_pxScale
                # random point offset
                ran_i = int(i + ran_i)
                ran_j = int(j + ran_j)
                if ran_i < 0:
                    ran_i = 0
                if ran_j < 0:
                    ran_j = 0
                if ran_i >= satCloud_img.shape[0]:
                    ran_i = satCloud_img.shape[0] - 1
                if ran_j >= satCloud_img.shape[1]:
                    ran_j = satCloud_img.shape[1] - 1
                ran_hOffset = cloud_h_px * config.GEE_pxScale / 10
                ran_min_px = cloud_h_px - (config.SatCloud_avg_cloudHeight + np.random.randint(-ran_hOffset/10, ran_hOffset)) / config.GEE_pxScale
                height_min_px = height_img_px[ran_i,ran_j] + (100 / config.GEE_pxScale)
                if ran_min_px < height_min_px:
                    ran_min_px = height_min_px
                if ran_min_px > cloud_h_px:
                    continue
                ran_height_px = [np.random.randint(ran_min_px, cloud_h_px) for _ in range(3)]

                # add random cloud points
                for ran_h_px in ran_height_px:
                    # color
                    if (ran_h_px * config.GEE_pxScale) > 8000:
                        gray_level = np.random.randint(220, 255)
                    else:
                        gray_level = np.random.randint(100, 255)
                    delta_gMax = 255 - gray_level
                    if delta_gMax > 25:
                        delta_gMax = 25
                    gray_blue = np.random.randint(gray_level, gray_level+delta_gMax)
                    if delta_gMax > 15:
                        delta_gMax = 15
                    gray_green = np.random.randint(gray_level, gray_level+delta_gMax)
                    color = [gray_level, gray_green, gray_blue]
                    # normalize color
                    color = [c / 255 for c in color]

                    # add cloud point                  
                    cloudP = [ran_i, ran_j, ran_h_px]
                    currentPoints.append(cloudP)
                    currentColors.append(color)
                    currentNormals.append([0, 0, 1])

    pcd.points = o3d.utility.Vector3dVector(currentPoints)
    pcd.colors = o3d.utility.Vector3dVector(currentColors)
    pcd.normals = o3d.utility.Vector3dVector(currentNormals)

def sampleImg_TriMesh(save_path, mesh, c2w, intrinsics, img_id, depth_inversion=True, filter=True, rgbd=False, show=False, save_result=True):
    '''
    Sample images from a trimesh
    Input:
        save_path: relative path after home path
        mesh: Trimesh object
        c2w: camera pose np matrix
        intrinsics: camera intrinsics as dict, (fx, fy, cx, cy, wid, hei)
        img_id: image id for saving string
        depth_inversion: invert depth image (True: far-to-near, False: near-to-far)
        filter: filter out images with low visibility (True: filter, False: no filter)
    Output:
        True/False: if image is saved/not saved
    '''
    home_path = getHomePath()

    # Create a scene with the mesh
    save_path = Path(home_path, save_path)
    if not save_path.exists() and save_result:
        save_path.mkdir()

    # smooth the mesh
    #mesh = mesh.subdivide()     # subdivide works for depth only, but takes long

    scene = pyrender.Scene()

    # Add the mesh to the scene using the mesh's vertices directly
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)

    # Find the correct node in the scene
    nodes = scene.get_nodes()
    target_node = None
    for node in nodes:
        if isinstance(node, pyrender.Node) and node.mesh == pyrender_mesh:
            target_node = node
            break

    origin = np.eye(4)
    
    if target_node:
        scene.set_pose(target_node, origin)
    else:
        raise ValueError("Could not find the mesh node in the scene.")

    camera = pyrender.IntrinsicsCamera(intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'], znear=0.1, zfar=10000)
    scene.add(camera, pose=c2w)

    if show:
        pyrender.Viewer(scene, use_raymond_lighting=True)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(intrinsics['wid'], intrinsics['hei'])
    flags = 0
    flags |= pyrender.RenderFlags.FLAT
    #flags |= pyrender.RenderFlags.ALL_SOLID       # default setting
    #flags |= pyrender.RenderFlags.ALL_WIREFRAME    # show triangles
    #flags |= pyrender.RenderFlags.FACE_NORMALS    # ?
    #flags |= pyrender.RenderFlags.SKIP_CULL_FACES  # bad gives more holes
    #flags |= pyrender.RenderFlags.VERTEX_NORMALS   # ?
    color, depth = r.render(scene, flags=flags)
    r.delete()

    depthMax_img = np.max(depth)
    depthMin_img = np.min(depth)
    depthMean_img = np.mean(depth)
    if show:
        print("Max/Min/mean depth", depthMax_img, depthMin_img, depthMean_img)

    if filter:
        if depthMax_img < 255:
            depthMax_img = 255
        if depthMean_img < depthMax_img / 30:
            print(f"Filtered out image {img_id} - low visibility, max/mean depth: {depthMax_img}/{depthMean_img}")
            return False

    # normalize depth
    vertices = mesh.vertices
    longest_side = min(np.max(vertices[:, 0]), np.max(vertices[:, 1]))
    depth = depth.clip(0, longest_side - 1)
    depth[depth == 0] = longest_side    # make backgound far away

    depth = scale_to_255(depth, 0, longest_side)

    if depth_inversion:
        depth = invert_depth_image(depth)

    # save the image
    if save_result:
        save_fPath = Path(save_path, str(img_id) + '.png')
        if rgbd:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
            rgbd = np.dstack((color, depth))
            cv2.imwrite(save_fPath.as_posix(), rgbd)
        else:
            cv2.imwrite(save_fPath.as_posix(), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_fPath.as_posix().replace('.png', '_depth.png'), cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
        return True
    else:
        return color, depth

def sample_TriMesh_direct(save_path, mesh : trimesh, c2w : np, intrinsics : list, img_id, depth_inversion=True, show=False):
    # TODO: sample without using pyrender
    print("Not implemented yet")
    '''
    # Create the camera
    fov_x = 2*np.arctan(intrinsics['wid']/(2*intrinsics['fx']))
    fov_y = 2*np.arctan(intrinsics['hei']/(2*intrinsics['fy']))
    camera = trimesh.scene.cameras.Camera(
        resolution=(intrinsics['wid'], intrinsics['hei']),
        fov=(fov_x, fov_y)  # Adjust field of view if needed
    )

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.camera = camera
    scene.camera_transform = c2w

    color = scene.save_image(resolution=(intrinsics['wid'], intrinsics['hei']))
    depth = scene.save_image(resolution=(intrinsics['wid'], intrinsics['hei']), depth=True)
    color = np.frombuffer(color, dtype=np.uint8)  # Convert bytes to numpy array
    color = cv2.imdecode(color, cv2.IMREAD_UNCHANGED)  # Decode into an image
    depth = np.frombuffer(depth, dtype=np.uint8)  # Convert bytes to numpy array
    depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)  # Decode into an image

    if show:
        plt.imshow(color)

    # save image
    save_fPath = Path(save_path, str(img_id) + '.png')
    cv2.imwrite(save_fPath.as_posix(), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    cv2.imwrite(save_fPath.as_posix().replace('.png', '_depth.png'), cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
    '''

def sampleImg_TriMesh_all(config, mesh : trimesh, transformMatrices : dict, folder_name = 'mesh', filter=False, rgbd=False, show=False):
    '''
    Sample images from a trimesh
    Input:
        mesh: Trimesh object
        transformMatrices: array of camera pose
    '''
    save_path = Path(Path(config.data_fileTransformMatrix).parent.parent, folder_name)
    print("Save to:", save_path)

    # loop through all cameras and sample images
    tqdm_bar = tqdm(transformMatrices.items())
    for cam_key, cam in tqdm_bar:
        tqdm_bar.set_description(f'Sample mesh {cam_key}')
        for frame_id in cam.keys():
            if cam_key not in frame_id:
                continue
            # else sample image
            transformMatrix = np.array(transformMatrices[cam_key][frame_id]['transform_matrix'])
        
            t_gps = transformMatrix[:3, 3].tolist()
            t_pix = geoCoord2Open3Dpx(config, t_gps)
            transformMatrix[:3, 3] = np.array(t_pix)

            intrinsics = {'fx': transformMatrices[cam_key][frame_id]['fx'],
                            'fy': transformMatrices[cam_key][frame_id]['fy'],
                            'cx': transformMatrices[cam_key]['cx'],
                            'cy': transformMatrices[cam_key]['cy'],
                            'wid': transformMatrices[cam_key]['wid'],
                            'hei': transformMatrices[cam_key]['hei']
                        }
                
            sampleImg_TriMesh(save_path, mesh, transformMatrix, intrinsics, frame_id, filter=filter, rgbd=rgbd)
    

def render_3d_world_to_camera_opencv(points3d : list, w2c : np, intrinsics_params : list, dist_coeffs = None, point_size=1, depth_max=4000, info=True):
    # using opencv function to do it
    # change w2c to rvec and tvec
    # w2c = opengl_to_opencv(w2c)
    # dist_coeffs (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 element
    projected3D_list = []
    w2c = w2c[:3, :]
    points3d_xyz = points3d['XYZ']
    points3d_color = points3d['COLOR']
    # TODO: use masking function to filter out points behind the camera, evtl. take directly rot, trans from w2c, see cam_lidar
    points3d_xyz = points3d_xyz.T  # 3xN
    points3d_xyz = np.vstack((points3d_xyz, np.ones((1, points3d_xyz.shape[1]))))   # 4xN
    # step2 change to camera coordinate
    points3d_xyz = w2c @ points3d_xyz  # 3xN
    points3d_xyz = points3d_xyz[:3, :] # 3xN

    # select based on depth > 0 and < depth_max
    points3d_xyz = points3d_xyz.T  # Nx3
    z_mask = (points3d_xyz[:, 2] > 0) & (points3d_xyz[:, 2] < depth_max)
    points3d_xyz = points3d_xyz[z_mask]
    points3d_color = points3d_color[z_mask]

    # prevent empty points error
    if len(points3d_xyz) == 0:
        print("No points in the camera view")
        return None, None, None, None

    # save this Nx3 as ply file for visualization using trimesh
    # cloud = trimesh.PointCloud(points3d_xyz)
    # # Save the colored point cloud as a .ply file
    # cloud.export('./street_debug/points_cam_coord.ply')
    
    rvec = cv2.Rodrigues(np.eye(3))[0]
    tvec = np.zeros(3)

    H = int(intrinsics_params['hei'])
    W = int(intrinsics_params['wid'])
    intrinsics = np.array([[intrinsics_params['fx'], 0, intrinsics_params['cx']],
                            [0, intrinsics_params['fy'], intrinsics_params['cy']],
                            [0, 0, 1]])

    # Project 3D points to 2D
    points2d, _ = cv2.projectPoints(points3d_xyz, rvec, tvec, intrinsics, dist_coeffs)

    color_map = np.zeros((H, W, 3), dtype=np.float64)
    depth_map = np.full((H, W), np.inf)
    mask_map = np.zeros((H, W), dtype=np.uint8)

    # Iterate over the projected points and update the depth map
    cnt_points = 0
    for i, (point, color, img_pt) in enumerate(zip(points3d_xyz, points3d_color, points2d)):
        x, y = int(img_pt[0][0]), int(img_pt[0][1])  # 2D image coordinates
        depth = point[-1]  # Depth (z-value in the original 3D point)
        if depth < 0:
            continue
    
        # depth and points within the image bounds
        if (depth > 0 and depth < depth_max and 0 <= x < W and 0 <= y < H):
            # Update depth map with the minimum depth (in case of overlapping points)
            smaller_bool = depth_map[y-point_size//2:y+point_size//2, x-point_size//2:x+point_size//2] < depth
            inf_bool = depth_map[y-point_size//2:y+point_size//2, x-point_size//2:x+point_size//2] != np.inf
            zero_bool = depth_map[y-point_size//2:y+point_size//2, x-point_size//2:x+point_size//2] != 0
            if not (np.any(smaller_bool & inf_bool & zero_bool)):  
                cnt_points += 1
                # bigger point size
                fPoint_size = point_size
                if depth <= 5:
                    fPoint_size = point_size + 150
                if depth > 5 and depth <= 10:
                    fPoint_size = point_size + 80
                if depth > 10 and depth <= 25:
                    fPoint_size = point_size + 50
                if depth > 25 and depth <= 50:
                    fPoint_size = point_size + 20
                if depth > 50 and depth <= 100:
                    fPoint_size = point_size + 10
                if depth > 300 and depth <= 500:
                    fPoint_size = point_size - 5
                if depth > 500:
                    fPoint_size = point_size - 8
                if fPoint_size < 1 or point_size == 1:
                    fPoint_size = 1
                if 0 <= x-fPoint_size//2 and x+fPoint_size//2 < W and 0 <= y-fPoint_size//2 and y+fPoint_size//2 < H:
                    depth_map[y-fPoint_size//2:y+fPoint_size//2+1, x-fPoint_size//2:x+fPoint_size//2+1] = depth
                    color_map[y-fPoint_size//2:y+fPoint_size//2+1, x-fPoint_size//2:x+fPoint_size//2+1] = color
                    mask_map[y-fPoint_size//2:y+fPoint_size//2+1, x-fPoint_size//2:x+fPoint_size//2+1] = 1
       
                projected3D_list.append([x, y, depth])
                    
    if info:
        print(cnt_points, "points in camera view from", len(points2d))
                
    color_map = (color_map * 255).astype(np.uint8)
    
    return depth_map, color_map, mask_map, projected3D_list

def sampleImgCam_OpenCV(config, pcd : o3d, transformMatrices : dict, point_size=1, depth_max=100, filtering = False, warping2Cyl=False, show=False):
    print(f"Sampling with point size {point_size}, depth max {depth_max}, filtering {filtering}, warping2Cyl {warping2Cyl}, show {show}")

    # loop through all cameras and sample images
    for cam in tqdm(transformMatrices.values()):
        try:
            cam_id = cam['cam_id']
        except:
            cam_id = Path(cam['file_path']).stem
        sampleImg1Cam_OpenCV(config, pcd, transformMatrices, cam_id, point_size=point_size, depth_max=depth_max, filtering=filtering, warping2Cyl=warping2Cyl, show=show)


def sampleImg1Cam_OpenCV(config, pcd : o3d, transformMatrices : dict, cam_id, distortion = None, point_size=1, depth_max=100, filtering = False, warping2Cyl=False, show=False):
    home_path = getHomePath()
    if warping2Cyl:
        save_path = Path(home_path, Path(config.data_fileTransformMatrix).parent.parent, 'SimImages_warped')
    else:
        save_path = Path(home_path, Path(config.data_fileTransformMatrix).parent.parent, 'SimImages')
    if not save_path.exists():
        save_path.mkdir()

    points = {}
    points['XYZ'] = np.array(pcd.points)
    points['COLOR'] = np.array(pcd.colors)
    points['POINT3D_ID'] = np.arange(len(points['XYZ']))

    # loop through all cameras and sample images
    cam = transformMatrices[cam_id]
    intrinsic = {
        'fx': 0,
        'fy': 0,
        'cx': cam['cx'],
        'cy': cam['cy'],
        'wid': cam['wid'],
        'hei': cam['hei'],
    }
    
    t_gps = [cam['transform_matrix'][0][3],
                cam['transform_matrix'][1][3],
                cam['transform_matrix'][2][3]]
    t_pix = geoCoord2Open3Dpx(config, t_gps)        
    
    c2w = np.array(cam['transform_matrix'])
    c2w[:3, 3] = t_pix
    c2w = opengl_to_opencv(c2w)
    w2c = np.linalg.inv(c2w)
    intrinsic['fx'] = cam['fx']
    intrinsic['fy'] = cam['fy']

    depth_map, color_map, mask_map, _ = render_3d_world_to_camera_opencv(points, w2c, intrinsic, dist_coeffs=distortion, point_size=point_size, depth_max=depth_max)

    # filter images by depth
    if filtering:
        filter_path = Path(save_path, 'SimFilter.json')
        with open(filter_path.as_posix()) as f:
            depth_filter = json.load(f)
        
        for filter_img in depth_filter.values():
            if filter_img['img_id'] == cam_id:
                mask_map = depth_map < filter_img['max_depth']
                depth_map = depth_map * mask_map
                color_map = color_map * mask_map[:, :, np.newaxis]

    # warp color map to cylindrical projection
    if warping2Cyl:
        cam_id = cam_id + '_warped'
        intrinsic_matrix = np.array([[intrinsic['fx'], 0, intrinsic['cx']],
                                    [0, intrinsic['fy'], intrinsic['cy']],
                                    [0, 0, 1]])

        color_map = cylindricalWarp(color_map, intrinsic_matrix)

    # Save depth map as an image
    if depth_map is not None:
        save_fPath = Path(save_path, cam_id + '.png')
        if not warping2Cyl:
            cv2.imwrite(save_fPath.as_posix(), depth_map)
        cv2.imwrite(save_fPath.as_posix().replace('.png', '_color.png'), cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))    # color does not show up yet

        if show:
            plot2Imgs(depth_map, color_map, title=cam_id)
            plotImg(mask_map, title=cam_id + '_mask')
            print("C2w (pix/gps):", cam_id, t_pix, "/", t_gps)
            print("C2w Rotation:", R.from_matrix(c2w[:3, :3]).as_euler('xyz', degrees=True))

    else: 
        print("No 3d points in the", cam_id)

def sample1ImgAnywhere_OpenCV(config, pcd : o3d, transformMatrices : dict, cam_id, distortion = None, point_size=1, depth_max=100, show=False):
    '''
    sample point cloud image from any location
    Args:
        config: Config() object
        pcd: Open3D point cloud object
        transformMatrices: array of camera pose(s)
        cam_id: camera ID as a string
        distortion: distortion coefficients
        point_size: point size
        depth_max: maximum depth
        show: show the image
    '''
    home_path = getHomePath()
    save_path = Path(home_path, Path(config.data_fileTransformMatrix).parent, 'pointCloud')
    if not save_path.exists():
        save_path.mkdir()

    points = {}
    points['XYZ'] = np.array(pcd.points)
    points['COLOR'] = np.array(pcd.colors)
    points['POINT3D_ID'] = np.arange(len(points['XYZ']))

    cam = transformMatrices[str(cam_id)]

    intrinsic = {
        'fx': cam['fx'],
        'fy': cam['fy'],
        'cx': cam['cx'],
        'cy': cam['cy'],
        'wid': cam['wid'],
        'hei': cam['hei'],
    }

    t_gps = [cam['transform_matrix'][0][3],
                cam['transform_matrix'][1][3],
                cam['transform_matrix'][2][3]]
    t_px = geoCoord2Open3Dpx(config, t_gps)
    
    c2w = np.array(cam['transform_matrix'])
    c2w[:3, 3] = t_px
    c2w = opengl_to_opencv(c2w)
    w2c = np.linalg.inv(c2w)

    depth_map, color_map, mask_map, used_points3d_cord = render_3d_world_to_camera_opencv(points, w2c, intrinsic, dist_coeffs=distortion, point_size=point_size, depth_max=depth_max)

    # Save depth map as an image
    img_id = cam['cam_id']
    if depth_map is not None:
        save_fPath = Path(save_path, f'{img_id}_{round(t_gps[0],6)}_{round(t_gps[1],6)}_{t_gps[2]}.png')
        cv2.imwrite(save_fPath.as_posix().replace('.png', '_color.png'), cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB))    # color does not show up yet

        if show:
            plotImg(color_map, title=img_id)

    else: 
        print("No 3d points in the", img_id)
    
    return depth_map, color_map, mask_map, used_points3d_cord

def sampleImgAnywhere_OpenCV(config, pcd : o3d, transformMatrices : dict, distortion = None, point_size=1, depth_max=100, show=False):
    print(f"Sampling with point size {point_size}, depth max {depth_max}, show {show}")

    # loop through all cameras and sample images
    for key, cam in tqdm(transformMatrices.items()):
        sample1ImgAnywhere_OpenCV(config, pcd, transformMatrices, key, distortion=distortion, point_size=point_size, depth_max=depth_max, show=show)


def point_cloud_to_panorama(points : dict, cameraLocation, intrinsics : np, v_res=0.42, h_res = 0.35, point_size=1,
                            v_fov = (-24.9, 2.0), h_fov = 270, h_middleA_south = 0, d_range = (0,100), y_offset=0, useIntrinsics=True):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (dict of np array)
            - points['XYZ'] is a Nx3 array of the point cloud locations
            - points['COLOR'] is a Nx3 array of the point cloud colors
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            vertical field of view in degrees (-min_negative_angle, max_positive_angle)
        h_fov: (float)
            horizontal field of view in degrees
        h_middleA_south: (float)
            horizontal middle angle of pano from south in math positive degrees
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_offset: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        depth_img: (np array), color_img: (np array)
    """
    # Adjust points relative to the camera location
    rel_points = points['XYZ'] - np.array(cameraLocation)
    x_points, y_points, z_points = rel_points[:, 0], rel_points[:, 1], rel_points[:, 2]
    color_points = points['COLOR']
    
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)

    # sort points by distance
    sorted_indices = np.argsort(d_points)
    descending_indices = sorted_indices[::-1]
    d_points = d_points[descending_indices]
    x_points = x_points[descending_indices]
    y_points = y_points[descending_indices]
    z_points = z_points[descending_indices]
    color_points = color_points[descending_indices]
    
    # Calculate resolution and FOV in radians
    v_res_rad = np.radians(v_res)
    h_res_rad = np.radians(h_res)
    
    # Calculate horizontal angles, adjust for h_northA, map x to cylinder
    angles = np.arctan2(y_points, x_points)  # Original angles in radians

    angles -= np.radians(h_middleA_south)  # Shift to align north with h_northA
    angles = (angles + np.pi) % (2 * np.pi) - np.pi  # Wrap angles to [-π, π]
    x_img = angles / h_res_rad

    # intrinsic are designed for high image resolution if resolution is lower it fails
    if useIntrinsics:  # Does not work
        x_img = intrinsics[0, 0] * angles + intrinsics[0, 2]  # Apply intrinsics --> north pixel calculation is off by a half an image --> explains that it started south
    # x_img = x_img     # fphi * phi + cx
    # y_img = y_img     # fy * Y/rho + cy

    # Define the horizontal FOV range centered around h_northA
    half_h_fov = np.radians(h_fov) / 2
    h_fov_min = -half_h_fov
    h_fov_max = half_h_fov
    
    # Calculate vertical angles, map y to cylinder
    y_angles = -np.arctan2(z_points, d_points)
    y_img = y_angles / v_res_rad
    if useIntrinsics:  # Does not work
        y_img = intrinsics[1, 1] * y_angles + intrinsics[1, 2]  # Apply intrinsics
        
    # Clip points within the specified FOVs
    valid_h = (h_fov_min <= angles) & (angles <= h_fov_max)
    valid_v = (np.radians(v_fov[0]) <= np.arctan2(z_points, d_points)) & (np.arctan2(z_points, d_points) <= np.radians(v_fov[1]))
    valid_indices = valid_h & valid_v
    
    x_img = x_img[valid_indices]
    y_img = y_img[valid_indices]
    d_points = d_points[valid_indices]
    color_points = color_points[valid_indices]
    
    # Convert to pixel coordinates
    # x_min = h_fov_min / h_res_rad
    # y_min = v_fov[0] / v_res - y_offset

    # Shift image coordinates to align within image dimensions
    #y_img = np.trunc(y_img - y_min).astype(np.int32)
    #x_img = np.trunc(-x_img - x_min).astype(np.int32)
    y_img = y_img.astype(np.int32)
    x_img = np.trunc(-x_img).astype(np.int32)

    img_width = int(np.ceil(h_fov / h_res))
    img_height = int(np.ceil((v_fov[1] - v_fov[0]) / v_res)) + int(y_offset)
    
    # Create image arrays
    depth_img = np.zeros([img_height, img_width], dtype=np.uint8)
    color_img = np.zeros([img_height, img_width, 3], dtype=np.uint8)
    
    # print("y_img", y_img.min(), y_img.max())
    # print("x_img", x_img.min(), x_img.max())

    # Loop through valid points and update images
    for idx, (y, x) in enumerate(zip(y_img, x_img)):
        depth = d_points[idx]
        color = color_points[idx]

        if depth > d_range[0] and depth < d_range[1]:
            # Update depth map with the minimum depth (in case of overlapping points)
            if not (np.any(depth_img[y-point_size//2:y+point_size//2, x-point_size//2:x+point_size//2] < depth) and 
                    depth == np.inf):
                depth_img[y, x] = depth
                color_img[y, x] = color
                # bigger point size
                fPoint_size = point_size
                if depth <= 5:
                    fPoint_size = point_size + 150
                if depth > 5 and depth <= 10:
                    fPoint_size = point_size + 80
                if depth > 10 and depth <= 25:
                    fPoint_size = point_size + 50
                if depth > 25 and depth <= 50:
                    fPoint_size = point_size + 20
                if depth > 50 and depth <= 100:
                    fPoint_size = point_size + 10
                if depth > 300 and depth <= 500:
                    fPoint_size = point_size - 5
                if depth > 500:
                    fPoint_size = point_size - 8
                if fPoint_size < 1 or point_size == 1:
                    fPoint_size = 1

                depth_img[y-fPoint_size//2:y+fPoint_size//2+1, x-fPoint_size//2:x+fPoint_size//2+1] = scale_to_255(depth, min=d_range[0], max=d_range[1])
                color_img[y-fPoint_size//2:y+fPoint_size//2+1, x-fPoint_size//2:x+fPoint_size//2+1,:] = (color * 255).astype(np.uint8)
   
    return depth_img, color_img

def samplePano1Cam_OpenCV(config, pcd : o3d, transformMatrices : dict, cam_id, distortion=None, point_size=5, depth_max=100):
    home_path = getHomePath()
    save_path = Path(home_path, Path(config.data_fileTransformMatrix).parent.parent, 'SimPano')
    if not save_path.exists():
        save_path.mkdir()

    # points to numpy array
    points = {}
    points['XYZ'] = np.array(pcd.points)
    points['COLOR'] = np.array(pcd.colors)
    for data_id in transformMatrices.keys():
        if cam_id in data_id:
            t_gps = [transformMatrices[cam_id][data_id][0]['transform_matrix'][0][3],
                        transformMatrices[cam_id][data_id][0]['transform_matrix'][1][3],
                        transformMatrices[cam_id][data_id][0]['transform_matrix'][2][3]]
            break
    intrinsics = np.array([[transformMatrices[cam_id]['fphi_pano'], 0, transformMatrices[cam_id]['cx_pano']],
                    [0, transformMatrices[cam_id]['fy_pano'], transformMatrices[cam_id]['cy_pano']],
                    [0, 0, 1]])
                    
    cameraLocation = geoCoord2Open3Dpx(config, t_gps)
    h_northA = (transformMatrices[cam_id]['pixel_north_pano']/transformMatrices[cam_id]['wid_pano']*transformMatrices[cam_id]['camera_angle_wid_pano']*180/np.pi)
    h_fov = transformMatrices[cam_id]['camera_angle_wid_pano']*180/np.pi
    v_angle = transformMatrices[cam_id]['camera_angle_hei']*180/np.pi
    tilt_angle = transformMatrices[cam_id]['tilt_angle']*180/np.pi
    v_fov = (-v_angle/2 + tilt_angle, v_angle/2 + tilt_angle)
    h_middleA_south = 180 - (h_fov/2 - h_northA)    # north is off by 180 degree, middle angle of pano: 180 - (fov_middle - north_angle), south=0, east=90, north=180, west=270
    #h_middleA_south = - (h_fov/2 - h_northA)
    # same resolution than panorama --> very sparse
    h_res = h_fov/transformMatrices[cam_id]['wid_pano']
    v_res = v_angle/transformMatrices[cam_id]['hei']
    # h_res = 0.2
    # v_res = 0.2
    
    depth_pano, color_pano = point_cloud_to_panorama(points, cameraLocation, intrinsics, point_size=point_size, v_res = v_res, h_res = h_res, 
                            v_fov = v_fov, h_fov = h_fov, h_middleA_south = h_middleA_south, d_range = (0,depth_max), useIntrinsics=True)
    
    # Save depth map as an image
    save_fPath = Path(save_path, f'{cam_id}.png')
    cv2.imwrite(save_fPath.as_posix(), cv2.cvtColor(depth_pano, cv2.COLOR_BGR2RGB))
    cv2.imwrite(save_fPath.as_posix().replace('.png', '_color.png'), cv2.cvtColor(color_pano, cv2.COLOR_BGR2RGB))


def samplePanos_OpenCV(config, pcd : o3d, transformMatrices : dict, distortion=None, point_size=5, depth_max=100):
    for cam in tqdm(transformMatrices.values()):
        samplePano1Cam_OpenCV(config, pcd, transformMatrices, cam['cam_id'], distortion=distortion, point_size=point_size, depth_max=depth_max)


def sample1MeshAnywhere(config, mesh : trimesh, transformMatrices : dict, cam_id, distortion = None, rgbd=False, show=False, gps_coord=True, img_filter=True, save_result=True):
    '''
    sample point cloud image from any location
    Args:
        config: Config() object
        mesh: Trimesh object
        transformMatrices: array of camera pose(s)
        cam_id: camera ID as a string
        distortion: distortion coefficients, TODO: not yet implemented
        show: show the image
    '''
    cam = transformMatrices[str(cam_id)]
    intrinsic = {
        'fx': cam['fx'],
        'fy': cam['fy'],
        'cx': cam['cx'],
        'cy': cam['cy'],
        'wid': cam['wid'],
        'hei': cam['hei'],
    }
    
    c2w = np.array(cam['transform_matrix'])

    if gps_coord:
        t_gps = c2w[:3, 3].tolist()
        t_px = geoCoord2Open3Dpx(config, t_gps)
        
        c2w[:3, 3] = np.array(t_px)

    save_path = Path(config.data_pathScene)

    sampleImg_TriMesh(save_path, mesh, c2w, intrinsic, cam_id, depth_inversion=True, rgbd=rgbd, show=show, filter=img_filter, save_result=save_result)

def sampleMeshAnywhere(config, mesh, transformMatrices, distortion = None, rgbd=False, show=False, gps_coord=True, img_filter=True, save_result=True):
    print(f"Sampling with from mesh, show {show}")

    # loop through all cameras and sample images
    for key, cam in tqdm(transformMatrices.items()):
        sample1MeshAnywhere(config, mesh, transformMatrices, key, distortion=distortion, rgbd=rgbd, show=show, gps_coord=gps_coord, img_filter=img_filter, save_result=save_result)

def sample_MeshRandomly(config, mesh : trimesh, num_samples, save_folder, resolution=[768, 512], aoi=None, seed=None, rgbd=False, show=False):
    '''
    Sample random locations of a mesh and save generated images and transformation matrices
    Input:
        mesh: Trimesh object
        num_samples: number of samples
        show: show the image
    '''
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    # sample random points
    np.random.seed(seed)
    if aoi is not None:
        smallMesh = cutMeshOnAoi(config, mesh, aoi)
        random_points = smallMesh.sample(num_samples)
    else:
        random_points = mesh.sample(num_samples)

    # add additional random height, TODO: random, points, e.g. 2/3 at surface, 1/3 somewhere above (airplane)
    random_points[:, 2] = random_points[:, 2] + np.random.uniform(0.1, 10, num_samples)

    # intrinsic parameters
    intrinsics = {'fx': resolution[0]/5*3, 'fy': resolution[1], 'cx': resolution[0]/2, 'cy': resolution[1]/2, 'wid': resolution[0], 'hei': resolution[1]}

    tMatrices = {}

    # loop through all random points and sample images
    for i, point in tqdm(enumerate(random_points)):
        # build c2w transformation matrix
        #tilt = np.random.uniform(-10*np.pi/180, 10*np.pi/180)
        tilt = 0
        rot_horiz = np.random.uniform(0, 2*np.pi)       # TODO: height dependent tilt
        rotation_angles = [tilt, rot_horiz]
        c2w = getCamTransform_np(point, rotation_angles, rotation_axis='xy')
        rot = c2w[:3, :3]
        rot = init_to_opengl(rot)
        c2w[:3, :3] = rot
        save_path = Path(Path(config.data_fileTransformMatrix).parent, save_folder)
   
        # sample mesh
        sampleImg_TriMesh(save_path, mesh, c2w, intrinsics, img_id=str(i), rgbd=rgbd, show=show)
        #sample_TriMesh_direct(save_path, mesh, c2w, intrinsics, img_id=str(i), show=show)

        # append transformation matrix
        t_pix = c2w[:3, 3].tolist()
        t_gps = Open3Dpx2geoCoord(config, t_pix)
        c2w[:3, 3] = np.array(t_gps)
        tMatrices[str(i)] = {
            '_comment': f"{t_gps}",
            'cam_id': f"{save_folder}_{i}",
            'hei': intrinsics['hei'],
            'wid': intrinsics['wid'],
            'cx': intrinsics['cx'],
            'cy': intrinsics['cy'],
            'fx': intrinsics['fx'],
            'fy': intrinsics['fy'],
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
            'transform_matrix': c2w.tolist()
        }

    # save transformation matrix
    saveTMat_path = Path(save_path.parent, f'transformMatrixRand_{save_folder}.json')
    with open(saveTMat_path.as_posix(), 'w') as f:
        json.dump(tMatrices, f, indent=2)

def cutMeshOnAoi(config, mesh : trimesh, aoi : list, gps=True):
    '''
    Cut a mesh on a given area of interest
    Input:
        mesh: Trimesh object
        aoi: Area of interest as a list [lonMin, lonMax, latMin, latMax]
    Return:
        cut_mesh: Cut mesh as Trimesh object
    '''
    min_lon, max_lon = aoi[0], aoi[1]
    min_lat, max_lat = aoi[2], aoi[3]

    # convert gps to pixel
    if gps:
        min_pix = geoCoord2Open3Dpx(config, [min_lon, min_lat, 0])
        max_pix = geoCoord2Open3Dpx(config, [max_lon, max_lat, 0])
        max_x, min_y = min_pix[0], min_pix[1]
        min_x, max_y = max_pix[0], max_pix[1]

    # programm it in the style of the two lines above
    box = trimesh.creation.box(bounds=[[min_x, min_y, 0], [max_x, max_y, config.GEE_dem_max_m/config.GEE_pxScale]])
    cut_mesh = trimesh.intersections.slice_mesh_plane(mesh, -box.facets_normal, box.facets_origin)

    # add color to the cut mesh
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        tree = cKDTree(mesh.vertices)
        _, indices = tree.query(cut_mesh.vertices)
        cut_mesh.visual.vertex_colors = mesh.visual.vertex_colors[indices]

    return cut_mesh

def cutPCDOnAoi(config, pcd : o3d, aoi : list, gps=True):
    '''
    Cut a point cloud to a given area of interest
    Input:
        pcd: Open3D point cloud object
        aoi: Area of interest as a list [lonMin, lonMax, latMin, latMax]
    Return:
        cut_pcd: Cut point cloud as Open3D point cloud object
    '''
    min_lon, max_lon = aoi[0], aoi[1]
    min_lat, max_lat = aoi[2], aoi[3]

    # convert gps to pixel
    if gps:
        min_pix = geoCoord2Open3Dpx(config, [min_lon, min_lat, 0])
        max_pix = geoCoord2Open3Dpx(config, [max_lon, max_lat, 0])
        max_x, min_y = min_pix[0], min_pix[1]
        min_x, max_y = max_pix[0], max_pix[1]

    # get the vertices of the point cloud
    vertices = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)

    # get the indices of the vertices inside the area of interest
    indices = np.where((vertices[:, 0] > min_x) & (vertices[:, 0] < max_x) & (vertices[:, 1] > min_y) & (vertices[:, 1] < max_y))[0]
    # get the vertices inside the area of interest
    cut_vertices = vertices[indices]
    cut_color = color[indices]
    cut_normals = normals[indices]

    # create the cut point cloud
    cut_pcd = o3d.geometry.PointCloud()
    cut_pcd.points = o3d.utility.Vector3dVector(cut_vertices)
    cut_pcd.colors = o3d.utility.Vector3dVector(cut_color)
    cut_pcd.normals = o3d.utility.Vector3dVector(cut_normals)

    return cut_pcd

def sampleMeshOfAoi(config, mesh : trimesh, num_samples, save_folder, aoi, resolution=[768, 512], rgbd=False, show=False):
    '''
    Sample random locations of a mesh and save generated images and transformation matrices
    Input:
        config: Config() object
        mesh: Trimesh object
        num_samples: number of samples
        save_folder: folder to save the images
        aoi: Area of interest as a list [lonMin, lonMax, latMin, latMax]
        show: show the image
    '''
    num_final_samples = 1
    center_gps = [(aoi[0] + aoi[1])/2, (aoi[2] + aoi[3])/2, 0]
    print("Center GPS:", center_gps)
    center = geoCoord2Open3Dpx(config, center_gps)
    vertices = np.array(mesh.vertices)
    for vertex in vertices:
        if vertex[0] == center[0] and vertex[1] == center[1]:
            center[2] = vertex[2]
            break
    aoi_min = [aoi[0], aoi[2], 0]
    aoi_max = [aoi[1], aoi[3], 0]
    aoi_min_pix = geoCoord2Open3Dpx(config, aoi_min)[:2]
    aoi_max_pix = geoCoord2Open3Dpx(config, aoi_max)[:2]
    distance = [aoi_min_pix[0] - aoi_max_pix[0], aoi_max_pix[1] - aoi_min_pix[1], 0]

    tMatrices = {}

    while num_final_samples < num_samples:
        print(f"Sampling {num_final_samples}/{num_samples}", end = "\r")
        # sample random points
        smallMesh = cutMeshOnAoi(config, mesh, aoi)
        random_points = smallMesh.sample(int(num_samples/10 + 1))

        # filter out points that are +-1/6 around the center
        for point in random_points:
            if (point[0] > center[0] - distance[0]/6 and point[0] < center[0] + distance[0]/6 and 
                    point[1] > center[1] - distance[1]/6 and point[1] < center[1] + distance[1]/6):
                random_points = np.delete(random_points, np.where((random_points == point).all(axis=1)), axis=0)
                print("Debug: removed point", point, "center", center, "distance", distance)

        # variable height
        dist2c = np.linalg.norm(random_points[:, :2]-center[:2], axis=1)
        maxOff_height = (5500/config.GEE_pxScale - random_points[:, 2])/(distance[1]/2)*dist2c + random_points[:, 2]     # max height: (5500m-height_c)/distance/2 * dist2c + height_c
        maxOff_height = np.clip(maxOff_height, 1/config.GEE_pxScale, 5500/config.GEE_pxScale - random_points[:, 2])
        random_points[:, 2] = random_points[:, 2] + np.random.uniform(1/config.GEE_pxScale, maxOff_height, random_points.shape[0])

        # intrinsic parameters
        intrinsics = {'fx': resolution[0]/5*3, 'fy': resolution[1], 'cx': resolution[0]/2, 'cy': resolution[1]/2, 'wid': resolution[0], 'hei': resolution[1]}

        # loop through all random points and sample images
        for i, point in enumerate(random_points):
            # build c2w transformation matrix
            distance2center = np.linalg.norm(point[:2]-center[:2])
            distance2center_norm = distance2center / np.sqrt((distance[0]/2)**2 + (distance[1]/2)**2)
            if distance2center_norm < 0.01:
                distance2center_norm = 0.01
            tilt2center = np.arctan2(2*(point[2]-center[2]), distance2center)
            tilt = np.random.uniform(tilt2center-5*np.pi/180, tilt2center+5*np.pi/180)
            deltax = point[0] - center[0]
            deltay = point[1] - center[1]
            rot2center = np.arctan2(deltay, deltax)
            rot_fac = 20 / distance2center_norm
            rot_fac = np.clip(rot_fac, 0, 55)
            rot_horiz = np.random.uniform(rot2center-rot_fac*np.pi/180, rot2center+rot_fac*np.pi/180)
            rotation_angles = [tilt, rot_horiz * 180/np.pi]
            c2w = getCamTransform_np(np.array(point), rotation_angles, rotation_axis='xy')
            rot = c2w[:3, :3]
            rot = init_to_opengl(rot)
            c2w[:3, :3] = rot
            save_path = Path(Path(config.data_fileTransformMatrix).parent, save_folder)
    
            # sample mesh
            if (sampleImg_TriMesh(save_path, mesh, c2w, intrinsics, img_id=str(num_final_samples), rgbd=rgbd, show=show)):
                num_final_samples += 1

                # append transformation matrix
                t_pix = c2w[:3, 3].tolist()
                t_gps = Open3Dpx2geoCoord(config, t_pix)
                c2w[:3, 3] = np.array(t_gps)
                tMatrices[str(num_final_samples-1)] = {
                    '_comment': f"{t_gps}",
                    'cam_id': f"{save_folder}_{num_final_samples-1}",
                    'file_path': save_path,
                    'hei': intrinsics['hei'],
                    'wid': intrinsics['wid'],
                    'cx': intrinsics['cx'],
                    'cy': intrinsics['cy'],
                    'fx': intrinsics['fx'],
                    'fy': intrinsics['fy'],
                    'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
                    'p1': 0, 'p2': 0,
                    'transform_matrix': c2w.tolist()
                }

    # save transformation matrix
    saveTMat_path = Path(save_path.parent, f'transformMatrixRand_{save_folder}.json')
    with open(saveTMat_path.as_posix(), 'w') as f:
        json.dump(tMatrices, f, indent=2)

def visualize_ply(title, path):
    mesh = trimesh.load(path)
    vertex_colors = mesh.visual.vertex_colors.astype(np.float32)
    vertex_colors /= 255.0
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                hoverinfo="none",
                flatshading=True,
                vertexcolor=vertex_colors,
                opacity=1.0,
                lighting=dict(
                    ambient=1.0, diffuse=0.8, specular=1, roughness=0.4, fresnel=0.2
                ),
                lightposition=dict(x=64, y=64, z=0),
            )
        ],
        layout=dict(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    eye=dict(x=-1.5, y=0, z=0.75),
                    up=dict(x=0, y=1, z=0),
                ),
            ),
            margin=dict(t=40, b=10, l=0, r=0),
        ),
    )

    fig.show()