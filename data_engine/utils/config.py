import pandas as pd
from pathlib import Path
import json

from .general import getHomePath

class Config:
    def __init__(self):
        self._GEE = {
            'save_path' : 'data/EarthEngine/gee.png',
            'mesh_path' : 'data/EarthEngine/mesh-8512.ply',
            'hash' : 0, # hash of config
            'dem_min_m' : 0,   # DEM height model min in meters
            'dem_max_m' : 4500,    # DEM height model max in meters
            'dem_save_mode': 'L',    # I;16 (16bit raw), L (8bit to 0-255)
            'dem_opacity' : 0.8,   # DEM height model opacity
            'land_min' : 0.0,  # LAND threshold min
            'land_max' : 0.6,  # LAND threshold max
            'land_min_ui' : 0, # LAND threshold min for uint8
            'land_max_ui' : 190, # LAND threshold max for uint8
            'land_gamma' : 1.2,    # LAND gamma correction
            'land_cloud_fill_max' : 10, # CLOUD fill max
            'land_sun_elev' : 30,
            'dem_model' : 'NASA30m',  # SRTM90m, ALOS30m, NASA30m
            'land_model' : 'swissT10cm',   # Landsat30m, Sentinel10m, SkySat80cm, swissT10cm
            'pxScale' : 60,  # One pixel in meters --> should be used for height as well
            'color_correction' : False,
            'aoi' : [[7.6, 46.4], [8.35, 46.4], [8.35, 46.85], [7.6, 46.85]], # all Bernese highland, [long, lat]
            'aoi_pixel': [[0, 0], [1600, 0], [1600, 1003], [0, 1003]],      # [x, y]
        }
        self._dataCam = {
            'pathMetadata' : 'configs/Livecams-Masterarbeit-Timo-Kleger.csv',
            'CamOrigMetadata': None,
            'pathScene' : 'data/2024-10-09/12-00-00/',
            'fileTransformMatrix' : 'data/transformation_matrices/transformation_2learn_v3.json',
            'hash' : 0,
            # camera intrinsics for my camera splitting
            'nbr_frames': 6,
            'overlapPix_wid_img': 40,
            'degPerPix_wid_img': 0.05625,
            'camera_angle_wid_img': 1.05,  # FOV in rad
            'camera_angle_hei': 0.79,  # FOV in rad
            'cx': 500,
            'cy': 500,
            'wid': 3072,  # image width 3:2 aspect ratio
            'hei': 2048,
            'origin_point': self._GEE['aoi'],
        }
        self._SatCloud = {
            'name': 'KachSuperHD',
            'satelliteMeta_path': 'data/EarthEngine/satelliteMeta',
            'hash': 0,
            'cloudThresh': 100,
            'cropLon_min': 127, 
            'cropLon_max': 673, 
            'cropLat_min': 79, 
            'cropLat_max': 561,
            'delta_vTemp': -19/33,
            'null_vTemp': 55,
            'delta_tempH': -9000/60,
            'null_tempH': 2500,
            'avg_cloudHeight': 1000,
            }
        self._sd = {
            "name": "stable diffusion v1.5",
            "sd_model_path": "./controlNet_engine/models/cldm_v15.yaml",
            "control_model_path": "./controlNet_engine/models/control_sd15_depthT.ckpt",
            "a_prompt": "best quality, extremely detailed",
            "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
            "num_samples": 1,
            "image_resolution": 512,
            "detect_resolution": 384,
            "ddim_steps": 51,
            "guess_mode": False,
            "strength": 1.0,
            "scale": 4.9,
            "seed": 1797625314,
            "eta": 0,
            "calcDepth": False,
            "control_img_folder": "./data/NovelViews/meshAoi",
            "save_folder": "./data/2024-10-09/12-00-00/imageSD_1SDepth",
            "transformMatrix_path": "./data/NovelViews/transformMatrix_meshAoi.json"
        }
        self.home_path = getHomePath()

############ setter functions ############
    def setGEE(self, GEE):
        for key in GEE:
            self._GEE[key] = GEE[key]

    def setData(self, data):
        for key in data:
            self._dataCam[key] = data[key]
        
    def setSatCloud(self, SatCloud):
        for key in SatCloud:
            self._SatCloud[key] = SatCloud[key]

    def setGEE_aoi(self, aoi):
        self._GEE['aoi'] = aoi

    def setGEE_aoi_pixel(self, aoi_pixel):
        self._GEE['aoi_pixel'] = aoi_pixel

    def setGEE_mesh_path(self, save_mesh_path):
        self._GEE['mesh_path'] = str(save_mesh_path)

    def setData_pathMetadata(self, pathMetadata):
        self._dataCam['pathMetadata'] = pathMetadata

    def setData_pathScene(self, pathScene):
        self._dataCam['pathScene'] = pathScene

    def setData_CamOrigmetadata(self, metadata):
        ''' metadata: pandas frame from csv file '''
        metadata_tmp = {}
        for row in range(metadata.shape[0]):
            meta4cam_id = {}
            meta4cam_id['camGeneration'] = metadata['Livecam generation'][row]
            meta4cam_id['Download Link'] = metadata['Download Link'][row]
            meta4cam_id['Download base'] = metadata['Download base'][row]
            meta4cam_id['f[mm]'] = metadata['Brennweite, mm'][row]
            meta4cam_id['sensor_hei[mm]'] = metadata['Sensorhoehe, mm'][row]
            meta4cam_id['sensor_wid[mm]'] = metadata['Sensorbreite, mm'][row]
            meta4cam_id['FOV_hei[°]'] = metadata['Field of view (FOV) height, °'][row]
            meta4cam_id['FOV_wid[°]'] = metadata['Field of view (FOV) width, °'][row]
            meta4cam_id['latitude[°N]_cor'] = metadata['GPS lat, °N, WGS84'][row] + metadata['GPS, °N offset'][row]
            meta4cam_id['longitude[°E]_cor'] = metadata['GPS long, °E'][row] + metadata['GPS, °E offset'][row]
            meta4cam_id['altitude[m]_cor'] = metadata['Altitude (camera), m'][row] + metadata['Altitude offset'][row]
            meta4cam_id['origPixel_hei'] = metadata['original pixel h'][row]
            meta4cam_id['origPixel_wid'] = metadata['original pixel w'][row]
            meta4cam_id['pixel_hei'] = metadata['vertikal pixel'][row]
            meta4cam_id['pixel_wid'] = metadata['horizontal pixel tot'][row]
            meta4cam_id['pixel_wid_perImg'] = metadata['horizontal pixel'][row]
            meta4cam_id['TotFOV_wid[°]'] = metadata['Winkel'][row]
            meta4cam_id['angle_north'] = metadata['Nord Grad'][row]
            meta4cam_id['pixel_north_cor'] = metadata['Nord Pixel'][row] + metadata['Nord Pixel offset'][row]
            meta4cam_id['TiltAngle[°]_cor'] = metadata['Tilt Grad'][row] + metadata['Tilt Grad offset'][row]
            meta4cam_id['delta_center'] = metadata['delta center'][row]

            metadata_tmp[metadata['ID'][row]] = meta4cam_id
        self._dataCam['CamOrigMetadata'] = metadata_tmp


    ############ helper functions ############
    def genGEEHash(self):
        geeWoPath = self._GEE.copy()
        save_path = geeWoPath.pop('save_path')
        mesh_path = geeWoPath.pop('mesh_path')
        hashNbr = hash(str(geeWoPath))
        self._GEE['hash'] = hashNbr
        mesh_path = Path(mesh_path)
        mesh_path = Path(mesh_path.parent, 'mesh_' + str(hashNbr)[:5] + '.ply')
        self._GEE['save_path'] = save_path
        self._GEE['mesh_path'] = str(mesh_path)
        return hashNbr
    
    def genSatCloudHash(self):
        satCloudWoName = self._SatCloud.copy()
        satCloudWoName.pop('name')
        satCloudWoName.pop('satelliteMeta_path')
        satCloudWoName.pop('hash')
        hashNbr = hash(str(satCloudWoName))
        self._SatCloud['hash'] = hashNbr
        return hashNbr
    
    def genCamHash(self):
        camWoPath = self._dataCam.copy()
        camWoPath.pop('pathMetadata')
        camWoPath.pop('CamOrigMetadata')
        camWoPath.pop('pathScene')
        camWoPath.pop('hash')
        hashNbr = hash(str(camWoPath))
        self._dataCam['hash'] = hashNbr
        return hashNbr

    def saveGEE(self, same_hash=False):
        save_path = Path(self.home_path, 'configs', 'EarthEngine')
        save_path.mkdir(parents=True, exist_ok=True)
        mesh_name = Path(self._GEE['mesh_path']).name
        mesh_name = str(mesh_name).replace('mesh', '')
        mesh_name = mesh_name.replace('.ply', '')
        if same_hash == False:
            hashNbr = self.genGEEHash()
            self._GEE['hash'] = hashNbr
            save_fName = Path('Scale' + str(self._GEE['pxScale']) + '_' + str(hashNbr)[:5] + str(mesh_name)[5:] + '.json')
        else:
            save_fName = Path('Scale' + str(self._GEE['pxScale']) + str(mesh_name) + '.json')
        self._GEE['save_fName'] = str(save_fName)
        save_path = Path(save_path, save_fName)
        with open(save_path.as_posix(), 'w') as f:
            json.dump(self._GEE, f, indent=2)
        print(f'Saved GEE config to {save_path}')

    def loadGEE(self, path):
        path = Path(self.home_path, path).as_posix()
        with open(path, 'r') as f:
            config = json.load(f)
        self.setGEE(config)

    def saveSatCloud(self):
        save_path = Path(self.home_path, 'configs')
        save_path.mkdir(parents=True, exist_ok=True)
        hashNbr = self.genSatCloudHash()
        self._SatCloud['hash'] = hashNbr
        save_fName = 'Cloud_' + self._SatCloud['name'] + '_' + str(hashNbr)[:5] + '.json'
        save_path = Path(save_path, save_fName)
        with open(save_path.as_posix(), 'w') as f:
            json.dump(self._SatCloud, f, indent=2)
        print(f'Saved SatCloud config to {save_path}')

    def loadSatCloud(self, path='configs/SatCloud_Kachelmann_superHD.json'):
        path = Path(self.home_path, path).as_posix()
        with open(path, 'r') as f:
            config = json.load(f)
        self._SatCloud = config

    def loadCamMetadata(self, pathMetadata, home_path=''):
        '''
            Load camera metadata from csv file
            Args:
                pathMetadata (str): path to camera metadata
            Returns:
                pandas.DataFrame: camera metadata
        '''
        pathMetadata = Path(self.home_path, pathMetadata).as_posix()
        camMetadata = pd.read_csv(pathMetadata, sep=';')

        print("Length metadata:", len(camMetadata))

        self.setData_CamOrigmetadata(camMetadata)

    def saveCam(self):
        save_path = Path(self.home_path, 'configs')
        save_path.mkdir(parents=True, exist_ok=True)
        hashNbr = self.genCamHash()
        save_fName = 'Cam_' + str(hashNbr)[:5] + '.json'
        save_path = Path(save_path, save_fName)
        with open(save_path.as_posix(), 'w') as f:
            json.dump(self._dataCam, f, indent=2)
        print(f'Saved camera config to {save_path}')

    def loadCam(self, pathCam):
        pathCam = Path(self.home_path, pathCam).as_posix()
        with open(pathCam, 'r') as f:
            config = json.load(f)

        self._dataCam = config

        # load camera metadata
        pathMetadata = Path(self.home_path, config['pathMetadata'])
        camMetadata = pd.read_csv(pathMetadata.as_posix(), sep=';')

        self.setData_CamOrigmetadata(camMetadata)

    def loadSD(self, path):
        path = Path(self.home_path, path).as_posix()
        with open(path, 'r') as f:
            config = json.load(f)
        self._sd = config


    ############ access variables ############
    @property
    def GEE(self):
        return self._GEE
    @property
    def dataCam(self):
        return self._dataCam
    @property
    def GEE_save_path(self):
        return self._GEE['save_path']
    @property
    def GEE_mesh_path(self):
        return self._GEE['mesh_path']
    @property
    def GEE_hash(self):
        return self._GEE['hash']
    @property
    def GEE_dem_min_m(self):
        return self._GEE['dem_min_m']
    @property
    def GEE_dem_max_m(self):
        return self._GEE['dem_max_m']
    @property
    def GEE_dem_save_mode(self):
        return self._GEE['dem_save_mode']
    @property
    def GEE_dem_opacity(self):
        return self._GEE['dem_opacity']
    @property
    def GEE_land_min(self):
        return self._GEE['land_min']
    @property
    def GEE_land_max(self):
        return self._GEE['land_max']
    @property
    def GEE_land_min_ui(self):
        return self._GEE['land_min_ui']
    @property
    def GEE_land_max_ui(self):
        return self._GEE['land_max_ui']
    @property
    def GEE_land_gamma(self):
        return self._GEE['land_gamma']
    @property
    def GEE_land_cloud_fill_max(self):
        return self._GEE['land_cloud_fill_max']
    @property
    def GEE_land_sun_elev(self):
        return self._GEE['land_sun_elev']
    @property
    def GEE_dem_model(self):
        return self._GEE['dem_model']
    @property
    def GEE_land_model(self):
        return self._GEE['land_model']
    @property
    def GEE_pxScale(self):
        return self._GEE['pxScale']
    @property
    def GEE_color_correction(self):
        return self._GEE['color_correction']
    @property
    def GEE_aoi(self):
        return self._GEE['aoi']
    @property
    def GEE_aoi_pixel(self):
        return self._GEE['aoi_pixel']
    @property
    def data_pathMetadata(self):
        return self._dataCam['pathMetadata']
    @property
    def data_camOrigMetadata(self):
        return self._dataCam['CamOrigMetadata']
    @property
    def data_pathScene(self):
        return self._dataCam['pathScene']
    @property
    def data_fileTransformMatrix(self):
        return self._dataCam['fileTransformMatrix']
    @property
    def data_cx(self):
        return self._dataCam['cx']
    @property
    def data_cy(self):
        return self._dataCam['cy']
    @property
    def data_w(self):
        return self._dataCam['wid']
    @property
    def data_h(self):
        return self._dataCam['hei']
    @property
    def data_origin_point(self):
        return self._dataCam['origin_point']
    @property
    def SatCloud(self):
        return self._SatCloud
    @property
    def SatCloud_name(self):
        return self._SatCloud['name']
    @property
    def SatCloud_satelliteMeta_path(self):
        return self._SatCloud['satelliteMeta_path']
    @property
    def SatCloud_cloudThresh(self):
        return self._SatCloud['cloudThresh']
    @property
    def SatCloud_cropLon_min(self):
        return self._SatCloud['cropLon_min']
    @property
    def SatCloud_cropLon_max(self):
        return self._SatCloud['cropLon_max']
    @property
    def SatCloud_cropLat_min(self):
        return self._SatCloud['cropLat_min']
    @property
    def SatCloud_cropLat_max(self):
        return self._SatCloud['cropLat_max']
    @property
    def SatCloud_delta_vTemp(self):
        return self._SatCloud['delta_vTemp']
    @property
    def SatCloud_null_vTemp(self):
        return self._SatCloud['null_vTemp']
    @property
    def SatCloud_delta_tempH(self):
        return self._SatCloud['delta_tempH']
    @property
    def SatCloud_null_tempH(self):
        return self._SatCloud['null_tempH']
    @property
    def SatCloud_avg_cloudHeight(self):
        return self._SatCloud['avg_cloudHeight']
    @property
    def sd(self):
        return self._sd
    @property
    def sd_name(self):
        return self._sd['name']
    @property
    def sd_sd_model_path(self):
        return self._sd['sd_model_path']
    @property
    def sd_control_model_path(self):
        return self._sd['control_model_path']
    @property
    def sd_a_prompt(self):
        return self._sd['a_prompt']
    @property
    def sd_n_prompt(self):
        return self._sd['n_prompt']
    @property
    def sd_num_samples(self):
        return self._sd['num_samples']
    @property
    def sd_image_resolution(self):
        return self._sd['image_resolution']
    @property
    def sd_detect_resolution(self):
        return self._sd['detect_resolution']
    @property
    def sd_ddim_steps(self):
        return self._sd['ddim_steps']
    @property
    def sd_guess_mode(self):
        return self._sd['guess_mode']
    @property
    def sd_strength(self):
        return self._sd['strength']
    @property
    def sd_scale(self):
        return self._sd['scale']
    @property
    def sd_seed(self):
        return self._sd['seed']
    @property
    def sd_eta(self):
        return self._sd['eta']
    @property
    def sd_calcDepth(self):
        return self._sd['calcDepth']
    @property
    def sd_control_img_folder(self):
        return self._sd['control_img_folder']
    @property
    def sd_save_folder(self):
        return self._sd['save_folder']
    @property
    def sd_transformMatrix_path(self):
        return self._sd['transformMatrix_path']
    
    
