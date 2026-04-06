import geemap as ee
import numpy as np
import math
from tqdm import tqdm

from dataVisu_engine.GEE_helpers import *
from utils.config import *


GEECONFIG = {
    'save_path' : 'data/EarthEngine/Scale5',
    'mesh_path' : 'data/EarthEngine/mesh_new.ply',
    'dem_min_m' : 0,   # DEM height model min in meters
    'dem_max_m' : 4500,    # DEM height model max in meters
    'dem_save_mode': 'I;16',    # I;16 (16bit raw), L (8bit to 0-255)
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
    'pxScale' : 5,  # One pixel in meters (more or less at least)
    'color_correction' : False,
    'aoi' : [[7.6, 46.4], [8.35, 46.4], [8.35, 46.85], [7.6, 46.85]],  # [[7.880222, 46.53806], [8.023, 46.53806], [8.023, 46.681889], [7.880222, 46.681889]]
    'aoi_pixel': [[0, 0], [0, 0], [0, 0], [0, 0]],
}
MAX_PIXEL_PER_REQUEST = 2000

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='seasonScapes-master')    # create an own Google Earth Engine project

# configuration
config = Config()
config.setGEE(GEECONFIG)


# process and save models in splitted up steps:
# TODO: I think the problem is that more north the same amount of longitude degree are less distance and thus less pixels!
lat_diff = abs(config.GEE_aoi[2][1] - config.GEE_aoi[0][1])
lon_diff = abs(config.GEE_aoi[2][0] - config.GEE_aoi[0][0])
distance_lat_km = 111.32 * lat_diff  # 1 degree latitude = 111.32 km
distance_lon_km = 40075.0 * math.cos(lat_diff / 2 * np.pi / 180) / 360 * (lon_diff)  # 1 degree longitude = 40075 km * cos(latitude) / 360
nbr_pixels_lat = int(1000*distance_lat_km / config.GEE_pxScale)
nbr_pixels_lon = int(1000*distance_lon_km / config.GEE_pxScale)
nbr_patches_lat = math.ceil(nbr_pixels_lat / MAX_PIXEL_PER_REQUEST)
nbr_patches_lon = math.ceil(nbr_pixels_lon / MAX_PIXEL_PER_REQUEST)
lat_per_patch = lat_diff / nbr_patches_lat
long_per_patch = lon_diff / nbr_patches_lon
aoi_full = config.GEE_aoi
long_base = aoi_full[0][0]
lat_base = aoi_full[0][1]

# loop over patches of the area of interest
img_land_list = []
img_dem_list = []
for i in tqdm(range(nbr_patches_lat)):
    for j in range(nbr_patches_lon):
        aoi = [[long_base + j*long_per_patch, lat_base + i*lat_per_patch],
                [long_base + (j+1)*long_per_patch, lat_base + i*lat_per_patch],
                [long_base + (j+1)*long_per_patch, lat_base + (i+1)*lat_per_patch],
                [long_base + j*long_per_patch, lat_base + (i+1)*lat_per_patch]]
        config.setGEE_aoi(aoi)

        _, land_np = processLAND_model(config.GEE)
        _, dem_np, _ = processDEM_model(config.GEE)

        img_land_list.append(land_np)
        img_dem_list.append(dem_np)
        
config.loadCamMetadata(config.data_pathMetadata)

# Resize all land images to the same size, dito dem images -> does not work for dem!!!
#for i in range(len(img_land_list)):
#    img_land_list[i], _ = normalizeResize(img_land_list[i], img_land_list[0])
#for i in range(len(img_dem_list)):
#    img_dem_list[i], _ = normalizeResize(img_dem_list[i], img_dem_list[0], normalize=False)

# Combine all images to one big image
for row in range(nbr_patches_lat):
    for col in range(nbr_patches_lon):
        if col == 0:
            land_row = img_land_list[row*nbr_patches_lon]
            dem_row = img_dem_list[row*nbr_patches_lon]
        else:
            # pad shape if not same size
            shape_diff_dem = img_dem_list[row*nbr_patches_lon + col].shape[0] - dem_row.shape[0]
            if shape_diff_dem > 0:
                dem_row_last = dem_row[-1, :]
                dem_row = np.vstack((dem_row, np.tile(dem_row_last, (shape_diff_dem, 1))))
            elif shape_diff_dem < 0:
                dem_row_last = img_dem_list[row*nbr_patches_lon + col][-1, :]
                img_dem_list[row*nbr_patches_lon + col] = np.vstack((img_dem_list[row*nbr_patches_lon + col], np.tile(dem_row_last, (-shape_diff_dem, 1))))
            shape_diff_land = img_land_list[row*nbr_patches_lon + col].shape[0] - land_row.shape[0]
            if shape_diff_land > 0:
                land_row_last = land_row[-1, :, :]
                land_row = np.vstack((land_row, np.tile(land_row_last, (shape_diff_land, 1, 1))))
            elif shape_diff_land < 0:
                land_row_last = img_land_list[row*nbr_patches_lon + col][-1, :, :]
                img_land_list[row*nbr_patches_lon + col] = np.vstack((img_land_list[row*nbr_patches_lon + col], np.tile(land_row_last, (-shape_diff_land, 1, 1))))

            land_row = np.hstack((land_row, img_land_list[row*nbr_patches_lon + col]))
            dem_row = np.hstack((dem_row, img_dem_list[row*nbr_patches_lon + col]))
    if row == 0:
        landFull_np = land_row
        demFull_np = dem_row
    else:
        # pad shape if not same size
        shape_diff_land = landFull_np.shape[1] - land_row.shape[1]
        if shape_diff_land > 0:
            land_col_last = land_row[:, -1, :]
            land_col_last = np.expand_dims(land_col_last, axis=1)
            land_row = np.hstack((land_row, np.tile(land_col_last, (1, shape_diff_land, 1))))
        elif shape_diff_land < 0:
            land_col_last = landFull_np[:, -1 :]
            land_col_last = np.expand_dims(land_col_last, axis=1)
            landFull_np = np.hstack(landFull_np, np.tile(land_col_last, (1, -shape_diff_land, 1)))
        shape_diff_dem = demFull_np.shape[1] - dem_row.shape[1]
        if shape_diff_dem > 0:
            dem_col_last = dem_row[:, -1]
            dem_row = np.hstack(dem_row, np.tile(dem_col_last, (1, shape_diff_dem)))
        elif shape_diff_dem < 0:
            dem_col_last = demFull_np[:, -1]
            demFull_np = np.hstack(demFull_np, np.tile(dem_col_last, (1, -shape_diff_dem)))

        landFull_np = np.vstack((land_row, landFull_np))
        demFull_np = np.vstack((dem_row, demFull_np))

landFull_np, _ = normalizeResize(landFull_np, demFull_np)

# reset aoi full and save config
config.setGEE_aoi(aoi_full)
calcAOI_pixel(config, landFull_np)  # based on combined image
saveLAND_model(config, landFull_np)
saveDEM_model(config, demFull_np)    
config.saveGEE()
    
