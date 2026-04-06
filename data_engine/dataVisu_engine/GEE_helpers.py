import geemap
import numpy as np
import matplotlib.pyplot as plt
import ee
from skimage.transform import resize
from PIL import Image
from pathlib import Path

from utils.general import getHomePath
from dataVisu_engine.texture import *
from utils.config import *

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='seasonScapes-master')    # create an own Google Earth Engine project


######################################## Helper functions ########################################
def maskS2clouds(image):
    """
    Mask clouds in Sentinel-2 image using the QA band
    Args:
        image (ee.Image): Sentinel-2 image
    Returns:
        ee.Image: Cloud masked Sentinel-2 image
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloudBitMask).eq(0)
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    )

    return image.updateMask(mask).divide(10000)

def apply_scale_factors(image):
  """
  Apply scale factors to Landsat
  """
  optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
  thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
  return image.addBands(optical_bands, None, True).addBands(
      thermal_bands, None, True
  )

def processDEM_model(config, plot=False):
    '''
    Get DEM model and process it
    Input:
        config: GEE dictionary
    Returns:
        ee.Image: DEM model
        numpy.array: DEM image
        ee.Terrain: slope model
    '''
    if config['dem_model'] == 'SRTM90m':
        elevation = ee.Image('CGIAR/SRTM90_V4').select('elevation')
    elif config['dem_model'] == 'ALOS30m':
        elevationCol = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
        elevation = elevationCol.mean()
    elif config['dem_model'] == 'NASA30m':
        elevation = ee.Image('NASA/NASADEM_HGT/001').select('elevation')    # USGS/SRTMGL1_003
    slope = ee.Terrain.slope(elevation)

    region = ee.Geometry.Polygon([config['aoi']], None, False)
    ##corrected_scale = config['pxScale'] / np.cos(np.radians(config['aoi'][0][1]))
    #dem_img_np = geemap.ee_to_numpy(elevation, region=region, scale=corrected_scale)
    dem_img_np = geemap.ee_to_numpy(elevation, region=region, scale=config['pxScale'])

    if plot:        # plot DEM
        print(dem_img_np.shape, "min/max:", np.min(dem_img_np), np.max(dem_img_np))
        plt.imshow(dem_img_np)
        plt.show()

    return elevation, dem_img_np, slope

def saveDEM_model(config, dem_img_np):
    '''
    Save the DEM model
    Input:
        config: Config() object
        numpy.array: DEM image
    '''
    print("Save dem model", dem_img_np.shape, "min/max:", np.min(dem_img_np), np.max(dem_img_np))
    if config.GEE_dem_save_mode == 'I;16':
        dem_img = Image.fromarray(dem_img_np, mode='I;16')
    elif config.GEE_dem_save_mode == 'L':      # only for visualization --> not properly scaled
        print("Warning: DEM image is not properly scaled, use only for visualization")
        dem_img = (255*((dem_img_np - config.GEE_dem_min_m)/(config.GEE_dem_max_m - config.GEE_dem_min_m))).astype(np.uint8)
        if np.max(dem_img) > 255:
            print("Warning: DEM image values are larger than 255, will be clipped to 255")
            dem_img = np.clip(dem_img, 0, 255)
        dem_img = dem_img.squeeze()
        dem_img = Image.fromarray(dem_img, mode='L')
    gee_hash = str(config.genGEEHash())[0:5]
    save_path = Path(getHomePath(), Path(config.GEE_save_path).parent, Path(config.GEE_save_path).stem + "Dem" + config.GEE_dem_model + "_" + gee_hash + ".png")
    dem_img.save(save_path.as_posix())

def processLAND_model(config, plot=False):
    '''
    Get LAND model from Earth Engine as numpy array
    Input:
        config: GEE dictionary
    Returns:
        ee.ImageCollection: LAND model
        numpy.array: processed LAND model
    '''
    region = ee.Geometry.Polygon(config['aoi'], None, False)
    #corrected_scale = int(config['pxScale'] / np.cos(np.radians(config['aoi'][0][1])))
    #if corrected_scale % 2 != 0 and corrected_scale % 5 != 0:
    #    corrected_scale = corrected_scale - corrected_scale % 2
    if config['land_model'] == 'Landsat30m':
        land = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate('2023-07-01', '2023-11-11')
            .filter(ee.Filter.lt('CLOUD_COVER', config['land_cloud_fill_max']))
            .filter(ee.Filter.gt('SUN_ELEVATION', config['land_sun_elev']))
        )
        land = land.map(apply_scale_factors)
        land_img = ee.Image(land.mean()).select(['SR_B4', 'SR_B3', 'SR_B2'])
        land_img_np = geemap.ee_to_numpy(land_img, region=region, scale=config['pxScale'])
    elif config['land_model'] == 'Sentinel10m':
        land = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate('2024-01-01', '2024-10-12')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', config['land_cloud_fill_max']))
            .map(maskS2clouds)
        )
        land_img = ee.Image(land.mean()).select(['B4', 'B3', 'B2'])
        land_img_np = geemap.ee_to_numpy(land_img, region=region, scale=config['pxScale'])
    elif config['land_model'] == 'SkySat80cm':
        land = ee.ImageCollection('SKYSAT/GEN-A/PUBLIC/ORTHO/RGB').select(['R', 'G', 'B'])
        land_img = ee.Image(land.mean())
        land_img_np = geemap.ee_to_numpy(land_img, region=region, scale=config['pxScale'])
    elif config['land_model'] == 'swissT10cm':
        land_img = ee.Image('Switzerland/SWISSIMAGE/orthos/10cm/2018').select(['R', 'G', 'B'])
        land_img2 = ee.Image('Switzerland/SWISSIMAGE/orthos/10cm/2017').select(['R', 'G', 'B'])
        land = ee.ImageCollection([land_img, land_img2])
        #land_img_np = geemap.ee_to_numpy(land_img, region=region, scale=corrected_scale)
        #land_img2_np = geemap.ee_to_numpy(land_img2, region=region, scale=corrected_scale)
        land_img_np = geemap.ee_to_numpy(land_img, region=region, scale=config['pxScale'])
        land_img2_np = geemap.ee_to_numpy(land_img2, region=region, scale=config['pxScale'])
        land_img_np = land_img_np + land_img2_np # add both images no overlaps

    if plot:    # plot LAND
        print(land_img_np.shape, "min/max:", np.min(land_img_np), np.max(land_img_np))
        plt.imshow(land_img_np)
        plt.show()

    return land, land_img_np

def saveLAND_model(config, land_img_np):
    '''
    Save the Land model
    Input:
        config: Config() object
        land_img_np: Land image as numpy array
    '''
    land_img_np *= 255
    print("Save land model", land_img_np.shape, "min/max:", np.min(land_img_np), np.max(land_img_np))
    gee_hash = str(config.genGEEHash())[0:5]
    save_path = Path(getHomePath(), Path(config.GEE_save_path).parent, Path(config.GEE_save_path).stem + "Land" + config.GEE_land_model + "_" + gee_hash + ".png")
    land_img = Image.fromarray(land_img_np.astype(np.uint8))
    land_img.save(save_path.as_posix())

def normalizeResize(image1, image2, custom_cut=False, normalize=True):
    '''
    Normalize img1 based on img2 and img1 own channels
    '''
    img1 = image1.copy()
    img2 = image2.copy()
    if normalize:
        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))

    # Ensure is 2D by squeezing the last dimension
    img2 = img2.squeeze()
    img1_shape_old = img1.shape
    if custom_cut:
        img1 = img1[3:img1.shape[0]-2, :img1.shape[1]-4, :]
    img1 = resize(img1, (img2.shape[0], img2.shape[1], img1.shape[2]), anti_aliasing=True, preserve_range=True)
    print("Resized img1 from", img1_shape_old, "to", img1.shape)

    return img1, img2

def showMap(config, center, landImg, demImg, landVisu=None, demVisu=None):
    '''
    Show the map
    Input:
        config: Configuration dictionary
        center: np.array([lat, lon, zoom])
        land: Land geemap.Image
        dem: DEM geemap.Image
        landVisu: Land visualization settings
        demVisu: DEM visualization settings
    '''
    m = geemap.Map()
    m.set_center(center[0], center[1], center[2])
    if config['land_model'] == 'Landsat_30m':
        m.addLayer(landImg.mean(), landVisu, 'True Color (432)')
    elif config['land_model'] == 'Sentinel_10m':
        m.addLayer(landImg.mean(), landVisu, 'RGB')
    elif config['land_model'] == 'SkySat_80cm':
        m.addLayer(landImg.mean(), landVisu, 'RGB')
    elif config['land_model'] == 'swissTopo_10cm':
        m.addLayer(landImg, None, 'swissTopo')
    m.addLayer(demImg, demVisu, 'elevation')
    m

def calcAOI_pixel(config, land_np):
    '''
    Calculate the AOI pixel coordinates
    Input:
        Config() object: Configuration
        numpy.array: Land image
    '''
    # Calculate the pixel coordinates of the AOI
    aoi_pixel = config.GEE_aoi_pixel
    aoi_pixel[0][0] = 0
    aoi_pixel[0][1] = 0
    aoi_pixel[1][0] = land_np.shape[1]
    aoi_pixel[1][1] = 0
    aoi_pixel[2][0] = land_np.shape[1]
    aoi_pixel[2][1] = land_np.shape[0]
    aoi_pixel[3][0] = 0
    aoi_pixel[3][1] = land_np.shape[0]
    config.setGEE_aoi_pixel(aoi_pixel)
    print("AOI pixel coordinates:", aoi_pixel, "\nAOI Geo coordinates:", config.GEE_aoi)
