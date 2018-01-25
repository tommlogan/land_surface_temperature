'''
This tool creates the land surface temperature map based on land satellite images

Input:
        tif file of
                landsat
                land cover
Output:
        land surface temperature (oC)
        albedo
        ndvi

Notes:
        It is essential that the two raster files have the same:
                - projection
                - raster size
                - cell size
        To ensure this, use the landcover image as the unmodified raster.
        1) Project the others into the same projection as the lc
        2) resample the other raster. using the resample tool. go to "environments" tab
                and pick the landuse raster as the "snap raster", under processing extent.
        3) clip the larger raster by the smaller one (you may need to do this multiple times)
                until they have the exact same cell size.
'''

from PIL import Image
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import shapefile
from matplotlib import pyplot as plt
from rpy2.robjects.packages import importr
import subprocess
import os
os.chdir('F:/UrbanDataProject/land_surface_temperature')

# init logging
import sys
sys.path.append("code")
from logger_config import *
logger = logging.getLogger(__name__)

def main():
    '''
        loops through satellite images and processes them
    '''
    # Read the source csv with the satellite id numbers of the
    source_city = pd.read_csv('data/data_source_city.csv')
    source_satellite = pd.read_csv('data/data_source_satellite.csv')
    info_satellite = source_satellite.iloc[0]
    # loop rows
    for index, info_satellite in source_satellite.iterrows():
        # process the image
        logger.info('Processing image {}'.format(info_satellite['landsat_product_id']))
        process_image(info_satellite, source_city)


def process_image(info_satellite, source_city):
    '''
        1. Reads in metadata
        2. Creates map of land surface temperature
        3. Creates map of NVDI
        4. Creates map of albedo
    '''
    # read metadata
    meta_dict = read_metadata(info_satellite)

    # clip the images to the city and ensure they are same projection
    clip_geographic_data(info_satellite, source_city)

    # create map of land surface temperature
    calc_LST(info_satellite, meta_dict, source_city)

    # create nvdi map
    calc_NDVI(info_satellite)

    # create map of albedo
    calc_albedo(info_satellite, meta_dict)


def read_metadata(info_satellite):
    '''
        For the image read the metadata txt file
        Return
            Dictionary of metadata
    '''
    logger.info('Reading metadata')
    # metadata file and location
    fn_metadata = 'data/raw/{}/{}_MTL.txt'.format(info_satellite['city'],info_satellite['landsat_product_id'])

    # list of variables needed from metadata
    meta_variables = set(['K1_CONSTANT_BAND_10','K2_CONSTANT_BAND_10'])
    bands = [1,2,3,4,5,10]
    # include radiance re-scaling factors
    for rad in ['MULT', 'ADD']:
        for b in bands:
            meta_variables.add('RADIANCE_{}_BAND_{}'.format(rad, b))

    # init dictionary
    meta_dict = dict.fromkeys(meta_variables)
    # open the metadata file
    with open(fn_metadata,'r') as fid:
        # loop lines in text file
        for line in fid:
            # divide line into words, split by space
            elements = line.split()
            # intersect with metavariables to return set of variable name if it is on line
            var = meta_variables.intersection(elements)
            if var: # if not empty set
                # extract the variable
                var = list(var)[0]
                # add to dictionary
                meta_dict[var] = float(elements[-1])

    return meta_dict


def clip_geographic_data(info_satellite, source_city):
    '''
        Run R code which clips in the satellite and land cover images to the city boundary
        and projects to WGS84
        Return
            images saved in /data/intermediate
            satellite
            land cover
    '''
    logger.info('Clipping geographic data if necessary')
    # Define command
    command = 'Rscript'
    path2script = 'code/processing/clip_geographic_data.R'

    # Define arguments
    city = info_satellite['city']
    city_idx = source_city.loc[source_city['city']==city].index
    # filename arguments
    landsat_product_id = info_satellite['landsat_product_id']
    fn_land_cover = source_city['land_cover'][city_idx].values[0]
    fn_boundary = source_city['city_parcels'][city_idx].values[0]
    bands = '1,2,3,4,5,10'
    # args into list
    args_clip = [city, landsat_product_id, fn_land_cover, fn_boundary, bands]

    # Build subprocess command
    cmd = [command, path2script] + args_clip

    # run the R function to clip and project the data as required
    x = subprocess.check_output(cmd, universal_newlines=True)


def calc_LST(info_satellite, meta_dict, source_city):
    '''
        Returns the land surface temperature (LST) based on satellite imagery
    '''
    logger.info('Starting LST calculations')

    # metadata file and location
    city = info_satellite['city']
    fn_b10 = 'data/intermediate/{}/{}_B10.tif'.format(info_satellite['city'], info_satellite['landsat_product_id'])

    # read in band 10 data
    image_b10 = gdal.Open(fn_b10)
    dn = image_b10.ReadAsArray()

    # conversion to TOA radiance
    TOA = calc_TOA(dn, meta_dict, 10)

    # emissivity correction
    emissivity = determine_emissivity(info_satellite, dn, source_city)

    # calculate the at-satellite brightness temperature
    temp_satellite = calc_satellite_temperature(TOA, meta_dict, emissivity)

    # atmospheric correction
    temp_surface = atmos_correction(temp_satellite, info_satellite, emissivity)

    # write to tif
    fn_out = 'data/processed/image/{}/{}_{}.tif'.format(city, 'lst', info_satellite['date'])
    array_to_raster(temp_surface, fn_out, image_b10)


def calc_TOA(dn, meta_dict, band_number):
    '''
        Calculate the Top Atmosphere Spectral Radiance (TOAr) from Band 10 digital
        number (DN) data (also refered to as the Q_cal - quantized and
        calibrated standard product pixel value)
    '''
    logger.info('Calculating TOAr')

    TOAr = meta_dict['RADIANCE_MULT_BAND_{}'.format(band_number)] * dn + meta_dict['RADIANCE_ADD_BAND_{}'.format(band_number)]

    return(TOAr)


def determine_emissivity(info_satellite, dn, source_city):
    '''
        Emissivity is determined by the land cover
        Return
            Emissivity array from land cover map
    '''
    logger.info('Determining emissivity map')

    city = info_satellite['city']

    # filename for land cover
    city_idx = source_city.loc[source_city['city']==info_satellite['city']].index
    landcover_id = source_city['land_cover'][city_idx].values[0]
    fn_landcover = '_'.join(landcover_id.split('_',2)[:2] + [city])
    fn_land_cover = 'data/intermediate/{}/{}.tif'.format(city, fn_landcover)
    print(fn_land_cover)

    # import
    land_cover = gdal.Open(fn_land_cover)

    # convert to array
    land_cover = land_cover.ReadAsArray()
    logger.info("LC tif size: " + str(np.shape(land_cover)))
    land_cover = land_cover.astype(float)
    emissivity = land_cover.copy()

    #convert land_cover to emissivity array
    emissivity[land_cover > 90] = 0.957 # Wetlands is assumed to be Sparse Vegetation
    emissivity[(1 < land_cover) & (land_cover < 20)] = 0.989 # Water
    emissivity[(1 < land_cover) & (land_cover < 30)] = 0.912 # Urban
    emissivity[(1 < land_cover) & (land_cover < 40)]  = 0.896 # Barren
    emissivity[(1 < land_cover) & (land_cover < 50)]  = 0.967 # Forest
    emissivity[(1 < land_cover) & (land_cover < 80)]  = 0.957 # Grass
    emissivity[(1 < land_cover) & (land_cover < 90)]  = 0.957 # Cropland

    return(emissivity)


def calc_satellite_temperature(TOA, meta_dict, emissivity):
    '''
        Calculate the At-Satellite Brightness temperature
        First, need to calculate the emissivity from the land use
        Then convert
        Returns temperature in Kelvin
    '''
    logger.info('calculating satellite temperature')
    # import code
    # code.interact(local = locals())
    # spectral radiance
    L_lambda = TOA/emissivity

    # calculate the satellite brightness
    temp_satellite = meta_dict['K2_CONSTANT_BAND_10']/(np.log((meta_dict['K1_CONSTANT_BAND_10']/L_lambda) + 1))

    return temp_satellite


def atmos_correction(temp_satellite, info_satellite, emissivity):
    '''
        Using the mono-window algorithm (Qin et al., 2001, International Journal of Remote Sensing)
        make the atmospheric correction
        Returns land surface temperature in celsius
    '''
    logger.info('making the atmospheric correction')

    # temparature from csv file
    temp_max = info_satellite['max_temp_celsius']

    # constants for the algorithm
    a_6 = -67.355351
    b_6 = 0.458606
    w = 1.6
    t_6 = 0.974290 - 0.08007*w
    # variables dependent on land cover (emissivity) and temperature
    c_6 = emissivity * t_6
    d_6 = (1 - t_6)*(1 + (1 - emissivity)*t_6)
    t_a = 16.0110 + 0.92621*temp_max

    # mono-window algorithm
    T = a_6*(1 - c_6 - d_6) + (b_6*(1 - c_6 - d_6) + c_6 + d_6)*temp_satellite - d_6*t_a
    temp_landsurface = T/c_6

    # converting to celsius
    temp_landsurface = temp_landsurface - 273.15

    return temp_landsurface


def calc_albedo(info_satellite, meta_dict):
    '''
        Calculate albedo from bands 1-5
        This is calculated using Smith's normalized Liang el al. algorithm
        Reference:
            Smith, R. B., 2010: The heat budget of the earth’s surface deduced from space. Tech. rep., Yale, http://www.yale.edu/ceo/Documentation/Landsat DN to Albedo.pdf.)
            Liang, S., 2001: Narrowband to broadband conversions of land surface albedo i: Algorithms. Re-mote Sensing of Environment, 76 (2), 213 – 238, doi:http://dx.doi.org/10.1016/S0034-4257(00)00205-4, URL http://www.sciencedirect.com/science/article/pii/S0034425700002054.
        Return
            albedo
    '''
    logger.info('Calculating the albedo')

    # calculating the reflectivity of each band
    reflect_band = dict()
    band = 1
    for band in [1,2,3,4,5]:
        fn_sat = 'data/intermediate/{}/{}_B{}.tif'.format(info_satellite['city'], info_satellite['landsat_product_id'], band)
        ds = gdal.Open(fn_sat)
        dn = ds.ReadAsArray()
        reflect_band[band] = calc_TOA(dn, meta_dict, band)

    # calculate the albedo
    albedo = ((0.356*reflect_band[1]) + (0.130*reflect_band[2]) +
            (0.373*reflect_band[3]) + (0.085*reflect_band[4]) +
            (0.072*reflect_band[5]) - 0.018) / 1.016

    # save
    fn_out = 'data/processed/image/{}/{}_{}.tif'.format(info_satellite['city'], 'albedo', info_satellite['date'])
    array_to_raster(albedo, fn_out, ds)


def calc_NDVI(info_satellite):
    '''
        calculate the NDVI
        For landsat8 it's (B5 – B4) / (B5 + B4) - see Ben's email dated 7/26/16
    '''
    # import bands
    landsat_band = dict()
    band = 1
    for band in [1,2,3,4,5]:
        fn_sat = 'data/intermediate/{}/{}_B{}.tif'.format(info_satellite['city'], info_satellite['landsat_product_id'], band)
        ds = gdal.Open(fn_sat)
        landsat_band[band] = ds.ReadAsArray()

    # calculate the NDVI
    ndvi = (landsat_band[5] - landsat_band[4])/(landsat_band[5] + landsat_band[4])

    # save NVDI
    fn_out = 'data/processed/image/{}/{}_{}.tif'.format(info_satellite['city'], 'ndvi', info_satellite['date'])
    array_to_raster(ndvi, fn_out, ds)


def array_to_raster(output, out_filename, ds):
    '''
        Convert the array back to a raster
        Save the raster
    '''
    logger.info('Calculating the albedo')

    # identify the number of pixels
    x_pixels = output.shape[1]
    y_pixels = output.shape[0]

    # create the output image
    driver = gdal.GetDriverByName('GTiff')
    # print driver
    dataset = driver.Create(
        out_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32)

    # write the data
    dataset.GetRasterBand(1).WriteArray(output)
    # Write to disk
    dataset.FlushCache()

    # georeference the image and set the projection
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())


if __name__ == '__main__':
    main()
