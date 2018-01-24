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


city = 'la'
file_code = 'LC80410362016174LGN00'
T_0 = 302.039 #max temp. observed in city (https://www.wunderground.com/history/?MR=1)
M_L = 3.342e-4 # Band-10 specific multiplicative rescaling factor from the metadata
A_L = 0.10000 # Band-10 specific additive rescaling factor from the metadata
K_1 = 774.8853 # Band-specific thermal conversion constant from the metadata
K_2 =  1321.0789 # Band-specific thermal conversion constant from the metadata
filename = file_code + '_B10_proj.tif'
out_filename = {'lst' : city + '_LST.tif', 'nvdi' : city + '_NVDI.tif', 'albedo' : city + '_albedo.tif'}
lc_filename = city + '_LC.tif'
b1 = file_code + '_B1.tif'
b2 = file_code + '_B2.tif'
b3 = file_code + '_B3.tif'
b4 = file_code + '_B4.tif'
b5 = file_code + '_B5.tif'
set1_5 = [b1, b2, b3, b4, b5]

def main():
    '''
        loops through satellite images and processes them
    '''
    # Read the source csv with the satellite id numbers of the
    source_satellite = pd.read_csv('data/data_source_satellite.csv')
    info_satellite = source_satellite.iloc[0]
    # loop rows
    for index, info_satellite in source_satellite.iterrows():
        # process the image
        logger.info('Processing image {}'.format(info_satellite['landsat_product_id']))
        process_image(info_satellite)


def process_image(info_satellite):
    '''
        1. Reads in metadata
        2. Creates map of land surface temperature
        3. Creates map of NVDI
        4. Creates map of albedo
    '''
    # read metadata
    meta_dict = read_metadata(info_satellite)

    # Create map of land surface temperature
    calc_LST(info_satellite, meta_dict)

    # create nvdi map
    calc_NDVI(set1_5)

    # create map of albedo
    albedo(set1_5)


def read_metadata(info_satellite):
    '''
        For the image read the metadata txt file
        Return
            Dictionary of metadata
    '''
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


def calc_LST(info_satellite, meta_dict):
    '''
        Returns the land surface temperature (LST) based on satellite imagery
    '''
    logger.info('Starting LST calculations')

    # metadata file and location
    fn_b10 = 'data/raw/{}/{}_B10.tif'.format(info_satellite['city'],info_satellite['landsat_product_id'])

    # read images and prepare for calculations

    # read in band 10 data
    image_b10 = gdal.Open(fn_b10)
    dn = image_b10.ReadAsArray()

    # print("L8 tif size: " + str(np.shape(dn)))

    # Conversion to TOA Radiance
    TOA = calc_TOA(dn, meta_dict, 10)

    # Emissivity Correction
    emissivity = determine_emissivity(info_satellite, dn)

    # Calculate the At-Satellite Brightness Temperature
    temp_satellite = calc_satellite_temperature(TOA, meta_dict, emissivity)

    # Atmospheric correction
    temp_surface = atmos_correction(temp_satellite, T_0,lc)

    # write to tif
    array_to_raster(temp_surface, out_filename['lst'], land_cover)


def read_geographic_data(info_satellite):
    '''
        Run R code which reads in the satellite and land cover images, crop to the city boundary
        Then read in the results
        Return
            satellite image
            land cover
    '''
    # Define command
    command = 'Rscript'
    path2script = 'code/processing/clip_geographic_data.R'

    # Define arguments
    city = info_satellite['city']
    landsat_product_id = info_satellite['landsat_product_id']
    fn_land_cover = source_city['land_cover'][city_idx].values[0]
    fn_boundary = source_city['city_parcels'][city_idx].values[0]
    args_clip = [city, landsat_product_id, fn_land_cover, fn_boundary]

    # Build subprocess command
    cmd = [command, path2script] + args_clip

    # run the R function to clip and project the data as required
    x = subprocess.check_output(cmd, universal_newlines=True)


    # import source file
    source_city = pd.read_csv('data/data_source_city.csv')

    # filename for satellite image
    fn_b10 = 'data/raw/{}/{}_B10.tif'.format(info_satellite['city'],info_satellite['landsat_product_id'])

    # filename for land cover
    city_idx = source_city.loc[source_city['city']==city].index
    fn_land_cover = source_city['land_cover'][city_idx].values[0]
    fn_land_cover = 'data/raw/{}/{}/{}.tif'.format(city, fn_land_cover, fn_land_cover)

    # filename for boundary
    fn_boundary = source_city['city_parcels'][city_idx].values[0]
    # fn_boundary = 'data/raw/{}/{}/{}.shp'.format(city, fn_boundary, fn_boundary)


    # import R functions
    import rpy2.robjects as ro
    rgdal = importr('rgdal')

    # import boundary
    city_boundary = rgdal.readOGR(dsn = 'data/raw/{}/{}'.format(city,fn_boundary), layer = fn_boundary)
    city_boundary <- gUnaryUnion(city_boundary)
    # read in band 10 data
    image_b10 = gdal.Open(fn_b10)

    # import land cover
    land_cover = gdal.Open(fn_land_cover)






    dn = image_b10.ReadAsArray()



def calc_TOA(dn, meta_dict, band_number):
    '''
        Calculate the Top Atmosphere Spectral Radiance (TOAr) from Band 10 digital
        number (DN) data (also refered to as the Q_cal - quantized and
        calibrated standard product pixel value)
    '''
    logger.info('Calculating TOAr')

    TOAr = meta_dict['RADIANCE_MULT_BAND_{}'.format(band_number)] * dn + meta_dict['RADIANCE_ADD_BAND_{}'.format(band_number)]

    return(TOAr)


def determine_emissivity(info_satellite, dn):
    '''
        Emissivity is determined by the land cover
        Return
            Emissivity array from land cover map
    '''
    logger.info('Determining emissivity map')

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


def calc_satellite_temperature(TOA, meta_dict):
    '''
        Calculate the At-Satellite Brightness temperature
        First, need to calculate the emissivity from the land use
        Then convert
        Returns temperature in Kelvin
    '''
    logger.info('calculating satellite temperature')

    # spectral radiance
    L_lambda = TOA/emissivity

    # calculate the satellite brightness
    temp_satellite = meta_dict['K2_CONSTANT_BAND_10']/(np.log((meta_dict['K1_CONSTANT_BAND_10']/L_lambda) + 1))
    return temp_satellite


def atmos_correction(T_sensor, T_0, emissivity):
    '''
        Using the mono-window algorithm (Qin et al., 2001, International Journal of Remote Sensing)
        make the atmospheric correction
        Returns land surface temperature in celsius
    '''
    logger.info('making the atmospheric correction')

    # constants for the algorithm
    a_6 = -67.355351
    b_6 = 0.458606
    w = 1.6
    t_6 = 0.974290 - 0.08007*w
    # variables dependent on land cover (emissivity) and temperature
    c_6 = emissivity * t_6
    d_6 = (1 - t_6)*(1 + (1 - emissivity)*t_6)
    t_a = 16.0110 + 0.92621*T_0

    # mono-window algorithm
    T = a_6*(1 - c_6 - d_6) + (b_6*(1 - c_6 - d_6) + c_6 + d_6)*T_sensor - d_6*t_a
    temp_landsurface = T/c_6

    # converting to celsius
    temp_landsurface = temp_landsurface - 273.15

    return temp_landsurface


def calc_albedo(set1_5):
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
    for name in set1_5:
        ds = gdal.Open(name)
        dn = ds.ReadAsArray()
        reflect_band[i] = calc_TOA(dn, meta_dict, band_number)
        i += 1

    # calculate the albedo
    albedo = ((0.356*reflect_band[0]) + (0.130*reflect_band[1]) +
            (0.373*reflect_band[2]) + (0.085*reflect_band[3]) +
            (0.072*reflect_band[4]) - 0.018) / 1.016

    array_to_raster(albedo, out_filename['albedo'], ds)


def calc_NDVI(set1_5):
    '''
        calculate the NDVI
        For landsat8 it's (B5 – B4) / (B5 + B4) - see Ben's email dated 7/26/16
    '''
    # import bands
    landsat_band = dict()
    band_number = 1
    for name in set1_5:
        ds = gdal.Open(name)
        landsat_band[band_number] = ds.ReadAsArray()
        band_number += 1

    # calculate the NDVI
    ndvi = (landsat_band[5] - landsat_band[4])/(landsat_band[5] + landsat_band[4])

    array_to_raster(ndvi, out_filename['nvdi'], ds)


def array_to_raster(output, out_filename, ds):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    # np.savetxt("foo.csv", output, delimiter=",")
    print(output.shape)


    # You need to get those values like you did.
    x_pixels = output.shape[1]
    # assert x_pixels == ds.RasterXSize  # number of pixels in x
    y_pixels = output.shape[0]
    # assert y_pixels == ds.RasterYSize  # number of pixels in y
    print(x_pixels,y_pixels)

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
    dataset.FlushCache()  # Write to disk.

    # georeference the image and set the projection
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())



if __name__ == '__main__':
    main()
