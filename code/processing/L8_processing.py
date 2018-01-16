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
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import os
import logging
logger = logging.getLogger(__name__)

os.chdir('F:/UrbanDataProject/land_surface_temperature/data/cities/losangeles/2016-06-22 LC80410362016174LGN00')

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



def create_heatmap(filename,out_filename):
    ''' returns the land surface temperatue based on satellite imagery
    '''

    print("You're operating in " + os.getcwd())
    # open the satellite imagery
    ds = gdal.Open(filename)
    dn = ds.ReadAsArray()
    print("L8 tif size: " + str(np.shape(dn)))

    # Step 1: Conversion to TOA Radiance
    TOA = step1(dn)

    # Step 2: Emissivity Correction
    #returns an emissivity map from land cove
    land_cover = gdal.Open(lc_filename)
    lc = land_cover.ReadAsArray()
    print("LC tif size: " + str(np.shape(lc)))
    emissiv = get_emissivity(lc)

    L_lambda = TOA/emissiv

    #Next Step 3 in ppt
    T_sensor = step3(L_lambda)

    temp_surface = step4(T_sensor, T_0,lc)
    temp_surface = celsius(temp_surface)
    # write to  tif
    output = temp_surface
    array_to_raster(output, out_filename['lst'], land_cover)


def step1(Q_cal):
    ''' band 10 DN data can be converted to TOA spectral radiance '''
    # First get Constants  -note: automate this step later (text parsing)

    L_lambda = M_L * Q_cal + A_L # TOA spectral radiance
    return(L_lambda)


def get_emissivity(land_cover):
    land_cover = land_cover.astype(float)

    #convert land_cover to emissivity array
    land_cover[land_cover > 90] = 0.957 # Wetlands is assumed to be Sparse Vegetation
    land_cover[(1 < land_cover) & (land_cover < 20)] = 0.989 # Water
    land_cover[(1 < land_cover) & (land_cover < 30)] = 0.912 # Urban
    land_cover[(1 < land_cover) & (land_cover < 40)]  = 0.896 # Barren
    land_cover[(1 < land_cover) & (land_cover < 50)]  = 0.967 # Forest
    land_cover[(1 < land_cover) & (land_cover < 80)]  = 0.957 # Grass
    land_cover[(1 < land_cover) & (land_cover < 90)]  = 0.957 # Cropland

    return(land_cover)


def step3(L_lambda):

    T = K_2/(np.log((K_1/L_lambda) + 1))
    return T


def step4(T_sensor, T_0,lc):
    a_6 = -67.355351
    b_6 = 0.458606
    w = 1.6
    t_6 = 0.974290 - 0.08007*w
    c_6 = get_emissivity(lc) * t_6
    d_6 = (1 - t_6)*(1 + (1 - get_emissivity(lc))*t_6)
    t_a = 16.0110 + 0.92621*T_0
    T = a_6*(1 - c_6 - d_6) + (b_6*(1 - c_6 - d_6) + c_6 + d_6)*T_sensor - d_6*t_a
    T = T/c_6
    return T


def celsius(T):
    return (T-273.15)


def reflectance(dn):
    mult_band = 2.0e-5
    add_band = -0.1
    refl = mult_band * dn + add_band
    return refl


def albedo(set1_5):
    reflect_band = dict()
    i = 0
    for name in set1_5:
        ds = gdal.Open(name)
        reflect_band[i] = reflectance(ds.ReadAsArray())
        i += 1

    alpha = ((0.356*reflect_band[0]) + (0.130*reflect_band[1]) +
            (0.373*reflect_band[2]) + (0.085*reflect_band[3]) +
            (0.072*reflect_band[4]) - 0.018) / 1.016

    array_to_raster(alpha, out_filename['albedo'], ds)


def calc_NDVI(set1_5):
    ''' calculate the NDVI '''
    # import bands
    landsat_band = dict()
    i = 0
    for name in set1_5:
        ds = gdal.Open(name)
        landsat_band[i] = ds.ReadAsArray()
        i += 1

    NDVI = (landsat_band[4] - landsat_band[3])/(landsat_band[4] + landsat_band[3])

    array_to_raster(NDVI, out_filename['nvdi'], ds)


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
    create_heatmap(filename, out_filename)
    albedo(set1_5)
    calc_NDVI(set1_5)
