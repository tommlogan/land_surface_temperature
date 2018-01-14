import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

def write_tif(array,lat,lon,filename):

  # For each pixel I know it's latitude and longitude.
  # As you'll see below you only really need the coordinates of
  # one corner, and the resolution of the file.

  array = np.flipud(array)
  lat = np.flipud(lat)
  lon = np.flipud(lon)

  xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
  nrows,ncols = np.shape(array)
  xres = (xmax-xmin)/float(ncols)
  yres = (ymax-ymin)/float(nrows)
  geotransform=(xmin,xres,0,ymax,0, -yres)   
  # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
  #         top left y, rotation (0 if North is up), n-s pixel resolution)
  # I don't know why rotation is in twice???

  output_raster = gdal.GetDriverByName('GTiff').Create(filename,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
  output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
  srs = osr.SpatialReference()                 # Establish its coordinate encoding
  srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                               # Anyone know how to specify the 
                                               # IAU2000:49900 Mars encoding?
  output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                     # to the file
  output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster