## 
## Read geographic data and clip
## 
## Inputs:
##  satellite filename
##  city boundary filename
##  land cover map filename
##  city name
##
## Outputs:
##  clipped satellite and land cover maps
##


library(rgdal)
library(raster)
library(rgeos)


main <- function(city, landsat_id, fn_land_cover, fn_tree, fn_imperv, fn_city_boundary, bands){
  # reads in the arguments passed
  
  # import city boundary
  city.buffer <- read_city_boundary(city, fn_city_boundary)
  
  # import satellite image and clip to city buffer
  bands <- as.list(strsplit(bands,',')[[1]])
  for (band in bands){
    # loop through the bands
    satellite.city <- clip_satellite(city.buffer, landsat_id, band)
  }
  
  # import land cover image and clip to city buffer
  clip_land_cover(city.buffer, fn_land_cover, satellite.city, 'LC')
  
  # import tree canopy image and clip to cify buffer
  clip_land_cover(city.buffer, fn_land_cover, satellite.city, 'CAN')
  
  # import impervious surface image and clip to cify buffer
  clip_land_cover(city.buffer, fn_land_cover, satellite.city, 'IMP')
  
}

read_city_boundary <- function(city, fn_city_boundary){
  # import the city boundary polygon, buffer it, and save it
  
  path.city <- file.path('data','intermediate',city)
  name.boundary <-paste0(city, '_boundary')
  name.buffer <- paste0(city, '_buffer')
  
  # check if processed already
  alreadyProcessed = file.exists(file.path(path.city, paste0(name.buffer, '.shp')))
  if (alreadyProcessed){
    # import the buffered shapefile
    city.buffer <- readOGR(dsn = path.city, layer = name.buffer)
  } else {
    # import the original
    city.boundary <- readOGR(dsn = file.path('data', 'raw', city, fn_city_boundary), layer = fn_city_boundary)
    
    # union shape file
    city.boundary <- gUnaryUnion(city.boundary)
    
    # buffer
    city.buffer <- city.boundary
    city.buffer <- spTransform(city.boundary, CRS("+init=epsg:26978"))
    city.buffer <- gBuffer(city.buffer, width = 2000, byid=TRUE) # 2km buffer
    
    # project all to WSG84
    city.buffer <- spTransform(city.buffer, CRS("+init=epsg:4326"))
    city.boundary <- spTransform(city.boundary, CRS("+init=epsg:4326"))
    
    # convert to spatial polygons dataframe for saving
    city.boundary <- SpatialPolygonsDataFrame(city.boundary, data.frame(city = city))
    city.buffer <- SpatialPolygonsDataFrame(city.buffer, data.frame(city = city))
    
    # save
    writeOGR(city.boundary, path.city, name.boundary, driver = 'ESRI Shapefile')
    writeOGR(city.buffer, path.city, name.buffer, driver = 'ESRI Shapefile')
  }
  return(city.buffer)
}


clip_satellite <- function(city.buffer, landsat_id, band){
  # import the satellite image, clip it to the city, and save it
  print('importing satellite image')
  path.satellite <- file.path('data','intermediate', city, paste0(landsat_id, '_B', band, '.tif'))
  
  # check if processed already
  alreadyProcessed = file.exists(path.satellite)
  if (alreadyProcessed){
    # import the buffered shapefile
    satellite.city <- raster(path.satellite)
  } else {
    # import the raw image
    print('import')
    satellite.all <- raster(file.path('data','raw',city,paste0(landsat_id, '_B', band, '.tif')))
    

    # change projection of city 
    city.proj <- spTransform(city.buffer, CRS(proj4string(satellite.all)))

    # clip
    print('mask')
    satellite.city <- mask(satellite.all, city.proj)
    print('crop')
    satellite.city <- crop(satellite.city, extent(city.proj))
    
    # project
    print('project')
    rasterOptions(maxmemory = 1e+07)
    satellite.city <- projectRaster(satellite.city, crs=CRS("+init=epsg:4326"))
    
    # save
    writeRaster(satellite.city, path.satellite)
  }
  return(satellite.city)
}


clip_land_cover <- function(city.buffer, fn_land_cover, satellite.city, cover_type){
  # import the land cover image, clip it to the city, and save it
  print(paste0('importing ', cover_type))
  path.landcover <- file.path('data','processed',city, paste0('NLCD2011_',cover_type,'_', city, '.tif'))
  
  # check if processed already
  alreadyProcessed = file.exists(path.landcover)
  if (alreadyProcessed){
    # import the buffered shapefile
    landcover.city <- raster(path.landcover)
  } else {
    # import the raw image
    print('import')
    landcover.all <- raster(file.path('data','raw',city,fn_land_cover, paste0(fn_land_cover, '.tif')))
    
    # change projection of satellite
    satellite.proj <- projectRaster(satellite.city, crs=CRS(proj4string(landcover.all)))
    
    # nearest neighbor resample
    print('resample')
    landcover.city <- resample(landcover.all, satellite.proj, 'ngb')
    
    # crop and mask
    print('crop and mask')
    landcover.city <- crop(landcover.city, extent(satellite.proj))
    landcover.city <- mask(landcover.city, satellite.proj)
    
    # change projection
    rasterOptions(maxmemory = 1e+07)
    print('project')
    landcover.city <- projectRaster(landcover.city, crs=CRS("+init=epsg:4326"), method = 'ngb')
    
    # repeat now that's in the correct projection
    print('resample')
    landcover.city <- resample(landcover.city, satellite.city, 'ngb')
    
    # crop and mask
    print('crop and mask')
    landcover.city <- crop(landcover.city, extent(satellite.city))
    landcover.city <- mask(landcover.city, satellite.city)
    
    # save
    writeRaster(landcover.city, path.landcover)
  }
}

# Fetch command line arguments
args.passed <- commandArgs(trailingOnly = TRUE)
# inputs
city <- args.passed[1]
landsat_id <- args.passed[2]
fn_land_cover <- args.passed[3]
fn_city_boundary <- args.passed[4]
bands <- args.passed[5]
fn_tree <- args.passed[6]
fn_imperv <- args.passed[7]
# # temporary - during writing
# city <- 'bal'
# fn_city_boundary <- 'tl_2012_24510_faces'
# fn_land_cover <- 'NLCD2011_LC_Maryland'
# landsat_id <- 'LC08_L1TP_015033_20170907_20170926_01_T1'

# run main
main(city, landsat_id, fn_land_cover, fn_tree, fn_imperv, fn_city_boundary, bands)

