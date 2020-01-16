library(horizon)
library(raster)


# main <- function(){
  
  city.name <- 'phx'
  res <- '6'
  filename.in <- paste0('/mnt/StorageArray/tlogan/research/urban_data/land_surface_temperature/data/',city.name,'/',city.name,'_dsm_wgs_',res,'.tif')
  filename.out <- paste0('/mnt/StorageArray/tlogan/research/urban_data/land_surface_temperature/data/',city.name,'/',city.name,'_svf_',res,'.tif')
  # filename.in <- paste0('F:/UrbanDataProject/land_surface_temperature/data/intermediate/',city.name,'/DSM_degrees.tif')
  # filename.in <- paste0('F:/UrbanDataProject/land_surface_temperature/data/intermediate/',city.name,'/',city.name,'_dsm_wgs.tif')
  # filename.out <- paste0('F:/UrbanDataProject/land_surface_temperature/data/processed/image/',city.name,'/',city.name,'_svf.tif')
  
  # import the raster
  ras <- raster(filename.in)
  
  # calculate the svf
  phi = 10
  R = 300
  svf_city <- svf(x=ras, nAngles=phi, maxDist = R)
  
  # write raster
  writeRaster(svf_city, filename.out, overwrite=T)
  
  # plot
  # plot(svf)

  # por - 16077
  # phx - 129187
  # bal - 127048
  # det - 121304
  