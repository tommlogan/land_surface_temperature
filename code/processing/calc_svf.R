library(horizon)
library(raster)


city.name <- 'bal'
filename.in <- paste0('F:/UrbanDataProject/land_surface_temperature/data/intermediate/',city.name,'/DSM_degrees.tif')
filename.out <- paste0('F:/UrbanDataProject/land_surface_temperature/data/processed/image/',city.name,'/',city.name,'_svf.tif')

# import the raster
ras <- raster(filename.in)

# calculate the svf
phi = 10
R = 300
svf_city <- svf(x=ras, nAngles=phi, maxDist = R)

# write raster
writeRaster(svf_city, filename.out, overwrite=T)

# plot
plot(svf)
