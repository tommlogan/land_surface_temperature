library(rgdal)
library(raster)


# import geojson
state <- 'Arizona'
city.name <- 'phx'
file.name <- paste0('B:/research/land_surface_temperature/data/building_footprint/',state,'/',state,'.geojson')
df <- readOGR(dsn=file.name,layer = 'OGRGeoJSON')

# import city polygon
city.dir <- paste0('B:/research/land_surface_temperature/data/', city.name)
city <- readOGR(dsn = city.dir, layer = paste0(city,'_boundary'), verbose = FALSE)

# clip the polygons to the city
bldg.city <- intersect(df,city)


# export shapefile
exp.dsn <- paste0('B:/research/land_surface_temperature/data/building_footprint/',state)
exp.layer <- paste0('building_footprint_',city.name)
writeOGR(obj=bldg.city, dsn=exp.dsn, layer=exp.layer, driver="ESRI Shapefile", overwrite_layer = T)
