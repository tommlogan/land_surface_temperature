library(rgdal)
library(raster)
library(rgeos)

# import geojson
state <- 'Michigan'
city.name <- 'det'
# file.name <- paste0('B:/research/land_surface_temperature/data/building_footprint/',state,'/',state,'.geojson')
file.name <- paste0('/mnt/StorageArray/tlogan/research/urban_data/land_surface_temperature/data/building_footprint/',state,'/',state,'.geojson')
df <- readOGR(dsn=file.name,layer = 'OGRGeoJSON')

# address invalid geometry issue
df <- gBuffer(df, width=0, byid=T)

# import city polygon
# city.dir <- paste0('B:/research/land_surface_temperature/data/', city.name)
city.dir <- paste0('/mnt/StorageArray/tlogan/research/urban_data/land_surface_temperature/data/', city.name)
city <- readOGR(dsn = city.dir, layer = paste0(city.name,'_boundary'), verbose = FALSE)

# clip the polygons to the city
bldg.city <- intersect(df,city)
df <- NULL # clear memory

# export shapefile
# exp.dsn <- paste0('B:/research/land_surface_temperature/data/building_footprint/',state)
exp.dsn <- paste0('/mnt/StorageArray/tlogan/research/urban_data/land_surface_temperature/data/building_footprint/',state)
exp.layer <- paste0('building_footprint_',city.name)
writeOGR(obj=bldg.city, dsn=exp.dsn, layer=exp.layer, driver="ESRI Shapefile", overwrite_layer = T)
