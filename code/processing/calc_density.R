# from the 2010 census data, clip the block polygon to the city, and assign the population, area, and density (pop/area)  to each block polygon

library(data.table)
library(rgdal)
library(sp)
library(rgeos)
library(raster)
library(maptools)

# import the population data
df <- fread('data/raw/nhgis/nhgis0029_csv/nhgis0029_ds172_2010_block.csv')
# assign a readable colum name
df$pop <- df$H7V001
df.pop <- df[,c('GISJOIN', 'pop')]



cities <- c('phx', 'bal','det','por')
states <- c('AZ', 'MD','MI','OR')
state_codes <- c('040', '240','260','410')

# loop through each city
for (i in seq(2,4)) {
  city <- cities[i]
  print(city)
  state <- states[i]
  st_code <- state_codes[i]

  # import the city boundary
  file.dir <- paste0('F:/UrbanDataProject/land_surface_temperature/data/intermediate/', city)
  file.name <- paste0(city,'_buffer')
  boundary = readOGR(dsn = file.dir, layer = file.name, verbose = FALSE)
  
  # import the state block polygons
  file.dir <- paste0('F:/UrbanDataProject/land_surface_temperature/data/raw/nhgis/nhgis0029_shape/nhgis0029_shapefile_tl2010_', st_code, '_block_2010')
  file.name <- paste0(state,'_block_2010')
  blocks = readOGR(dsn = file.dir, layer = file.name, verbose = FALSE)
  
  # clip the polygons to the city
  boundary <- spTransform(boundary, proj4string(blocks))
  blocks.city <- intersect(blocks, boundary)
  
  
  # calculate the area of the polygons
  blocks.city$polyArea <- gArea(blocks.city, byid = T)
  
  # drop blocks with no area
  blocks.city <- blocks.city[blocks.city$polyArea > 1,]
  
  # join the pop to the polygons
  blocks.city <- merge(x=blocks.city, y = df.pop, by = 'GISJOIN', all.x = TRUE)
  
  # calculate the population density (pden)
  # shape_area is in m2
  # therefore, divide it by 1e6 to get km2
  blocks.city$pop_dens <- blocks.city$pop / (blocks.city$polyArea/1e6)
  
  # drop all variables except pop_dens
  blocks.city <- blocks.city[,'pop_dens']
  
  # write the polygon
  file.dir <- paste0('F:/UrbanDataProject/land_surface_temperature/data/processed/', city)
  file.out <- paste0(city, '_block_pop')
  writeOGR(blocks.city, file.dir, file.out, driver = 'ESRI Shapefile', overwrite_layer = T)
  # writeSpatialShape(blocks.city, paste(file.dir, '/', file.out, sep=''))
}
