### see github.com/tommlogan/spatial_data_discretiser
### aggregate spatial information into a square grid
### 
### INPUT:
###      data.fn of csv file with information on database
###         this database has a specific format and will be outlined here
###         - in the meantime, look at the example database
###      the grid size
###      city.name = string, capitalise first e.g. 'Baltimore', or 'Detroit'
### OUTPUT:
###      csv with aggregated spatial information
###      shapefile containing same information
###
###      
### AUTHOR:
###      Tom Logan
###
### NOTES:
###      A function for R
###      The database has a specific format and will be outlined here - in the 
###      meantime, look at the example database
###      We can aggregate all three types of spatial data (point pattern, area level, and geostatistical)
###      *** the first row of the datasource file must be a shapefile with the entire area data
###      All shapefiles must be It also must be projected in the units you want. e.g. NAD83 (feet)
###      Polygons where interested in the distance need to be in both 
####
## Libraries
####
library(rgdal)
library(raster)
library(rgeos)
library(ggplot2)
library(spdep)
library(pracma)
library(maptools)
library(parallel)
library(spex)
library(pbapply)

no_cores <<- floor(detectCores() - 4) # Calculate the number of cores
kPathGriddedData <- file.path('data', 'processed', 'grid')
kPathDataSource <- file.path('code','processing')
kPathDataTemp <- file.path('data','intermediate')

main = function(data.fn,a=1, gridSize = 500){
  ####
  ## USER INPUTS
  ####
  # what is the square grid size in meters?
  # gridSize = 100#500
  # what is minimum grid area you'll include?
  minGridArea = 0
  # what projection are you using? 
  
  ####
  ## Import the data source catalogue
  ####
  dir.data <- file.path(kPathDataSource, data.fn)
  database <- read.csv(dir.data,header=T,fill = T,stringsAsFactors=FALSE)
  database.complete <- database[!apply(is.na(database) | database == "", 1, all), ]
  
  # automate this with a loop later
  cities <- unique(database.complete$City)
  for (city.name in cities){
    # get city database
    database <- database.complete[database.complete$City == city.name,]
    print(paste0(Sys.time(), ': Beginning to grid the data for ', city.name, ' at ', gridSize, 'm'))
    
    ####
    ## Create grid
    ####
    sg = createGrid(gridSize,database)
    attr(sg,'grid_size') = gridSize
    
    ####
    ## Loop through database. extract file names and data types for each
    ####
    a = 1 # startRow (useful to change sometimes for debugging)
    b = nrow(database) 
    for (i in seq(a,b)){
      # import the data
      database.row = database[i,]
      data.current = ImportData(database.row)
      data.type = attr(data.current,'data.type')
      data.name = attr(data.current,'var.name')
      print(paste('processing #',i,': ',attr(data.current,'file.name'),sep=''),row.names = FALSE)
      ####
      ## process the data type appropriately
      ####
      if ((strcmp('ACS_2014',substr(attr(data.current,'file.name'),1,8)) & !exists("before_ACS"))) {
        before_ACS = ncol(sg@data)
      } 
      if (data.type == 'areaLevel'){
        sg = areaLevel(sg,data.current)
      } else if (data.type == 'polyIntersect'){
        sg = areaInGrid(sg,sf = data.current,data.name,database)
      } else if (data.type == 'pointCount'){
        sg = pointPatternCount(sg,data.point = data.current)
      } else if (data.type == 'pointDistance' || data.type == 'polyDistance'){
        sg = determineDistance(sg,distData = data.current)
      } else if (data.type == 'lineLength'){
        sg = lineLength(sg,data.current)
      } else if (data.type == 'raster'){
        sg = processRaster(sg,data.raster = data.current)
      } else if (data.type == 'rasterCategory'){
        sg = categoriseRaster(sg, data.raster = data.current, database)
      } else if (data.type == 'areaLevel_Count'){
        sg = areaCount(sg,sf=data.current,database)
      } else if (data.type == 'GeoStat'){  
        sg = geostat(sg,data.point = data.current)
      } else {
        print('WARNING: data type not recognised')
      }
      print(i)
      print(length(sg))
      
    }
    
    ####
    ## clip to boundary
    ####
    sg@data$cId = 1:nrow(sg@data)
    
    ####
    ## spatial lag
    ####
    num_vars_pre_spat_lag = ncol(sg@data)
    sg = spatialLag(sg)
    
    ####
    ## Set the working directory for saving
    ####
    today = Sys.Date()
    date_str = format(today,format="%Y-%m-%d")
    dir.save = file.path(kPathGriddedData, city.name, date_str)
    dir.create(file.path(kPathGriddedData, city.name))
    dir.create(dir.save)
    dir.previous <- getwd()
    setwd(dir.save)
    # what is the output filename?
    outputFileName <- paste(tolower(city.name),'_data_',gridSize,sep='')
    outvar.name <- paste0('data.',tolower(city.name))
    
    ####
    ## save as RData file
    ####
    assign(outvar.name, sg@data)
    save(list = outvar.name, file = paste(outputFileName,'.RData',sep='')) 
    
    ####
    ## add to the csv file
    ####
    write.csv(sg@data,paste(outputFileName,'.csv',sep=''))
    
    ####
    ## create a shape file
    ####
    if (strcmp('ACS_2014',substr(attr(data.current,'file.name'),1,8))){
      write_ints = c(1,seq(before_ACS,ncol(sg@data),by=500),ncol(sg@data))
      for (i in 2:length(write_ints)){
        sg_write = sg[write_ints[i-1]:write_ints[i]]
        out_name = paste(outputFileName,'_',(i-1),sep='')
        
        writeOGR(sg_write,dir.save, out_name, driver = 'ESRI Shapefile')
        writeSpatialShape(sg_write, paste(dir.save,'/',out_name,sep=''))
      } 
    } else {
      dir.save <- getwd()
      writeOGR(sg,dir.save, outputFileName, driver = 'ESRI Shapefile')
      writeSpatialShape(sg, paste(dir.save,'/',outputFileName,sep=''))
    }
    # go back to previous directory (I realize this is a hack, but maybe I'll fix it)
    setwd(dir.previous)
  }
}


ImportData = function(database.row){
  # this imports the data and assigns the attribute for what spatial data type it is
  dir.work <- getwd()
  file.type <- database.row$fileType
  file.dir <- file.path('data',database.row$Path, database.row$City)
  file.name <- paste0(database.row$FileName,database.row$fileType)
  file.epsg <- database.row$projection_epsg
  data.projection <- CRS(paste0("+init=epsg:", file.epsg))
  
  # different opening methods depending on file.type
  if (file.type == ".shp"){
    data.current = readOGR(dsn = file.dir, layer = database.row$FileName, verbose = FALSE)
    # <if there is an error "incompatible geometry: 4", go into arcGIS and use multipart to singlepart tool and save>
    # project data into meters
    data.current <- spTransform(data.current, data.projection)
  } else if (file.type == ".tif"){
    data.current = raster(file.path(file.dir, file.name))
    # project data into meters
    data.current <- projectRaster(data.current, crs = data.projection, method = 'ngb')
  } else { # .csv
    data.current = read.csv(file.name,header=T,fill = T,stringsAsFactors=FALSE)
  }
  attr(data.current,'data.type') = database.row$dataType
  attr(data.current,'var.name') = database.row$varName
  attr(data.current,'file.type') = database.row$fileType
  attr(data.current,'file.name') = database.row$FileName
  
  return (data.current)
}


createGrid = function(gridSize,database){
  ## takes the grid size input
  ## returns a grid which covers entire area
  
  # first use an area level data from database to define grid limits
  sf = ImportData(database[1,]) 
  
  # just get the outline - not internal census tracts
  # sf   <- gUnionCascaded(sf)
  
  # find max and min lat and lon
  latRange = bbox(sf)[2,2]-bbox(sf)[2,1] 
  lonRange = bbox(sf)[1,2]-bbox(sf)[1,1]
  offset = bbox(sf)[,1] + c(lonRange%%gridSize/2,latRange%%gridSize/2) # the cell offset is the location of the bottom left centroid
  nRow = ceiling(latRange/gridSize)
  nCol = ceiling(lonRange/gridSize)
  nCel = nRow*nCol 
  grd <- GridTopology(cellcentre.offset=offset,cellsize=c(gridSize,gridSize),cells.dim=c(nCol,nRow))
  # turn grid into shapefile
  polys <- as(grd, "SpatialPolygons")
  centroids <- coordinates(polys)
  x <- centroids[,1]
  y <- centroids[,2]
  z = 1:nCel
  sg = SpatialPolygonsDataFrame(polys,data=data.frame(x=x, y=y,cId=z, row.names=row.names(polys)))
  sg$area = 0
  proj4string(sg) = proj4string(sf)
  # if you get an error here: <Geographical CRS given to non-conformant data>, then you're not projected in right coordinate plane.
  # clip grid (?)
  # sg = gIntersection(sg,sf,byid=T)
  
  #plot(sg); lines(sf)
  return(sg)
}

areaLevel = function(sg,sf){
  # process area-level spatial data
  # confirm same projection
  if (proj4string(sg) != proj4string(sf)){
    sf = spTransform(sf, CRS(proj4string(sg)))
    # print(sf)
  }
  # address geometry issues
  if (length(sf) > 1) {
    sf = gBuffer(sf,width=0,byid=TRUE)
  }
  
  # intersect with the grid
  int = intersect(sg,sf)
  # calculate the subarea for each of the intersect pieces
  int$subArea = sapply(slot(int, "polygons"), slot, "area")
  
  # dissolve into the grid cells with area weighted average
  # either take just the field names not already in sg
  
  newCovars = setdiff(names(sf),names(sg)) # which covariates don't we have yet?
  if (max(sg$area)==0){ newCovars <- c(newCovars, 'area')}
  # or take all and modify the name if the name already exists
  # [is this necessary?]
  
  # iterate through the covariates and add to sg
  for (var in newCovars){
    
    if (is.factor(int[,var]@data[,1])){
      # if the variable is a factor (e.g. the name of the neighborhood, paste strings)
      newList = character(length(sg)) # initialise data list to add to sg
      for (id in unique(int$cId)){ # loop through the cIds - when intersecting, some will be repeated.
        if (sum(int$cId==id)!=0){ 
          # if there is one or more value for this cId, merge them and add to list
          newList[[id]]=  paste(int[int$cId==id,var]@data[,1],collapse = '&')
        }
      }
    }
    else {
      newList = rep(NA,length(sg)) # initialise data list
      for (id in unique(int$cId)){ 
        # for each cId, get the subArea and value data
        vals = int[int$cId==id,c('subArea',var)]@data 
        if (sg$area[id] == 0){
          # if the area is currently zero, update it
          sg$area[id]=sum(vals[,1])
        }
        # take a weighted average to determine the value in the cell
        newList[id] = weighted.mean(vals[,2],vals[,1])
      }
    }
    # add new covariate to dataframe
    if (var != 'area'){
      sg@data[,var]=newList
    }
    
  }
  # omit cells that don't have a CSA (there is no data for them)
  
  # writes to shapefile
  # writeOGR(sg,dir.work, outputFileName, driver = 'ESRI Shapefile') # writePolyShape(sg,outputFileName)
  # writes to csv
  
  return(sg)
}

areaInGrid = function(sg,sf,data.name,database,to_save = TRUE){
  # intersects the polygons with the grid and determines the area
  city.name <- unique(database$City)
  # union the polygons by id
  # lps <- getSpPPolygonsLabptSlots(sf)
  # ID.one.bin <- cut(lps[,1], range(lps[,1]), include.lowest=TRUE)
  # sf <- unionSpatialPolygons(sf, ID.one.bin)
  if (to_save){
    # check if a saved RData file already exists
    gridSize = attr(sg,'grid_size')
    gridded_filename = paste(kPathDataTemp, '/', city.name, '/',data.name,'_gridsize_',gridSize,'.RData',sep='')
    alreadyProcessed = file.exists(gridded_filename)
    
    if (alreadyProcessed) {
      # if it does exist, load the data file
      load(gridded_filename)
    }  else {
      
      if (proj4string(sg_temp) != proj4string(sf)){
        sf = spTransform(sf, CRS(proj4string(sg_temp)))
      }
      # address geometry issues
      sf_buffer <- function(sf) {
        out <- tryCatch(
          {
            # Just to highlight: if you want to use more than one 
            # R expression in the "try" part then you'll have to 
            # use curly brackets.
            # 'tryCatch()' will return the last evaluated expression 
            # in case the "try" part was completed successfully
            sf = gSimplify(sf, tol = 0.00001);
            if (!gIsValid(sf)){
              sf = gBuffer(sf,width=0,byid=TRUE);
            }
          },
          error=function(cond) {
            
            # Choose a return value in case of error
            return(sf)
          }
        )    
        return(sf)
      }
      
      # intersect the grid with the shapefile
      sg_temp <- gIntersection(sf,sg,byid = TRUE)
      
      # calculate the area
      sg_temp$area <- area(sg_temp)
      sg_temp@data[,data.name] <- sg_temp$area
      sg_temp$area <- NULL
      
      # fix the row names
      ids <- do.call(rbind, strsplit(row.names(sg_temp), ' '))
      # sg_temp$g_idx <- ids[,2]
      sg_temp$idx <- match(ids[,2], rownames(sg@data))
      
    }
    # save
    save(sg_temp, file = gridded_filename)
  }
  
  ## JOIN THE SG
  sg <- merge(sg, sg_temp@data, by.x = "cId", by.y = 'idx')
  
  return(sg)
}

areaCount = function(sg,sf=data.current,database){
  # process area-level data by finding taking the area weighted sum.
  city.name <- unique(database$City)
  # check if a saved RData file already exists
  gridSize = attr(sg,'grid_size')
  data.name = attr(sf,'var.name')
  gridded_filename = paste(kPathDataTemp, '/', city.name, '/',data.name,'_gridsize_',gridSize,'.RData',sep='')
  alreadyProcessed = file.exists(gridded_filename)
  
  if (alreadyProcessed) {
    # if it does exist, load the data file
    load(gridded_filename)
  }  else {
    ## CREATE TEMPORARY GRID
    sg_temp = createGrid(gridSize,database)
    
    # confirm same projection
    if (proj4string(sg) != proj4string(sf)){
      sf = spTransform(sf, CRS(proj4string(sg)))
      # print(sf)
    }
    # address geometry issues
    sf_buffer <- function(sf) {
      out <- tryCatch(
        {
          # Just to highlight: if you want to use more than one 
          # R expression in the "try" part then you'll have to 
          # use curly brackets.
          # 'tryCatch()' will return the last evaluated expression 
          # in case the "try" part was completed successfully
          
          if (!gIsValid(sf)){
            sf = gBuffer(sf,width=0,byid=TRUE);
            if (!gIsValid(sf)){
              sf = gSimplify(sf, tol = 0.00001);
            }
          }
        },
        error=function(cond) {
          
          # Choose a return value in case of error
          return(sf)
        }
      )    
      return(sf)
    }
    sf = sf_buffer(sf)
    
    # make a list of the polygons in the grid
    grid_cells = SpatialPolygons(sg@polygons)
    # list of polygons in the sf
    blocks = SpatialPolygons(sf@polygons)
    # function to calculate area of a polygon
    poly_area = function(j){
      p1 = blocks[j]
      proj4string(p1) = proj4string(sg)
      area = gArea(p1)
      return(area)
    }
    # calculate the area of each block
    # parallelise
    cl <- makeCluster(no_cores,outfile="")
    clusterExport(cl, c("sg","blocks","var","poly_area"), envir=environment())
    clusterEvalQ(cl, c(library(rgeos),library(raster)))
    # area of each block
    newList = parLapply(cl,seq(1,length(blocks)),function(j) poly_area(j))
    stopCluster(cl)
    block_areas = unlist(newList)
    sf@data$area = block_areas
    
    # function to determine sum within a cell
    count_in_poly = function(j){
      p1 = grid_cells[j]
      proj4string(p1) = proj4string(sf)
      # intersect the cell with the blocks
      int = intersect(p1,sf)
      # find the area of each block which intersects the grid cell
      if (is.null(int)){
        cell_count=0
      } else {
        intersect_area = sapply(slot(int, "polygons"), slot, "area")
        # extract the area and variable information for each block
        block_area = int@data$area
        block_count = int@data[,var]
        if (strcmp(var,'bldg_storey')){
          ignore = block_count == 999999999999
          block_count = block_count[!ignore]
          block_area = block_area[!ignore]
        }
        block_count
        # for each block find the ration of area in cell to total area, then multiply by the count
        cell_count = sum(block_count * intersect_area/block_area)
      }
      return(cell_count)
    }
    
    # take the field names not already in sg
    newCovars = setdiff(names(sf),names(sg))
    # iterate through the covariates and add to sg
    for (var in newCovars){
      
      if (is.factor(sf[,var]@data[,1])){
        # if the variable is a factor (e.g. the name of the neighborhood)
        # ignore
      }
      else {
        # parallelise
        
        cl <- makeCluster(no_cores,outfile="")
        clusterExport(cl, c("sf","grid_cells","var","count_in_poly"), envir=environment())
        clusterEvalQ(cl, c(library(raster),library(pracma)))
        # compute the count value within each grid cell
        newList = parLapply(cl,seq(1,length(grid_cells)),function(j) count_in_poly(j))
        stopCluster(cl)
        
        # add new covariate to dataframe
        sg_temp@data[,var]=unlist(newList)
      }
      
    }
    # remove excess fields
    sg_temp$area = NULL; sg_temp$cId = NULL; sg_temp@data$x = NULL; sg_temp@data$y = NULL; 
    save(sg_temp, file = gridded_filename)
  }
  
  ## JOIN THE SG
  sg@data = cbind(sg@data, sg_temp@data)
  
  # writes to shapefile
  # writeOGR(sg_temp,dir.work, 'height_cells', driver = 'ESRI Shapefile') # writePolyShape(sg,outputFileName)
  # writes to csv
  
  return(sg)
}

pointPatternCount = function(sg,data.point){
  # process point pattern data
  # return a count of number of points within each grid cell
  data.name = attr(data.point,'var.name')
  file.type = attr(data.point,'file.type')
  
  if (file.type == ".csv"){
    data.point = csvToSpatial(data.point)
  }
  # projection
  if (proj4string(sg) != proj4string(data.point)){
    data.point = spTransform(data.point,CRS(proj4string(sg)))
    # print(sf)
  }
  
  # overlay with polygons
  whichCid <- over(data.point, sg[,"cId"])
  
  # count and assign to sg
  counts = table(unlist(whichCid))
  sg@data[data.name] = 0*nrow(sg)
  sg@data[as.integer(rownames(counts)),data.name] = counts
  
  # writes to shapefile
  # writeOGR(data.point,dir.work, 'data.point_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  
  #   #bubble plot
  #   plot(sf)
  #   count.max = max(sg@data$vbCrime)
  #   colors = sapply(sg@data$vbCrime, function(n) hsv(sqrt(n/count.max),.7, .7, .5))
  #   points(sg@data$x + 1/2, sg@data$y + 1/2, cex = sqrt(sg@data$vbCrime/100), pch=19,col=colors)
  #   title('Baltimore Crime')
  
  return(sg)
}

determineDistance = function(sg,distData){
  # process data
  # return the distance from center of each grid cell to nearest point/polygon
  
  data.name = attr(distData,'var.name')
  file.type = attr(distData,'file.type')
  data.type = attr(distData,'data.type')
  
  if (file.type == ".csv"){
    distData = csvToSpatial(distData)
  } else if (data.type == 'polyDist'){
    # if shapefile, convert to line so is distance to edge
    linesData = as(distData,'SpatialLines')  
  }
  # are the projections the same?
  sameProj = strcmp(proj4string(sg),  proj4string(distData))
  if (!sameProj){
    # projection
    distData = spTransform(distData,CRS(proj4string(sg)))
  }
  
  # determine minimum distance and assign to sg
  sg@data[data.name] = 0*nrow(sg)
  for (j in seq(nrow(sg))){
    # determine centroid of cell
    centroid = readWKT(paste("POINT(",toString(coordinates(sg[j,])[[1]]),toString(coordinates(sg[j,])[[2]]),")"))
    proj4string(centroid) = proj4string(sg)
    # sp2   <- SpatialPoints(centroid,proj4string=CRS(proj4string(sg)))
    
    if (data.type == 'polyDist'){
      # check to see if point is in polygon
      inPoly = gContains(distData,centroid)
      # print(inPoly)
      if (inPoly){
        # print('in poly')
        # distance is zero
        sg@data[j,data.name] = 0
      } else {
        sg@data[j,data.name] = min(gDistance(centroid, linesData, byid=TRUE))
      }
    } else{
      sg@data[j,data.name] = min(gDistance(centroid, distData, byid=TRUE))
    }
  }
  
  
  # writes to shapefile
  # writeOGR(sg,dir.work, 'data.pointDist_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}

csvToSpatial = function(data.point){
  # the csv file needs a column 'Latitude' and a column 'Longitude'
  # data.point = read.csv('Data/pointPatternCount_example_vbcrime.csv',header=T,fill = T,stringsAsFactors=FALSE)
  # these columns need to be numeric, remove any NAs
  latCol = which(tolower(colnames(data.point))=='latitude' | tolower(colnames(data.point))=='lat')
  lonCol = which(tolower(colnames(data.point))=='longitude' | tolower(colnames(data.point))=='long' | tolower(colnames(data.point))=='lon'| tolower(colnames(data.point))=='lng')
  data.point[,latCol] = as.numeric(data.point[,latCol])
  data.point <- data.point[!is.na(data.point[,latCol]),] # this should remove all rows with NAs
  data.point <- data.point[!(data.point[,latCol] >= 999),] # this should remove all rows with 99999 for lat e.g. non conformant lat/lon
  data.point[,lonCol] = as.numeric(data.point[,lonCol])
  # converts to spatial points data frame
  coordinates(data.point) = c(lonCol,latCol)
  
  # projection(data.point) = crs(sg)
  proj4string(data.point) =  "+proj=longlat +datum=WGS84"
  
  return(data.point)
}

lineLength = function(sg,lineData){
  # returns the length of lines within the cell (e.g. bike lane)
  data.name = attr(lineData,'var.name')
  file.type = attr(lineData,'file.type')
  
  # address projection
  lineData = spTransform(lineData, CRS(proj4string(sg)))
  # split into cells and find total length
  rp <- intersect(lineData, sg)
  rp$length <- gLength(rp, byid=TRUE) / 1000
  lengthCell <- tapply(rp$length, rp$cId, sum)
  # append to datastructure
  sg@data[data.name] = 0*nrow(sg)
  sg@data[as.integer(rownames(lengthCell)),data.name] = lengthCell
  
  return (sg)
}

spatialLag = function(sg){
  # a list of neighbours for each gridcell
  neighs = poly2nb(sg)
  dimSg = dim(sg)
  var.nameList = colnames(sg@data)
  # loop through the data variables
  for (i in seq(5,dimSg[2])){
    # for each variable that is after the 4th (which are the presets (area, coordinates, id))
    if (class(sg@data[,i]) != 'character'){
      # not including strings
      var.name = paste(var.nameList[i],'_sl',sep='')
      for (j in seq(dimSg[1])){ 
        # loop through the cells
        sg@data[j,var.name] = mean(sg@data[neighs[[j]],i])
        # I think a better way to do this is using one of the apply functions
      }
    }
  }
  return (sg)
}

processRaster = function(sg,data.raster = data.current){
  # process raster data
  # determine the average of the values within the grid cell
  
  data.name = attr(data.raster,'var.name')
  file.type = attr(data.raster,'file.type')
  
  # projection
  data.raster = projectRaster(data.raster,crs=CRS(proj4string(sg)))
  
  # clip raster to grid
  cropbox = extent(sg)
  data.raster = crop(data.raster,cropbox)
  
  # extract heat values for each cell from raster
  cell_vals = extract(data.raster,sg)
  cell = list()
  # Use list apply to calculate mean for each grid cell
  cell$mean = lapply(cell_vals, FUN=mean, na.rm=TRUE)
  cell$max = lapply(cell_vals, FUN=max, na.rm=TRUE)
  cell$min = lapply(cell_vals, FUN=min, na.rm=TRUE)
  cell$sd = lapply(cell_vals, FUN=sd, na.rm=TRUE)
  # unlist, brings nested values into single list
  cell$mean = unlist(cell$mean)
  cell$max = unlist(cell$max)
  cell$min = unlist(cell$min)
  cell$sd = unlist(cell$sd)
  # Join values to polygon data
  # append to datastructure
  funs = c('mean','max', 'min','sd')
  for (fun in funs){
    newVar = paste(data.name,'_',fun,sep='')
    sg@data[newVar] = 0*nrow(sg)
    sg@data[,newVar] = cell[fun]
  }
  # writes to shapefile
  # writeOGR(sg,dir.work, 'raster_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}

categoriseRaster = function(sg, data.raster, database){
  # process raster data
  # determine the average of the values within the grid cell
  
  data.name = attr(data.raster,'var.name')
  file.type = attr(data.raster,'file.type')
  city.name <- unique(database$City)
  # this function takes a long time.
  # check if a saved RData file already exists
  gridSize = attr(sg,'grid_size')
  gridded_raster_filename = paste(kPathDataTemp, '/', city.name, '/',data.name,'_gridsize_',gridSize,'.RData',sep='')
  alreadyProcessed = file.exists(gridded_raster_filename)
  print('grid size is:')
  print(gridSize)
  
  if (alreadyProcessed) {
    # if it does exist, load the data file
    load(gridded_raster_filename)
  } else { # if not, process the data
    
    ## CREATE TEMPORARY GRID
    sg_temp = createGrid(gridSize,database)
    attr(sg_temp,'grid_size') = gridSize
    
    # identify the categories in the raster
    if (strcmp(file.type,'.tif')){
      ## PROCESS THE RASTER
      ####
      # get cropbox for raster clip
      cropbox = extent(sg_temp)
      data.raster = crop(data.raster,cropbox)
      
      cats = unique(data.raster)
    } else {
      cats = unique(data.raster@data$Color)
    }
    
    # init the dataframe with the categories
    area.classes <- paste0(data.name, '_',cats)
    sg_temp@data[,area.classes] <- 0
    
    # subset raster
    rast <- data.raster
    # rast[rast != catg] <- NA
    
    # make a list of the polygons in the grid
    grid.cells <- SpatialPolygons(sg_temp@polygons)
    grid.number <- length(grid.cells)
    
    raster_area_in_poly = function(j){
      # subset raster to grid cell
      p1 = grid.cells[j]
      proj4string(p1) = proj4string(rast)
      r.cell = mask(rast, p1)
      
      # get areas
      rast.area <- tapply(raster::area(r.cell), r.cell[], sum)
      if (length(rast.area) == 0){
        rast.area <- array(0, dim = c(length(cats)))
        names(rast.area) <- cats
      }
      
      return(rast.area)
    }
    
    # parallelise
    cl <- makeCluster(no_cores,outfile="")
    clusterExport(cl, c("rast","grid.cells","raster_area_in_poly"), envir=environment())
    clusterEvalQ(cl, c(library(raster)))
    area.list = pblapply(seq(1,grid.number),function(j) raster_area_in_poly(j))#, cl = cl)
    stopCluster(cl)
    
    # add new covariate to dataframe
    for (g in seq(1,grid.number)){
      area.g <- area.list[[g]]
      area.classes <- names(area.g)
      area.classes <- paste0(data.name, '_',names(area.g))
      sg_temp@data[g, area.classes] <- area.g
    }
    
    ## SAVE CATEGORISED RASTER
    ####
    # remove excess fields
    sg_temp$area = NULL; sg_temp$cId = NULL; sg_temp@data$x = NULL; sg_temp@data$y = NULL; 
    # save
    save(sg_temp, file = gridded_raster_filename)
  }
  
  ## JOIN THE SG
  sg@data = cbind(sg@data, sg_temp@data)
  
  # writes to shapefile
  # writeOGR(sg,dir.work, 'raster_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}




geostat = function(sg,data.point = data.current){
  # this function processes geostatistical data
  # it returns the min, max, median, and mean of the point geostat data
  
  data.name = attr(data.point,'var.name')
  file.type = attr(data.point,'file.type')
  
  if (file.type == ".csv"){
    data.point = csvToSpatial(data.point)
  }
  # projection
  if (proj4string(sg) != proj4string(data.point)){
    data.point = spTransform(data.point,CRS(proj4string(sg)))
    # print(sf)
  }
  
  # make a list of the polygons in the grid
  grid_cells = SpatialPolygons(sg@polygons)
  
  # initialise data
  funs = c('mean','max', 'min','median')
  for (fun in funs){
    newVar = paste(data.name,'_',fun,sep='')
    sg@data[newVar] = 0*nrow(sg)
  }
  
  # loop through cIds
  for (j in sg@data$cId){
    # get the grid cell
    p1 = grid_cells[j]
    proj4string(p1) = proj4string(data.point)
    # get sub points
    sub_points = data.point[p1,]
    vals = strtoi(sub_points@data[,1])
    # loop through the functions
    sg@data[j, paste(data.name,'_mean',sep='')] = mean(vals, na.rm=TRUE)
    sg@data[j, paste(data.name,'_median',sep='')] = median(vals, na.rm=TRUE)
    sg@data[j, paste(data.name,'_max',sep='')] = max(vals, na.rm=TRUE)
    sg@data[j, paste(data.name,'_min',sep='')] = min(vals, na.rm=TRUE)
    
  }
  
  
  return(sg)
}

