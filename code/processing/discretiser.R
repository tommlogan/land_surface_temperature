### see github.com/tommlogan/spatial_data_discretiser
### aggregate spatial information into a square grid
### 
### INPUT:
###      data.fn of csv file with information on database
###         this database has a specific format and will be outlined here
###         - in the meantime, look at the example database
###      the grid size
###      cityName = string, capitalise first e.g. 'Baltimore', or 'Detroit'
### OUTPUT:
###      csv with aggregated spatial information
###      shapefile containing same information
###
###      
### AUTHOR:
###      Tom Logan
### MODIFIED:
###      June 2016
###
### NOTES:
###      A function for R
###      The database has a specific format and will be outlined here - in the 
###      meantime, look at the example database
###      We can aggregate all three types of spatial data (point pattern, area level, and geostatistical)
###      *** the first row of the datasource file must be a shapefile with the entire area data
###      All shapefiles must be It also must be projected in the units you want. e.g. NAD83 (feet)
###      Polygons where interested in the distance need to be in both 
#######
## Libraries
#######
# library(maptools)
library(rgdal)
library(raster)
library(rgeos)
library(ggplot2)
library(spdep)
library(pracma)
library(maptools)
library(parallel)
no_cores <<- floor(detectCores() - 6) # Calculate the number of cores

main = function(data.fn,a=1){
  #######
  ## USER INPUTS
  #######
  # what is the square grid size in feet?
  gridSize = 2000
  # what is minimum grid area you'll include?
  minGridArea = 0
  # what projection are you using? 
  
  ####
  ## Import the data source catalogue
  ####
  database = read.csv(file.path('code','processing', data.fn),header=T,fill = T,stringsAsFactors=FALSE)
  database = database[!apply(is.na(database) | database == "", 1, all), ]
  
  ####
  ## Create grid
  ####
  sg = createGrid(gridSize,database)
  attr(sg,'grid_size') = gridSize
  
  ####
  ## Loop through database. extract file names and data types for each
  ####
  # a = 1 # startRow (useful to change sometimes for debugging)
  b = nrow(database) #40 was last to work #45 didn't work
  for (i in seq(a,b)){
    # import the data
    theData = importData(databaseRow = database[i,])
    dataType = attr(theData,'dataType')
    dataName = attr(theData,'varName')
    print(paste('processing #',i,': ',attr(theData,'fileName'),sep=''),row.names = FALSE)
    ####
    ## process the data type appropriately
    ####
    if ((strcmp('ACS_2014',substr(attr(theData,'fileName'),1,8)) & !exists("before_ACS"))) {
      before_ACS = ncol(sg@data)
    } 
    if (dataType == 'areaLevel'){
      sg = areaLevel(sg,theData)
    } else if (dataType == 'polyIntersect'){
      sg = areaInGrid(sg,sf = theData,dataName,database)
    } else if (dataType == 'pointCount'){
      sg = pointPatternCount(sg,pointData = theData)
    } else if (dataType == 'pointDistance' || dataType == 'polyDistance'){
      sg = determineDistance(sg,distData = theData)
    } else if (dataType == 'lineLength'){
      sg = lineLength(sg,theData)
    } else if (dataType == 'raster'){
      sg = processRaster(sg,rasterData = theData)
    } else if (dataType == 'rasterCategory'){
      sg = categoriseRaster(sg,rasterData = theData,database)
    } else if (dataType == 'areaLevel_Count'){
      sg = areaCount(sg,sf=theData,database)
    } else if (dataType == 'GeoStat'){  
      sg = geostat(sg,pointData = theData)
    } else {
      print('WARNING: data type not recognised')
    }
    print(i)
    print(length(sg))
    
    #     print(warnings())
    #     assign("last.warning", NULL, envir = baseenv())
    #     if (i==1){
    #       ####3
    #       ## clip to boundary
    #       ####
    #       sg = sg[sg$area>minGridArea,]
    #     }
  }
  
  ####
  ## clip to boundary
  ####
  #if (i == 1){ # only need to do it once
  # if (cityName=='Baltimore'){
  #   sg = sg[sg$area>minGridArea,]
  # } else if (cityName=='Detroit'){
  #   sg = sg[sg@data$districts!="",]
  # }
  
  # for Detroit we'll use the districts. not sure why the area is not working.
  # sg = sg[sg@data$districts!="",]
  # you will get " Error in poly2nb(sg) : non-positive number of entities " if no shape files in this dataset to provide area.
  # if this is the case, remove the minArea requirement.
  #}
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
  save_dir = paste('F:/UrbanDataProject/cities/',cityName,'/gridded_data/',date_str,sep='')
  dir.create(save_dir)
  setwd(save_dir)
  # what is the output filename?
  outputFileName = paste(tolower(cityName),'_data',sep='')
  outvar.name <- paste('data.',tolower(cityName),sep='')
  ####
  ## save as RData file
  ####
  assign(outvar.name, sg@data)
  save(list = outvar.name, file = paste(outputFileName,'.RData',sep='')) 
  # if (cityName=='Baltimore'){
  #   baltimoreData = sg@data
  #   save(baltimoreData, file = paste(outputFileName,'.RData',sep='')) 
  # } else if (cityName=='Detroit'){
  #   detroitData = sg@data
  #   save(detroitData, file = paste(outputFileName,'.RData',sep='')) 
  # }
  
  ####
  ## add to the csv file
  ####
  write.csv(sg@data,paste(outputFileName,'.csv',sep=''))
  
  ####
  ## create a shape file
  ####
  # half_len = ncol(sg@data)/2
  #   writeOGR(sg,workDir, outputFileName, driver = 'ESRI Shapefile')
  #   writeSpatialShape(sg, outputFileName)   
  if (strcmp('ACS_2014',substr(attr(theData,'fileName'),1,8))){
    write_ints = c(1,seq(before_ACS,ncol(sg@data),by=500),ncol(sg@data))
    for (i in 2:length(write_ints)){
      sg_write = sg[write_ints[i-1]:write_ints[i]]
      out_name = paste(outputFileName,'_',(i-1),sep='')
      
      # sg_2 = sg[-c(1:num_vars_pre_spat_lag)]; out_name_2 = paste(outputFileName,'_2',sep='')
      writeOGR(sg_write,save_dir, out_name, driver = 'ESRI Shapefile')
      writeSpatialShape(sg_write, paste(save_dir,'/',out_name,sep=''))
    # writeOGR(sg_2,workDir, out_name_2, driver = 'ESRI Shapefile')
    # writeSpatialShape(sg_2, out_name_2)
    } 
  } else {
    # sg_2 = sg[-c(1:num_vars_pre_spat_lag)]; out_name_2 = paste(outputFileName,'_2',sep='')
    writeOGR(sg,save_dir, outputFileName, driver = 'ESRI Shapefile')
    writeSpatialShape(sg, paste(save_dir,'/',outputFileName,sep=''))
  }
    
  
  ### Plot
  # plot(ra, col=ra@data$COLOUR)
}

importData = function(databaseRow){
  # this imports the data and assigns the attribute for what spatial data type it is
  workDir = getwd()
  fileType = databaseRow$fileType
  fileName = paste('Data/',databaseRow$FileName,databaseRow$fileType,sep='')
  # print(paste('importing: ',fileName,sep=''),row.names = FALSE)
  # different opening methods depending on filetype
  if (fileType == ".shp"){
    theData = readOGR(dsn = paste(workDir,'/Data',sep=''), layer = databaseRow$FileName,verbose = FALSE)
    # <if there is an error "incompatible geometry: 4", go into arcGIS and use multipart to singlepart tool and save>
  } else if (fileType == ".tif"){
    theData = raster(fileName)
  } else { # .csv
    theData = read.csv(fileName,header=T,fill = T,stringsAsFactors=FALSE)
  }
  attr(theData,'dataType') = databaseRow$dataType
  attr(theData,'varName') = databaseRow$varName
  attr(theData,'fileType') = databaseRow$fileType
  attr(theData,'fileName') = databaseRow$FileName
  
  
  return (theData)
}

createGrid = function(gridSize,database){
  ## takes the grid size input
  ## returns a grid which covers entire area
  
  # first use an area level data from database to define grid limits
  sf = importData(database[1,]) 
  
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
  # writeOGR(sg,workDir, outputFileName, driver = 'ESRI Shapefile') # writePolyShape(sg,outputFileName)
  # writes to csv
  
  return(sg)
}

areaInGrid = function(sg,sf,dataName,database,to_save = TRUE){
  # intersects the polygons with the grid and determines the area
  if (to_save){
    # check if a saved RData file already exists
    gridSize = attr(sg,'grid_size')
    gridded_filename = paste('Data/',dataName,'_gridsize_',gridSize,'.RData',sep='')
    alreadyProcessed = file.exists(gridded_filename)
    
    if (alreadyProcessed) {
      # if it does exist, load the data file
      load(gridded_filename)
    }  else {
      ## CREATE TEMPORARY GRID
      sg_temp = createGrid(gridSize,database)
      
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
      
      sf = sf_buffer(sf)
      # plot(sf)
      
      area_in_poly = function(j){
        p1 = grid_cells[j]
        proj4string(p1) = proj4string(sf)
        int = intersect(p1,sf)
        if (is.null(int)){
          area=0
        } else {
          area = int@polygons[[1]]@area
        }
        return(area)
      }
      
      # make a list of the polygons in the grid
      grid_cells = SpatialPolygons(sg_temp@polygons)
      # parallelise
      
      cl <- makeCluster(no_cores,outfile="")
      clusterExport(cl, c("sf","grid_cells","area_in_poly"), envir=environment())
      clusterEvalQ(cl, c(library(raster)))
      area_list = parLapply(cl,seq(1,length(grid_cells)),function(j) area_in_poly(j))
      stopCluster(cl)
      
      # add new covariate to dataframe
      sg_temp@data[,dataName]=unlist(area_list)
      # }
      
      # writes to shapefile
      # writeOGR(sg,workDir, outputFileName, driver = 'ESRI Shapefile') # writePolyShape(sg,outputFileName)
      # writes to csv
      
    }
    # remove excess fields
    sg_temp$area = NULL; sg_temp$cId = NULL; sg_temp@data$x = NULL; sg_temp@data$y = NULL; 
    save(sg_temp, file = gridded_filename)
  }
  
  ## JOIN THE SG
  sg@data = cbind(sg@data, sg_temp@data)
  
  return(sg)
}

areaCount = function(sg,sf=theData,database){
  # process area-level data by finding taking the area weighted sum.
  
  # check if a saved RData file already exists
  gridSize = attr(sg,'grid_size')
  dataName = attr(sf,'varName')
  gridded_filename = paste('Data/',dataName,'_gridsize_',gridSize,'.RData',sep='')
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
  # writeOGR(sg_temp,workDir, 'height_cells', driver = 'ESRI Shapefile') # writePolyShape(sg,outputFileName)
  # writes to csv
  
  return(sg)
}

pointPatternCount = function(sg,pointData){
  # process point pattern data
  # return a count of number of points within each grid cell
  dataName = attr(pointData,'varName')
  fileType = attr(pointData,'fileType')
  
  if (fileType == ".csv"){
    pointData = csvToSpatial(pointData)
  }
  # projection
  if (proj4string(sg) != proj4string(pointData)){
    pointData = spTransform(pointData,CRS(proj4string(sg)))
    # print(sf)
  }
  
  # overlay with polygons
  whichCid <- over(pointData, sg[,"cId"])
  
  # count and assign to sg
  counts = table(unlist(whichCid))
  sg@data[dataName] = 0*nrow(sg)
  sg@data[as.integer(rownames(counts)),dataName] = counts
  
  # writes to shapefile
  # writeOGR(pointData,workDir, 'pointData_test', driver = 'ESRI Shapefile')
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
  
  dataName = attr(distData,'varName')
  fileType = attr(distData,'fileType')
  dataType = attr(distData,'dataType')
  
  if (fileType == ".csv"){
    distData = csvToSpatial(distData)
  } else if (dataType == 'polyDist'){
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
  sg@data[dataName] = 0*nrow(sg)
  for (j in seq(nrow(sg))){
    # determine centroid of cell
    centroid = readWKT(paste("POINT(",toString(coordinates(sg[j,])[[1]]),toString(coordinates(sg[j,])[[2]]),")"))
    proj4string(centroid) = proj4string(sg)
    # sp2   <- SpatialPoints(centroid,proj4string=CRS(proj4string(sg)))
    
    if (dataType == 'polyDist'){
      # check to see if point is in polygon
      inPoly = gContains(distData,centroid)
      # print(inPoly)
      if (inPoly){
        # print('in poly')
        # distance is zero
        sg@data[j,dataName] = 0
      } else {
        sg@data[j,dataName] = min(gDistance(centroid, linesData, byid=TRUE))
      }
    } else{
      sg@data[j,dataName] = min(gDistance(centroid, distData, byid=TRUE))
    }
  }
  
  
  # writes to shapefile
  # writeOGR(sg,workDir, 'pointDataDist_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}

csvToSpatial = function(pointData){
  # the csv file needs a column 'Latitude' and a column 'Longitude'
  # pointData = read.csv('Data/pointPatternCount_example_vbcrime.csv',header=T,fill = T,stringsAsFactors=FALSE)
  # these columns need to be numeric, remove any NAs
  latCol = which(tolower(colnames(pointData))=='latitude' | tolower(colnames(pointData))=='lat')
  lonCol = which(tolower(colnames(pointData))=='longitude' | tolower(colnames(pointData))=='long' | tolower(colnames(pointData))=='lon'| tolower(colnames(pointData))=='lng')
  pointData[,latCol] = as.numeric(pointData[,latCol])
  pointData <- pointData[!is.na(pointData[,latCol]),] # this should remove all rows with NAs
  pointData <- pointData[!(pointData[,latCol] >= 999),] # this should remove all rows with 99999 for lat e.g. non conformant lat/lon
  pointData[,lonCol] = as.numeric(pointData[,lonCol])
  # converts to spatial points data frame
  coordinates(pointData) = c(lonCol,latCol)
  
  # projection(pointData) = crs(sg)
  proj4string(pointData) =  "+proj=longlat +datum=WGS84"
  
  return(pointData)
}

lineLength = function(sg,lineData){
  # returns the length of lines within the cell (e.g. bike lane)
  dataName = attr(lineData,'varName')
  fileType = attr(lineData,'fileType')
  
  # address projection
  lineData = spTransform(lineData, CRS(proj4string(sg)))
  # split into cells and find total length
  rp <- intersect(lineData, sg)
  rp$length <- gLength(rp, byid=TRUE) / 1000
  lengthCell <- tapply(rp$length, rp$cId, sum)
  # append to datastructure
  sg@data[dataName] = 0*nrow(sg)
  sg@data[as.integer(rownames(lengthCell)),dataName] = lengthCell
  
  return (sg)
}

spatialLag = function(sg){
  # a list of neighbours for each gridcell
  neighs = poly2nb(sg)
  dimSg = dim(sg)
  varNameList = colnames(sg@data)
  # loop through the data variables
  for (i in seq(5,dimSg[2])){
    # for each variable that is after the 4th (which are the presets (area, coordinates, id))
    if (class(sg@data[,i]) != 'character'){
      # not including strings
      varName = paste(varNameList[i],'_sl',sep='')
      for (j in seq(dimSg[1])){ 
        # loop through the cells
        sg@data[j,varName] = mean(sg@data[neighs[[j]],i])
        # I think a better way to do this is using one of the apply functions
      }
    }
  }
  return (sg)
}

processRaster = function(sg,rasterData = theData){
  # process raster data
  # determine the average of the values within the grid cell
  
  dataName = attr(rasterData,'varName')
  fileType = attr(rasterData,'fileType')
  
  # projection
  rasterData = projectRaster(rasterData,crs=CRS(proj4string(sg)))
  
  # clip raster to grid
  cropbox = extent(sg)
  rasterData = crop(rasterData,cropbox)
  
  # extract heat values for each cell from raster
  cell_vals = extract(rasterData,sg)
  cell = list()
  # Use list apply to calculate mean for each grid cell
  cell$mean = lapply(cell_vals, FUN=mean, na.rm=TRUE)
  cell$max = lapply(cell_vals, FUN=max, na.rm=TRUE)
  cell$min = lapply(cell_vals, FUN=min, na.rm=TRUE)
  # unlist, brings nested values into single list
  cell$mean = unlist(cell$mean)
  cell$max = unlist(cell$max)
  cell$min = unlist(cell$min)
  # Join values to polygon data
  # append to datastructure
  funs = c('mean','max', 'min')
  for (fun in funs){
    newVar = paste(dataName,'_',fun,sep='')
    sg@data[newVar] = 0*nrow(sg)
    sg@data[,newVar] = cell[fun]
  }
  # writes to shapefile
  # writeOGR(sg,workDir, 'raster_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}

categoriseRaster = function(sg,rasterData = theData,database){
  # process raster data
  # determine the average of the values within the grid cell
  
  dataName = attr(rasterData,'varName')
  fileType = attr(rasterData,'fileType')
  
  # this function takes a long time.
  # check if a saved RData file already exists
  gridSize = attr(sg,'grid_size')
  gridded_raster_filename = paste('Data/',dataName,'_gridsize_',gridSize,'.RData',sep='')
  alreadyProcessed = file.exists(gridded_raster_filename)
  print('grid size is:')
  print(gridSize)
  
  if (alreadyProcessed) {
    # if it does exist, load the data file
    load(gridded_raster_filename)
  }  else { # if not, process the data
    
    ## CREATE TEMPORARY GRID
    sg_temp = createGrid(gridSize,database)
    attr(sg_temp,'grid_size') = gridSize
    
    
    # loop through each of the categories in the raster
    if (strcmp(fileType,'.tif')){
      ## PROCESS THE RASTER
      ####
      # get cropbox for raster clip
      cropbox = extent(sg_temp)
      rasterData = crop(rasterData,cropbox)
      
      cats = unique(rasterData)
    } else {
      cats = unique(rasterData@data$Color)
    }
    for (catg in cats){
      varName = paste(dataName,catg,sep='_')
      
      ## Create polygons for each land cover
      # this function takes a long time.
      # check if a saved RData file already exists
      polyName = paste('Data/',varName,'-polygon','.RData',sep='')
      print(paste('processing ',varName,sep=''))
      alreadyProcessed = file.exists(polyName)
      if (alreadyProcessed) {
        # if it does exist, load the data file
        load(polyName)
      }  else {
        # extract the raster of just one category
        # create a polygon of the land cover type
        rast = rasterData
        if (strcmp(fileType,'.tif')){
          rast[rast != catg] = NA
          sf = gdal_polygonizeR(rast)
        } else {
          #           sf = rasterData
          #           sf@data = rasterData@data[rasterData@data$Color == catg]
          sf = SpatialPolygons(rasterData@polygons)[rasterData@data$Color == catg]
          proj4string(sf) = proj4string(rasterData)
          # print(catg)
          # plot(sf)
        }
        
        
        save(sf, file = polyName)
      }
      
      
      # determine the area in each grid of the raster and append to grid
      sg_temp = areaInGrid(sg_temp,sf,varName,database,TRUE)
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
  # writeOGR(sg,workDir, 'raster_test', driver = 'ESRI Shapefile')
  # writes to csv
  # write.csv(sg@data,paste(outputFileName,'.csv',''))
  return(sg)
}

geostat = function(sg,pointData = theData){
  # this function processes geostatistical data
  # it returns the min, max, median, and mean of the point geostat data
  
  dataName = attr(pointData,'varName')
  fileType = attr(pointData,'fileType')
  
  if (fileType == ".csv"){
    pointData = csvToSpatial(pointData)
  }
  # projection
  if (proj4string(sg) != proj4string(pointData)){
    pointData = spTransform(pointData,CRS(proj4string(sg)))
    # print(sf)
  }
  
  # make a list of the polygons in the grid
  grid_cells = SpatialPolygons(sg@polygons)
  
  # initialise data
  funs = c('mean','max', 'min','median')
  for (fun in funs){
    newVar = paste(dataName,'_',fun,sep='')
    sg@data[newVar] = 0*nrow(sg)
  }
  
  # loop through cIds
  for (j in sg@data$cId){
    # get the grid cell
    p1 = grid_cells[j]
    proj4string(p1) = proj4string(pointData)
    # get sub points
    sub_points = pointData[p1,]
    vals = strtoi(sub_points@data[,1])
    # loop through the functions
    sg@data[j, paste(dataName,'_mean',sep='')] = mean(vals, na.rm=TRUE)
    sg@data[j, paste(dataName,'_median',sep='')] = median(vals, na.rm=TRUE)
    sg@data[j, paste(dataName,'_max',sep='')] = max(vals, na.rm=TRUE)
    sg@data[j, paste(dataName,'_min',sep='')] = min(vals, na.rm=TRUE)
    
  }
  
  
  return(sg)
}


gdal_polygonizeR <- function(x, outshape=NULL, gdalformat = 'ESRI Shapefile',
                             pypath=NULL, readpoly=TRUE, quiet=TRUE) {
  ## a function from https://johnbaumgartner.wordpress.com/2012/07/26/getting-rasters-into-shape-from-r/
  if (isTRUE(readpoly)) require(rgdal)
  if (is.null(pypath)) {
    pypath <- Sys.which('gdal_polygonize.py')
  }
  if (!file.exists(pypath)) stop("Can't find gdal_polygonize.py on your system.")
  owd <- getwd()
  on.exit(setwd(owd))
  setwd(dirname(pypath))
  if (!is.null(outshape)) {
    outshape <- sub('\\.shp$', '', outshape)
    f.exists <- file.exists(paste(outshape, c('shp', 'shx', 'dbf'), sep='.'))
    if (any(f.exists))
      stop(sprintf('File already exists: %s',
                   toString(paste(outshape, c('shp', 'shx', 'dbf'),
                                  sep='.')[f.exists])), call.=FALSE)
  } else outshape <- tempfile()
  if (is(x, 'Raster')) {
    require(raster)
    writeRaster(x, {f <- tempfile(fileext='.tif')})
    rastpath <- normalizePath(f)
  } else if (is.character(x)) {
    rastpath <- normalizePath(x)
  } else stop('x must be a file path (character string), or a Raster object.')
  system2('python', args=(sprintf('"%1$s" "%2$s" -f "%3$s" "%4$s.shp"',
                                  pypath, rastpath, gdalformat, outshape)))
  if (isTRUE(readpoly)) {
    shp <- readOGR(dirname(outshape), layer = basename(outshape), verbose=!quiet)
    return(shp)
  }
  return(NULL)
}