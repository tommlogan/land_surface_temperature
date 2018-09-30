# Land Surface Temperature
Tom M Logan  
www.tomlogan.co.nz

## Description:
Understanding the factors influencing urban land surface temperature during the night and day.

## Cities analysed:
* [x] Baltimore (bal)
* [ ] Detroit (det)
* [ ] Phoenix (phx)
* [ ] Portland (por)
* [ ] ?Atlanta (atl)

## Steps:
1. Collect the raw data
1. Process the LandSat images to land surface temperatures
2. Prepare other images (tree canopy, land cover, impervious surface)
3. Grid the data for analysis as necessary
2. Statistical analysis

### 1. Raw data
#### Nighttime Light Intensity
This will be gridded with the nearest raster point.

### 1. LandSat images to LST, albedo, and NDVI:

#### 1.1 Download the satellite images
I selected the most recent four/five images per city and day/night where there was no cloud cover over the city of interest.

  1. Website: https://earthexplorer.usgs.gov/
  2. Enter Search Criteria:
      1. Address/Place - type the city name and state, e.g. 'Baltimore, MD' - click on the correct address and it should appear on the map.
      2. Date Range - we want to use summer dates
        I'm looking at years 2013-2017 and use months of May (05/01) - September (so up until 10/01) inclusive
        Need to find images that don't have too much cloud cover.
  3. Data Sets
      1. Select Landsat -> Landsat Collection 1 Level-1 -> Landsat 8 OLI/TIRS C1 Level-1
  4. Additional Criteria
      1. Sensor Identifier: OLI_TIRS
      2. Data type level-1: All
      3. Day/Night Indicator: Select relevant
      4. Cloud Cover: I leave these blank because I care about the cloud cover of the city, rather than the image and it's possible to have an image with high cloud cover but a clear sky above the city.
  5. View each image in turn and select ones with low cloud cover of the city

      When an image is selected
      * downloaded the Level-1 GeoTIFF Data Product
      * added to the `data/raw/<city>` directory
  6. Land Cover (NLCD) data for 2011 was downloaded using https://viewer.nationalmap.gov and the state the city is in was downloaded.
  7. Impervious surface and tree canopy was downloaded from https://viewer.nationalmap.gov as well.
  8. The shapefile of the city was downloaded from [catalog.data.gov](https://catalog.data.gov/dataset?collection_package_id=89f89c6f-741c-4121-98e3-d3f1f528ff53) dataset of city boundaries (for Baltimore, the others were the shapesfiles from green space.)
  9. The elevation (m) 1/3 arc second as downloaded from the same as 6 and 7. In some cases it had to be unioned: https://support.esri.com/en/technical-article/000015258

#### 1.2 Metadata
  `data_source_satellite.csv` in `/data` provides information from each of the raw satellite images necessary for them to be processed. <br>
  `data_source_city.csv` in `/data` records the location of city specific data such as the land cover and tree canopy
  1. As data is downloaded, add it to the csv
  2. The maximum daily temperature (in Celsius) for the day needs to be retrieved from https://www.wunderground.com/history/?MR=1

#### 1.3 Process satellite images to LST, albedo, NDVI
This generally follows the process described in [Sahana, M., Ahmed, R., & Sajjad, H. (2016). Analyzing land surface temperature distribution ... *Modeling Earth Systems and Environment.*](https://www.researchgate.net/publication/301797360_Analyzing_land_surface_temperature_distribution_in_response_to_land_useland_cover_change_using_split_window_algorithm_and_spectral_radiance_model_in_Sundarban_Biosphere_Reserve_India)
  1. The code `L8_processing.py` takes the raw satellite images and land cover images and calculates the surface temperature, albedo, and ndvi.
  2. In doing so, `L8_processing.py` calls the function `clip_geographic_data.R` which is an R function that takes the raw images and clips them to the city size. The output of this is satellite and land cover images which are the clipped to the city limit (with 2km buffer). These are saved in `data/intermediate/<city>`.

      I may need to come back to [this link](https://gis.stackexchange.com/questions/103166/simplest-way-to-limit-the-memory-that-the-raster-package-uses-in-r) if I run into further raster memory issues during projection.
  3. The final images are saved in `data/processed/image/<city>`

#### 1.4 Calculate mean LST, albedo, NDVI
  This is done within running `code/processing/L8_processing.py`

  This is a plot of the mean LST for Baltimore
    ![image](fig/map/lst_day_mean.jpg)

    30m resolution

#### 1.5 Job density
This data is available [here](https://lehd.ces.census.gov/data/lodes/LODES7/md/wac/). The WAC data provides work locations. JT01 is for primary jobs. S000 is for total number of jobs with no filters on age or income. The codebook is [here](https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.3.pdf).

#### 1.6 Building floor area shapefile
https://github.com/Microsoft/USBuildingFootprints
This is then clipped to the city boundary with `processing\footprint.R`.


#### 1.7 Lidar
I could estimate the  
* mean building height: https://developmentseed.org/blog/2014/08/07/processing-lidar-point-cloud/
* or, i could just get an estimate of mean lidar height (which is an estimate of development)

When a lidar point cloud is available (like in the case of Baltimore) it is imported as with `Create LAS Dataset` and then using the `LAS Dataset to Raster` tool, a raster is created.
When a set of tif tiles is available (like from NOAA), create a new raster catalog (left click on a geodb in ArcMap), add the files to the dataset, then using the `Raster Catalog to Raster Dataset` tool.

The sky view factor is calculated using the DSM in the code `processing\calc_svf.R`.


| City | Source |  Year |
| Baltimore | http://imap.maryland.gov/Pages/lidar-download-files.aspx | |
| Phoenix | https://asu.maps.arcgis.com/apps/Embed/index.html?webmap=a7da66e467bd43ed8d5f686637ec0ee5&extent=-112.6892,33.1391,-111.3331,33.8258&home=true&zoom=true&scale=false&search=true&searchextent=true&details=true&legendlayers=true&active_panel=legend&disable_scroll=false&theme=dark | |
| Portland | https://gis.dogami.oregon.gov/maps/lidarviewer/ | |
| Detroit | https://coast.noaa.gov/htdata/lidar1_z/geoid12a/data/4809/ | 2009 |

https://gisgeography.com/top-6-free-lidar-data-sources/

### 2 Prepare land cover, tree canopy, impervious surface, elevation data
This actually occurs during the code `clip_geographic_data.R` which is called during the previous step.

### 3 Grid data for analysis
I modified the code from www.github.com/tommlogan/spatial_data_discretiser: `code/processing/discritiser.R`
  2. add information into the file `code/processing/data_to_grid.csv`
  3. this must include the epsg projection reference for the appropriate state plane in meters: e.g. http://www.spatialreference.org/ref/?search=Maryland
  4. run the code's function `main('data_to_grid.csv')`
  5. I think I should remove the area column and instead turn the lcov_# variables into a percentage of the area - which i've done somewhere...

### 4 Exploratory data analysis
  1. See the Jupyter notebook `explore.ipynb`

### 5 Statistical inference on the dataset:
  1. Fit some quick regressions to this data to look at trends and see if there are any patterns or trends emerging
  2. Look at the variable importance and compare between the diurnal and nocturnal temperatures



## Cook book: Variable descriptions

    Table 1: variables collected and their units

| var-name      | Description   | Unit  | Source |
| ------------- |:-------------| -----:| --- |
| lst_day_mean  | land surface temperature during the day (averaged over different images ) | oC |
| lst_night_mean  | as above, but night images      |   oC |
| alb_mean | albedo during the day, averaged over different images      |    an index (divide by 100 to normalize) |
| ndvi_mean | vegetation index during the day, averaged over different images    |    |
| tree | tree canopy cover   |     |
| imp | % impervious surface |   % [0,1] |
| lc_# | area within grid cell that is of land cover type # (see https://www.mrlc.gov/nlcd11_leg.php for the number definitions)     | m2 |
| ntl_mean | nighttime light | | https://www.ngdc.noaa.gov/eog/dmsp/downloadV4composites.html |
| job_density | | | https://lehd.ces.census.gov/data/lodes/LODES7/md/wac/ |
