###
# Cluster data on impervious surface
# 




library(ggplot2)
# import the data
baltimoreFile <- "F:/UrbanDataProject/cities/Baltimore/gridded_data/2016-07-08/baltimoreData.RData"
detroitFile <- "F:/UrbanDataProject/cities/Detroit/detroitData.RData"
load(baltimoreFile)
load(detroitFile)
imp.data.bal <- baltimoreData[,c('imp_mean','lst_mean')]
imp.data.det <- detroitData[,c('imp_mean','lst_mean')]
# standardise lst data
imp.data.bal$lst_mean <- imp.data.bal$lst_mean - mean(imp.data.bal$lst_mean,na.rm=TRUE)
imp.data.det$lst_mean <- imp.data.det$lst_mean - mean(imp.data.det$lst_mean,na.rm=TRUE)
imp.data <- rbind(imp.data.bal,imp.data.det)
imp.data = imp.data[complete.cases(imp.data),]

# do the clustering
clus <- kmeans(imp.data$imp_mean,centers = 4)

# plot the clustering
plot(imp.data,col=c(1:4)[clus$cluster])
