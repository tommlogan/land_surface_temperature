# Analyse the land surface temperature data

# Tom Logan

# import libraries
library(plyr)
library(ggplot2)
Sys.setenv(JAVA_HOME="C:/Program Files (x86)/Java/jre1.8.0_111")
options(java.parameters = "- Xmx1024m")
library(bartMachine)
# library(bnlearn)



# import the data
setwd('F:/UrbanDataProject/land_surface_temperature/code')
cities <- c('baltimore', 'detroit', 'portland', 'chicago', 'losangeles')
ds = data.frame()
i0 <- 0
for (city in cities){
  # load data
  load(paste('../data/',city,'_data.RData',sep=""))
  newdata <- get(paste('data.',city,sep = ""))
  # remove zeros
  newdata <- newdata[newdata$area > 0,]
  # normalise the LST
  newdata$lst_mean <- (newdata$lst_mean - mean(newdata$lst_mean))/sd(newdata$lst_mean)
  newdata$lst_max <- (newdata$lst_max - mean(newdata$lst_max))/sd(newdata$lst_max)
  newdata$lst_min <- (newdata$lst_min - mean(newdata$lst_min))/sd(newdata$lst_min)
  newdata$lst_mean_sl <- (newdata$lst_mean_sl - mean(newdata$lst_mean_sl))/sd(newdata$lst_mean_sl)
  newdata$lst_max_sl <- (newdata$lst_max_sl - mean(newdata$lst_max_sl))/sd(newdata$lst_max_sl)
  newdata$lst_min_sl <- (newdata$lst_min_sl - mean(newdata$lst_min_sl))/sd(newdata$lst_min_sl)
  # add data to dataframe
  ds <- rbind.fill(ds,newdata)
  # add cityname as variable
  i1 <- dim(ds)[1]
  ds$city[seq(i0,i1)] <- city
  i0 <- dim(ds)[1] + 1
}
# what are the lcov columns
lcov.indx <- grepl('lcov',colnames(ds))
lcov.complete <- ds[,lcov.indx]
lcov.complete[is.na(lcov.complete)] <- 0
ds[,lcov.indx] <- lcov.complete
# remove incomplete columns
ds <- ds[ lapply( ds, function(x) sum(is.na(x)) / length(x) ) < 0.1 ]
# convert city to a factor
ds$city <- factor(ds$city)
# remove incomplete rows
ds <- ds[complete.cases(ds),]
# plot the histogram of LST mean
ggplot(ds,aes(lst_mean, fill = city, colour = city)) + geom_density(alpha = 0.1) + labs(x = bquote('normalised land surface temperature ('^o~C*')')) +
  theme(text = element_text(size = 20))
# ggsave('lst_histogram.png',plot = last_plot(), dpi = 500)
# output a txt file
# write.table(ds,"lst_data.txt",sep="\t",row.names=FALSE)
save(ds,file = 'complete_data_lst.RData')

# fit a Bayesian Regression Tree to the data to predict the lst
ds.variables <- c("lst_mean", "lst_max","lst_min",
                  # "x","y",
                  "elev_mean","elev_max","elev_min",
                  "imp_mean","imp_max","imp_min",
                  "tree_mean","tree_max", "tree_min",
                  "lcov_11","lcov_21","lcov_22",  "lcov_23","lcov_24","lcov_31","lcov_41","lcov_42","lcov_43","lcov_52","lcov_71",     
                  "lcov_90",  "lcov_95",
                  "albd_mean","albd_max","albd_min",
                  "nvdi_mean","nvdi_max","nvdi_min")
                  # "lst_mean_sl", "lst_max_sl",   "lst_min_sl", "elev_mean_sl", "elev_max_sl",  "elev_min_sl",  "imp_mean_sl",  "imp_max_sl",   "imp_min_sl",   "tree_mean_sl",
                  # "tree_max_sl",  "tree_min_sl","lcov_11_sl",   "lcov_21_sl",   "lcov_22_sl",   "lcov_23_sl",   "lcov_24_sl",   "lcov_31_sl",   "lcov_41_sl",  
                  # "lcov_42_sl",   "lcov_43_sl", "lcov_52_sl",   "lcov_71_sl",   "lcov_90_sl",   "lcov_95_sl",   "albd_mean_sl", "albd_max_sl",  "albd_min_sl", 
                  # "nvdi_mean_sl", "nvdi_max_sl","nvdi_min_sl","city", "lcov_0", "lcov_81","lcov_82","lcov_0_sl",    "lcov_81_sl",  
                  # "lcov_82_sl")
# bn.gs <- cextend(gs(ds[,ds.variables]))
# bn.rsmax2 <- rsmax2(ds[,ds.variables])
# bn.tan <- tree.bayes(ds[,ds.variables])

# fit a BART


