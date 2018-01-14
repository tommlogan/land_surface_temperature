# Hierarchical model with OpenBugs


library(R2OpenBUGS)
library(plyr)
library(coda)
library(MASS) 
library(ggplot2)
setwd('F:/UrbanDataProject/land_surface_temperature/code')
# import the data
data.filename = "complete_data_lst.RData"
load(data.filename)
ds <- ds[,c('lst_mean','imp_mean','albd_mean','city')]

# get mean and sd from each city
cities <- ddply(ds,~city,summarise,mean=mean(lst_mean),sd=sd(lst_mean))

# prepare data inputs
J <- 5
y <- cities$mean
sigma.y <- cities$sd
data <- list("J", "y", "sigma.y")

inits <- function(){
  list(theta = rnorm(J, 0, 100), mu.theta = rnorm(1, 0, 100), sigma.theta = runif(1, 0, 100))
}

cities.sim <- bugs(data, inits, model.file = 'cities_bugs.txt',
                   parameters = c('theta','mu.theta','sigma.theta'),
                   n.chains = 3, n.iter = 10000)
cities.coda.enab <- bugs(data, inits, model.file = 'cities_bugs.txt',
                    parameters = c('theta','mu.theta','sigma.theta'),
                    n.chains = 3, n.iter = 1000, codaPkg=TRUE)
cities.coda <- read.bugs(cities.coda.enab)

xyplot(cities.coda)

# plot the bayesian distributions
city.names <- c('baltimore', 'detroit', 'portland', 'chicago', 'losangeles')
df = data.frame()
i0 <- 0
for (j in seq(1,length(cities))){
  # load data
  newdata = data.frame(lst = cities.sim$sims.matrix[,j], city = city.names[j])
  # add data to dataframe
  df <- rbind.fill(ds,newdata)
  # add cityname as variable
  # i1 <- dim(ds)[1]
  # ds$city[seq(i0,i1)] <- city
  # i0 <- dim(ds)[1] + 1
}
plt.lst.hist <-ggplot(ds,aes(lst, fill = city, colour = city)) + geom_density(alpha = 0.1) + labs(x = bquote('normalised land surface temperature ('^o~C*')')) +
  theme(text = element_text(size = 40), legend.key.size = unit(2, "cm"))  
ggsave('lst_posterior.png',plot = plt.lst.hist, dpi = 500, width = 20, height = 12.36)
