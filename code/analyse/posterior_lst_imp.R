# Hierarchical model with OpenBugs


# library(R2OpenBUGS)
library(plyr)
library(coda)
library(MASS) 
library(ggplot2)
library(rstan)
set.seed(20161211)
setwd('F:/UrbanDataProject/land_surface_temperature/code')
# import the data
data.filename = "complete_data_lst.RData"
load(data.filename)
ds <- ds[,c('lst_mean','imp_mean','albd_mean','city')]

# do a bayesian regression model for the entire dataset with a non informative prior.
# explanatory variable
dat <- ds[,c('imp_mean')]
# model
X <- model.matrix(~imp_mean, ds)
# regression slopes
betas <- runif(2,-1,1)
# response var
sigma <- sd(ds$lst_mean)
y <- ds[,c('lst_mean')]
# matrix for predictions
new_X <- model.matrix(~imp_mean,expand.grid(imp_mean = seq(min(ds$imp_mean),max(ds$imp_mean),length = 20)))
# data inputs
data.inputs <- list(K=4,y=y,X=X,new_X=new_X, N = length(ds$lst_mean), N2 = length(new_X))
# fit the model
cities.sim <- stan(file = 'lst_regression_imp.stan', data = data.inputs,
                   pars = c('beta','sigma','y_pred'))
# get the distribution from the posterior of the impervious surface coefficient
# use that as my prior for each of the Bayesian regressions for each city.

# do a bayesian regression for each city and plot the posterior distributions for each of
# coefficients for impervious surface


# include that plot in my presentation

# 
# 
# 
# # get mean and sd from each city
# cities <- ddply(ds,~city,summarise,mean=mean(lst_mean),sd=sd(lst_mean))
# 
# # prepare data inputs
# data <- list(imp = ds$imp_mean, Y = ds$lst_mean, n = length(ds$lst_mean))
# 
# inits <- function(){
#   list(beta0 = 0, beta1 = 1, tau = 1)
# }
# 
# cities.sim <- bugs(data, inits, model.file = 'lst_imp_coef_bug.txt',
#                    parameters = c('beta0','beta1','tau','sigma'),
#                    n.chains = 3, n.iter = 10000, debug = FALSE,save.history = TRUE)
# cities.coda.enab <- bugs(data, inits, model.file = 'lst_imp_coef_bug.txt',
#                     parameters = c('theta','mu.theta','sigma.theta'),
#                     n.chains = 3, n.iter = 10000, codaPkg=TRUE)
# cities.coda <- read.bugs(cities.coda.enab)
# library(lattice)
# library(coda) 
# xyplot(cities.coda)
# 
# # plot the bayesian distributions
# city.names <- c('intercept','gradient')
# df = data.frame()
# i0 <- 0
# for (j in seq(2,length(city.names))){
#   # load data
#   newdata = data.frame(lst = cities.sim$sims.matrix[,j], city = city.names[j])
#   # add data to dataframe
#   df <- rbind.fill(df,newdata)
#   # add cityname as variable
#   # i1 <- dim(ds)[1]
#   # ds$city[seq(i0,i1)] <- city
#   # i0 <- dim(ds)[1] + 1
# }
# plt.lst.hist <-ggplot(df,aes(lst, fill = city, colour = city)) + geom_density(alpha = 0.1) + labs(x = bquote('relationship % impervious surface \n and land surface temp')) +
#   theme(text = element_text(size = 40), legend.key.size = unit(2, "cm"))  
# ggsave('lst_imp_slope_posterior.png',plot = plt.lst.hist, dpi = 500, width = 20, height = 12.36)
# 
# plt.trace <- ggplot(df,aes(y = lst, x = as.numeric(rownames(df)))) + geom_point() + labs(x = "index", y = "gradient") + theme(text = element_text(size = 40), legend.key.size = unit(2, "cm"))
# ggsave('lst_imp_slope_trace.png',plot = plt.trace, dpi = 500, width = 20, height = 12.36)
# 
# gelman.plot(cities.coda)
