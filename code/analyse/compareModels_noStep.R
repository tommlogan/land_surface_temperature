### This function takes a dataset, trains multiple models on it, using cross-validation, and returns the
### out-of-sample results.


# libraries
library(mgcv)
library(car)
library(tree)
library(randomForest)
library(gbm)
library(nnet)
library(earth)
# Sys.setenv(JAVA_HOME="C:/Program Files (x86)/Java/jre1.8.0_91") #C:/Program Files/Java/jdk1.8.0_25/jre")
# options(java.parameters = "- Xmx1024m")
# library(bartMachine)
library(glmnet)
source("tTestMulti.R")


### User inputs
data_filename = "complete_data_lst.RData"
working_drive = "F:/UrbanDataProject/land_surface_temperature/code"
setwd(working_drive)


compareModels = function(){
  #   rm(list=ls())
  setwd(working_drive)
  
  
  
  ### IMPORT DATA
  # import the data, recognise row and column headings. 
  # initial <- read.table(data_filename, header=T,fill = T,sep="\t", nrows = 100)
  # classes <- sapply(initial, class) # by telling R what the classes are of the dataframe it speeds up import
  # theData = read.table(data_filename,header=T,fill = T,sep="\t")
  load(data_filename)
  theData <- ds;
  # set the ID as the rowname
  rownames(theData)=theData$cId
  drops <- c("cId")
  theData <- theData[ , !(names(theData) %in% drops)]
  # theData=theData[,-'cId']
  
  # move the response to the LHS
  theData$response = theData$lst_mean
  drops <- c("lst_mean","lst_max","lst_min","lcov_0","lcov_0_sl","lst_mean_sl","lst_max_sl","lst_min_sl")
  theData <- theData[ , !(names(theData) %in% drops)]
  
  theData = na.omit(theData)
  namesData = names(theData)
  ### ANALYSE DATA
  # view the data
  # scatterplotMatrix(theData)
  
  ### lets test each variable to see if the F-stat says it's worth including:
  polyOrder = f_test_vars(theData[,-c(which(grepl("city",namesData)))])
  

    
  # investigate multicolinearity.
  #alias( lm( theData$response ~ ., data = theData))#[,c(5,7:28)]) )
  # there could be issues with multicollinearity between the temperature, rainfall, sun values.
  

  
  # Let's look at the variance inflation factors, start with 5
  glmVIF5 <<- vifVarSelect(theData)
  # therefore my vif5 formuli
  namesVif5 = names(glmVIF5$coefficients)[-1]  
  fmlaVif5 = as.formula(paste("response ~ ", paste(namesVif5,collapse="+")))
  polyIndx = vector(); for (i in 1:length(namesData)){if (sum(grepl(namesData[i],namesVif5))){polyIndx = c(polyIndx,i) } }
  polyOrdVif5 = polyOrder[polyIndx]
  polyOrdVif5 <- polyOrdVif5[!is.na(polyOrdVif5)]
  #   polyOrdVif5 = polyOrdVif5[!is.na(polyOrdVif5)]
  polyVif5 = data.frame(coeff=namesVif5,ord = polyOrdVif5); polyVif5 = polyVif5[-c(which(polyOrdVif5==0),which(is.na(polyOrdVif5))),]
  coeffs = paste(polyVif5$coeff,polyVif5$ord,sep=",")
  fmlaVif5poly = as.formula(paste("response ~ poly(", paste(coeffs,collapse=",raw=T) + poly("),",raw=T)"))
  fmlaVif5GAM = as.formula(paste("response ~ s(", paste(namesVif5[-c(4,9)],collapse=",k=3) + s("),",k=3)"))
  # this removes values with a VIF > 10
  glmVIF10 = vifVarSelect(theData,vifLim = 10)
  fmlaVif10 = as.formula(paste("response ~ ", paste(names(glmVIF10$coefficients)[-1],collapse="+")))
  
  # very few covariates are removed.
  
  # let's also PCA the data:
#   pcaData = prcomp(theData[,-1],scale=T)
#   pcaData = data.frame(pcaData$x)
#   pcaData$response = theData$response
  
  ## enter a holdout and begin fitting models:
  # the models that will be fitted are:
  modelNames = c("meanOnly", "glm1",  "gam1","mars1","tree1", "bag1", "rf1", "boost1", "nnet1", "glm2", "mars2","tree2", "bag2", "rf2", "boost2","nnet2")#,"glm3")
              #c("meanOnly", "glm1", "glm2","glm3","glm4", "glm5", "glm6", "glm8", "glmStep",  "gam1",       "tree1", "bag1", "rf1", "rf2","boost1","mars1","bart1","tree2", "bag2",         "rf3","boost2","mars2") 
  
### Random Holdout
  
  k = 5 # number of holdouts
  z = 0.20;  
  numModels = length(modelNames)
  
  
  # MSE initialise
  mseTable = matrix(data=NA, nrow = k+1, ncol = numModels)
  colnames(mseTable) = modelNames
  # MAE initialise
  maeTable = matrix(data=NA, nrow = k+1, ncol = numModels)
  colnames(maeTable) = modelNames
  # MPE initialise
  mpeTable = matrix(data=NA, nrow = k+1, ncol = numModels)
  colnames(mpeTable) = modelNames
  # R2 initialise
  R2Table = matrix(data=NA, nrow = k+1, ncol = numModels)
  colnames(R2Table) = modelNames
  
  numObs = dim(theData)[1]
#   numObsP = dim(pcaData)[1]
  
  
  for (i in 1:k) {
    # split the data
    randIndx = sample(1:numObs,numObs*z,replace=FALSE)
    testData = data.frame(theData[randIndx,])
    trainData <<- data.frame(theData[-randIndx,])
    # split pca
#     testDataPCA = data.frame(pcaData[yearIndx,])
#     trainDataPCA <<- data.frame(pcaData[-yearIndx,])
    
    # models
    pred_meanOnly = mean(trainData$response)
    
    # generalized linear models
    glm1 = glm(response ~ ., data = trainData)
    pred_glm1 = predict(glm1, newdata = testData, type = "response")
    
    glm2 = glm(fmlaVif5, data = trainData) #vif5
    pred_glm2 = predict(glm2, newdata = testData, type = "response")
    
    # glm3 = glm(fmlaVif5poly, data = trainData) #vif5
    # pred_glm3 = predict(glm3, newdata = testData, type = "response")
    # 
#     glm6 = glm(fmlaVif10,data = trainData) #vif10
#     glmPCA = glm(response ~ ., data = trainDataPCA) #pca one
#     pred_glmPCA = predict(glmPCA, newdata = testDataPCA, type = "response")
    
    # glmStep = step(glm(response ~. , data = trainData), direction = "both", k = 3.84)
    # pred_glmStep = predict(glmStep, newdata = testData, type = "response")
#     glm5 = step(glm(response ~. , data = trainDataPCA), direction = "both", k = 3.84)
#     glm8 = step(glm(fmlaVif5poly, data = trainData), direction = "both", k = 3.84) #vif5
    
    
  # generalized additive models
    gam1 = gam(fmlaVif5GAM, data=trainData)   
    pred_gam1 = predict(gam1, newdata = testData, type = "response")
    
    
    # tree based models
    tree1 = tree(response~., data=data.frame(trainData))
    cv1 = cv.tree(tree1)
    bestIndx1 = which.min(cv1$dev)
    bestSel1 = cv1$size[bestIndx1]
    prunedTree1 = prune.tree(tree1,best=bestSel1)
    pred_tree1 = predict(prunedTree1, newdata = testData)

    tree2 = tree(as.formula(paste("response ~ ", paste(names(glmVIF5$coefficients)[-1],collapse="+"))), data=data.frame(trainData))
    cv2 = cv.tree(tree2)
    bestIndx2 = which.min(cv2$dev)
    bestSel2 = cv2$size[bestIndx2]
    prunedTree2 = prune.tree(tree2,best=bestSel2)
    pred_tree2 = predict(prunedTree2, newdata = testData)
     
    bag1 = randomForest(response ~ ., data = trainData,  mtry = length(names(trainData))-1, importance = T) # mtry is = to number of variables
    pred_bag1 = predict(bag1, newdata = testData)
    
    bag2 = randomForest(fmlaVif5, data = trainData,  mtry = length(names(glmVIF5$coefficients)[-1]), importance = T)
    pred_bag2 = predict(bag2, newdata = testData)
    
    rf1 = randomForest(response ~ ., data = trainData, importance = T)
    pred_rf1 = predict(rf1, newdata = testData)
    
    rf2 = randomForest(fmlaVif5, data = trainData, importance = T)
    pred_rf2 = predict(rf2, newdata = testData)
    
#     rf3 = randomForest(fmlaVif5poly, data = trainData, importance = T)
#     rf2 = randomForest(response ~ ., data = trainData,  mtry = 22, importance = T)
  
    boost1 = gbm(response ~ ., data = trainData, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
    pred_boost1 = predict(boost1, newdata = testData,n.trees = 5000)
    
    boost2 = gbm(fmlaVif5, data = trainData, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
    pred_boost2 = predict(boost2, newdata = testData,n.trees = 5000)
    
    # Multivariate Adaptive Splines
    mars1 = earth(response ~ ., data = trainData)
    pred_mars1 = predict(mars1, newdata = testData)
    
    mars2 = earth(fmlaVif5, data = trainData)
    pred_mars2 = predict(mars2, newdata = testData)
    
    nnet1 = nnet(response~.,data=trainData, size=10, linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
    pred_nnet1 = predict(nnet1, newdata = testData)
    
    nnet2 = nnet(fmlaVif5,data=trainData, size=10, linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
    pred_nnet2 = predict(nnet2, newdata = testData)

    # options(java.parameters = "-Xmx5000m")
    # set_bart_machine_num_cores(4)
    # y = trainData$response
    # X = trainData; X$response = NULL
    # bart1 = bartMachine(X,y)
    # Xtest1 = testData; Xtest1$response = NULL
    # pred_bart1 = predict(bart1, Xtest1)
    
    #lasso1 = glmnet(X,y,alpha=1)
    #  cv.out1 = cv.glmnet(X,y,alpha=1)
    #  bestlam1 = cv.out1$lambda.min
    
    
    # add to results tables
    for (n in modelNames){
      # get the prediction data for the model
      predicted = get(paste('pred_',n,sep = ""))
      # calculate the MSE of the predictions
      mseTable[i,n] = mean((testData$response - predicted)^2)
      maeTable[i,n] = mean(abs(testData$response - predicted))
      mpeTable[i,n] = mean(abs(testData$response - predicted)/testData$response)
      R2Table[i,n] = sum((predicted-mean(testData$response))^2)/sum((testData$response-mean(testData$response))^2) 
    }

  }
  
  # now's time to average all these
  mseTable[k+1,] = colMeans(mseTable[-(k+1),])
  cat("mean MSEs are:", mseTable[k+1,],"\n")
  maeTable[k+1,] = colMeans(maeTable[-(k+1),])
  cat("mean MAEs are:", maeTable[k+1,],"\n")
  mpeTable[k+1,] = colMeans(mpeTable[-(k+1),])
  meanErrors = list("mseTable"= mseTable, "maeTable"=maeTable)
  
  # plot the bar charts 
  #par(mfrow=c(1,2))
  barMSE = barplot(mseTable[k+1,-c(9,16)],xlab="Models",ylab="mean square error",las=2)
  mse.means = colMeans(mseTable)
  std = colStdev(mseTable)
  error.bar(barMSE,mse.means, 1.96*std/sqrt(30))
  
  barMAE = barplot(maeTable[k+1,],xlab="Models",ylab="mean absolute error",las=2)
  mae.means = colMeans(maeTable)
  std = colStdev(maeTable)
  error.bar(barMAE,mae.means, 1.96*std/sqrt(30))
  

mseTable2 = mseTable[,-c(9,16)]#,6,10,12,18)]
maeTable2 = maeTable[,-c(9,16)]#
  # box plots
  boxplot(mseTable2,xlab="Models",ylab="mean square error",las=2)
  boxplot(maeTable2,xlab="Models",ylab="mean absolute error",las=2)
  boxplot(mpeTable[,-c(9,16)]*100,xlab="Models",ylab="mean percentage error",las=2)
  
  # copy errors
  write.table(mseTable[(k+1),],"clipboard",sep="\t")
  write.table(maeTable[(k+1),],"clipboard",sep="\t")
  
#   boxplot(mseTable[,c(3,11,27,22,23,17,25,28)],xlab="Models",ylab="MSE")
#   boxplot(maeTable[,c(3,11,27,22,23,17,25,28)],xlab="Models",ylab="MAE")
  mseTTest = tTestMulti(mseTable2)
  write.table(mseTTest,"clipboard",sep="\t")
  maeTTest = tTestMulti(maeTable)
  write.table(maeTTest,"clipboard",sep="\t")
  
  # plot models predictions vs. actual response values
#   glm8Pred = predict(glm8, newdata = theData)
#   mars2Pred = predict(mars2, newdata = testData)
#   rf3Pred = predict(rf3, newdata = testData)
#   actual = theData$response
#   plot(actual, glm8Pred, xlab="Actual mean weight", ylab="Predicted mean weight")
#   abline(0,1,lty=2)
#   plot(actual, mars2Pred, xlab="Actual mean weight", ylab="Predicted mean weight")
#   abline(0,1,lty=2)
#   plot(actual, rf3Pred, xlab="Actual mean weight", ylab="Predicted mean weight")
#   abline(0,1,lty=2)

# the best model is the glm8. so let's train it on all of the data.
#glm8 = step(glm(response ~ poly(weight5,3,raw=T) + colour5 + fleshFirm5 + brix5 + poly(spi5,2,raw=T) + weight11 + poly(colour11,2,raw=T) + fleshFirm11 + brix11 + spi11 + poly(weight12up,2,raw=T) + poly(colour12up,2,raw=T) + poly(fleshFirm12up,2,raw=T) + brix12up + spi12up + poly(rainAug,2,raw=T) + poly(rainNov,3,raw=T) + rainDec + rainJan + rainFeb, data = theData), direction = "both", k = 3.84)
glm8Pred = predict(glm8, newdata = theData)
actual = theData$response
plot(actual, glm8Pred, xlab="Actual mean weight", ylab="Predicted mean weight")
abline(0,1,lty=2)

# and the glmStep - which is simplier as it doesn't have the polynomial terms
glmStep = step(glm(response ~. , data = trainData), direction = "both", k = 3.84)
glmStepPred = predict(glmStep, newdata = theData)
actual = theData$response
plot(actual, glmStepPred, xlab="Actual mean weight", ylab="Predicted mean weight")
abline(0,1,lty=2)

# random forest
rfGood = randomForest(response ~ ., data = theData, importance = T)
rfGoodPred = predict(rfGood, newdata = theData)
actual = theData$response
plot(actual, rfGoodPred, xlab="Actual average size", ylab="Predicted average size",main="RF3")
abline(0,1,lty=2)
rfor.R2 = sum((rfGoodPred-mean(actual))^2)/sum((actual-mean(actual))^2)
rfor.MAE = mean(abs(actual - rfGoodPred))
rfor.MSE = mean((actual - rfGoodPred)^2)
rfor.MPE = mean(abs(actual - rfGoodPred)/actual)

# variable importance
varImpPlot(rfGood)

# neural net on entire set
actual = theData$response
nnet3 = nnet(response~.,data=theData,size=10, linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
nnet3Pred = predict(nnet3, newdata = theData)
plot(actual, nnet3Pred, xlab="Actual average size", ylab="Predicted average size",main="Nnet")
abline(0,1,lty=2)
net.R2 = sum((nnet3Pred-mean(actual))^2)/sum((actual-mean(actual))^2)
net.MAE = mean(abs(actual - nnet3Pred))
net.MSE = mean((actual - nnet3Pred)^2)
net.MPE = mean(abs(actual - nnet3Pred)/actual)
}

error.bar <- function(x, y, upper, lower=upper, length=0.1){
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length)
}

colStdev = function(data){
  # returns a row vector containing the standard deviation for each column
  nCol = dim(data)[2]
  stdevs = matrix(data=NA, nrow = 1, ncol = nCol)
  for (j in 1:nCol){
    stdevs[j] = sd(data[,j])
  }
  return(stdevs)
}

meanOnly = function(){
  for (i in 1:k) {
    # split the data
    randIndx = sample(1:numObs,numObs*z,replace=FALSE)
    testData = data.frame(theData[randIndx,])
    trainData <<- data.frame(theData[-randIndx,])
    
    # add to MSE table
    mseTable[i,"meanOnly"] = mean((testData$response - mean(trainData$response))^2)
    
    # add to MAE table
    maeTable[i,"meanOnly"] = mean(abs(testData$response - mean(trainData$response)))
    
  }
}

vifVarSelect <- function(theData,vifLim=5){
  # This function finds the VIF based on a GLM,then identifies and removes the variable with
  # the largest VIF above 5. Another GLM is fit and the process repeats until all the VIFs are below 5.
  
  indices = 2:dim(theData)[2] # indices of explan variables
  remIndcs = c()
  
  vifResult = vif(glm(theData$response~.,data=theData[,indices]))
  while (sum(vifResult>vifLim)>0) { # while any of the VIF values exceeds 5
    indx = which.max(vifResult) # which explanatory variable has the largest VIF
    indices = indices[-indx]
    remIndcs = union(remIndcs,indx)
    vifResult = vif(glm(theData$response~.,data=theData[,indices]))
    # print(indx)
  }
  glmVIF = glm(theData$response~.,data=theData[,indices])
  return(glmVIF)
}

f_test_vars = function(passedData){
  # lets test each variable to see if the F-stat says it's worth including:
  namesData = names(passedData)
  polyOrder = matrix(data=0, nrow = (length(namesData)-1), ncol = 1)
  for (n in 2:length(namesData)){
    o = 3
    
    while (o > 0 & polyOrder[n-1]==0) {
      fmla = as.formula(paste(namesData[1],"~ poly(",namesData[n],",",o,", raw=TRUE)"))
      lm1 = lm(fmla,data=passedData)
      if (o < (dim(summary(lm1)$coefficients)[1])) {
        pVal = summary(lm1)$coefficients[o+1,4]
        if (pVal < 0.05){
          polyOrder[n-1] = o
        }
      }
      o = o - 1
    }
  }
  return(polyOrder)
}
  

holdout = function(theData,z){
  # randomly select donors for test/train sets
  numObs = nrow(theData)
  randIndx = sample(1:numObs,numObs*z,replace=FALSE)
  
  
  set = list()
  set$train = data.frame(theData[-randIndx,])
  set$test = data.frame(theData[randIndx,])
  # check to see if all factor levels are present in training set
  factorVars = lapply(theData,is.factor)
  i = 0
  for (tf in factorVars){
    i = i + 1
    if (tf){
      # only considering variables that are factors
      # are all levels present?
      dataLevels = levels(theData[,i])
      trainLevels = list(); for (j in 1:length(summary(set$train[,i]))){if (summary(set$train[,i])[j]) {trainLevels = cbind(trainLevels,(names(summary(set$train[,i]))[j]))}}
      notOk = length((setdiff(dataLevels,trainLevels)))
      if (notOk){
        # there are more levels in the data then the training set.
        # need to resample
        set = holdout(theData,z)
        break
      }
    }
  }
  return(set)
}