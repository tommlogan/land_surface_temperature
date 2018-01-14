tTestMulti = function(dataTable){
  numDists = dim(dataTable)[2] # number of distributions to test
  # t-test table initialise
  ttestTable = matrix(data=NA, nrow = numDists, ncol = numDists)
  colnames(ttestTable) = colnames(dataTable);   rownames(ttestTable) = colnames(dataTable)
  for (j in 1:numDists){
    for (i in j:numDists){
      if (i != j){
       testResult = t.test(dataTable[,j],dataTable[,i],alternative="two.sided", var.equal=T)
       ttestTable[i,j] = testResult$p.value
      }
    }
  }
 
  library(reshape2)
  pvals = melt(ttestTable)
  pvals = na.omit(pvals)
  pvals$value = p.adjust(pvals$value,"bonferroni")
  numPs = dim(pvals)[1]
  for (k in 1:numPs){
    colT = pvals[k,2]
    rowT = pvals[k,1]
    ttestTable[rowT,colT] = pvals[k,3]
  }
    
  return (ttestTable)
}