library(ggplot2)
library(dplyr)

#Read CSV
data<-read.csv("/Users/kruttikajain/Documents/pima-indians-diabetes.csv")
data_dfX<-data.frame(data)
data_dfY<-data.frame(data)
data_dfX$X1 <- NULL
print(data_dfX)
data_dfY$X6 <- NULL
data_dfY$X148 <- NULL
data_dfY$X72 <- NULL
data_dfY$X35 <- NULL
data_dfY$X0 <- NULL
data_dfY$X33.6 <- NULL
data_dfY$X0.627 <- NULL
data_dfY$X50 <- NULL
print(data_dfY)

#Defining sigmoid function
sigmoid_func <- function(z){1/(1+exp(-z))}

#Defining Cost function
cost_func <- function(theta, dataX, datay){
  a <- length(datay) # 
  b <- sigmoid(dataX %*% theta)
  cost_func <- (t(-datay)%*%log(b)-t(1-datay)%*%log(1-b))/a
  cost_func
}

#Defining Gradient function
grad_func <- function(theta, dataX, datay){
  a <- length(datay) 
  b <- sigmoid(dataX%*%theta)
  grad <- (t(dataX)%*%(b - datay))/a
  grad
}

logisticReg <- function(dataX, datay){
  dataX <- na.omit(dataX)
  datay <- na.omit(datay)
  dataX <- as.matrix(dataX[, c(ncol(dataX), 1:(ncol(dataX)-1))])
  datay <- as.matrix(datay)
  #theta initial weights 
  theta <- matrix(rep(0, ncol(dataX)), nrow = ncol(dataX))
  #Optimizing cost
  costOpti <- optim(theta, fn = cost, gr = grad, dataX=X, datay=y)
  return(costOpti$par)
}

log_Prob <- function(theta, dataX){
  dataX <- na.omit(dataX)
  dataX <- mutate(dataX, bias =1) #bias
  dataX <- as.matrix(dataX[,c(ncol(dataX), 1:(ncol(dataX)-1))])
  return(sigmoid(dataX%*%theta))
}

log_Pred <- function(prob){
  return(round(prob, 0))
}

# Data training
theta <- logisticReg(data_dfX, data_dfY)
prob <- log_Prob(theta, data_dfX)
pred <- log_Pred(prob)

## built in LR
db = read.csv('/Users/kruttikajain/Documents/pima-indians-diabetes.csv', header=TRUE)
require(caTools)
set.seed(3)
sample = sample.split(db$X1, SplitRatio=0.80)
train = subset(db, sample==TRUE)
test = subset(db, sample==FALSE)

#Fit on all
nt<-proc.time()
All_x <- glm(X1 ~ ., data = train, family=binomial(link="logit"))
proc.time()-nt
summary

PredictTest <- predict(All_x, type = "response", newdata = test)
test_tab <- table(test$X1, PredictTest > 0.5)
test_tab

accuracy_test <- round(sum(diag(test_tab))/sum(test_tab),2)
sprintf("Accuracy obtained", accuracy_test)

##Naive Bayes
library(e1071)
ntx<-proc.time()
model = naiveBayes(X1 ~ . , data = train)
proc.time()-ntx

prednb <- predict(model, test)
table(prednb,test$X1) 

