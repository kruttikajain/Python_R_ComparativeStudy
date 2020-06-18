#DNN for CSE210A
library('dplyr')
library('keras')
library('tensorflow')
library('keras')
library('ggplot2')
library('caret')

train_data <- read.csv("/Users/kruttikajain/Documents/train.csv", na.strings = c("", "NA"))
test_data <- read.csv("/Users/kruttikajain/Documents/test.csv")

# Data summary

summary(train_data)
# 60% of first class pass. have survived, and third class only a quarter
tapply(train_data$Survived,train_data$Pclass,mean)



# Embarked encoding.
tapply(train_data$Survived,train_data$Embarked,mean)


#removing non predictors
training_data <- subset(train, select = -c(PassengerId, Name, Ticket, Cabin))
training_data$Pclass <- as.factor(training_data$Pclass)

# dealing with nanvalues
c(sum(is.na(train_data$Survived)),sum(is.na(train_data$Pclass)),sum(is.na(train_data$Sex)),sum(is.na(train_data$Age)),
  sum(is.na(train_data$SibSp)),sum(is.na(train_data$Parch)),sum(is.na(train_data$Fare)),sum(is.na(train_data$Embarked)))
nanEmb <- is.na(training_data$Embarked)
training_data <- training_data[!nanEmb ,]

#handling Nanfor Age and Fare 
training_data.t <- select(training_data, -(Survived))
preObj <- preProcess(training_data.t, method = "knnImpute")
age <- predict(preObj, training_data.t)$Age
age.test <- predict(preObj, test_data)$Age
fare <- predict(preObj, training_data.t)$Fare
fare.test <- predict(preObj, test_data)$Fare

training_data$Age <- age
training_data$Fare <- fare
test_data$Age <- age.test
test_data$Fare <- fare.test

dummies <- dummyVars( ~., data=training_data)
training_data <- data.frame((predict(dummies, newdata = training_data)))
test_data$Pclass <- as.factor(test_data$Pclass)
dummies.test <- dummyVars( ~., subset(test_data, select = -c(Name, Ticket, Cabin)))
test_imputed <- data.frame((predict(dummies.test, newdata = subset(test, select = -c(Name, Ticket, Cabin)))))


#Testing and validation
inTrain <- createDataPartition(y=training_data$Survived, p=0.7, list = FALSE)
training_DNN <- training_data[inTrain,]
testing_DNN <- training_data[-inTrain,]
trainingDNN_y <- as.matrix(training_DNN$Survived)
testingDNN_y <- as.matrix(testing_DNN$Survived)
trainingNN_x <- as.matrix(subset(training_DNN, select = -c(Survived)))
testingDNN_x <- as.matrix(subset(testing_DNN, select = -c(Survived)))
dimnames(trainingNN_x) <- NULL
dimnames(trainingNN_y) <- NULL
trainingDNN_y <- to_categorical(trainingDNN_y,2)
testingDNN_y <- to_categorical(testingDNN_y,2)

# Model training 
set.seed(100)

model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 24, activation = 'relu', input_shape = c(12)) %>% 
    layer_dense(units = 2, activation = 'softmax')

summary(model)

model %>% compile(
    loss = 'binary_crossentropy', 
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)
history <- model %>% fit(
    as.matrix(trainingNN_x), as.matrix(trainingNN_y), 
    epochs = 10, batch_size = 10, 
    validation_split = 0.2
)
plot(history)


