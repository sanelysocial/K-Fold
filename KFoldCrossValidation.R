install.packages("titanic")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("DAAG")
install.packages("caret")

install.packages("rpart")
install.packages("gmodels")
install.packages("Hmisc")
install.packages("ResourceSelection")
install.packages("lattice")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("InformationValue")
library(caret)
library(rpart)
library(gmodels)
library(Hmisc)
library(ResourceSelection)
library(lattice)
library(ggplot2)
library(dplyr)
library(InformationValue)


library(titanic)
library(rpart.plot)
library(gmodels)
library(Hmisc)
library(pROC)
library(ResourceSelection)
library(car)
library(caret)
library(plyr)
library(dplyr)
library(InformationValue)
library(rpart)
library(randomForest)
library(DAAG)

cat("\014") #Clearing the screen

getwd()
setwd("C:/YYYYYY/AMMA 2017/Data/data_2017/titanic") #This working directory is the folder where all the bank data is stored

#Train titanic dataset to remove unecessary variables
titanic_data_raw<-read.csv('train.csv')
titanic_data_imp <- subset(titanic_data_raw, select = c(2,3,5:8,10))
titanic_data_imp$Survived=as.factor(titanic_data_imp$Survived)
titanic_data_imp$Pclass=as.factor(titanic_data_imp$Pclass)

# Treating missing values of Age with the mean of Age
titanic_data_imp$Age[is.na(titanic_data_imp$Age)] <- mean(titanic_data_imp$Age, na.rm = T)
summary(titanic_data_imp)

set.seed(6291)

#Performing manual partitioning for GLM
Train <- createDataPartition(titanic_data_imp$Survived, p=0.7, list=FALSE)
titanic_traindata <- titanic_data_imp[ Train, ]
titanic_testdata <- titanic_data_imp[ -Train, ]

titanic_model <- train(Survived ~ Pclass + Sex + Age + SibSp + 
                   Parch + Fare,  data=titanic_traindata, method="glm", family="binomial")

predictors(titanic_model)

#Converting log values to exponetial values
exp(coef(titanic_model$finalModel))

predict(titanic_model, newdata=titanic_testdata)
predict(titanic_model, newdata=titanic_testdata, type="prob")

varImp(titanic_model)

titanic_model_rmParch <- train(Survived ~ Pclass + Sex + Age + SibSp +
                           Fare,  data=titanic_traindata, method="glm", family="binomial")
titanic_model_rmParch

pred = predict(titanic_model, newdata=titanic_testdata)
accuracy <- table(pred, titanic_testdata[,"Survived"])
sum(diag(accuracy))/sum(accuracy)
accuracy


confusionMatrix(data=pred, titanic_testdata$Survived)


#K-fold cross validation built into GLM

ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

titanic_model_traincontrol <- train(Survived ~ Pclass + Sex + Age + SibSp + 
                           Parch + Fare, data=titanic_data_imp, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

pred_1 = predict(titanic_model_traincontrol)
confusionMatrix(data=pred_1, titanic_data_imp$Survived)

accuracy_1 <- table(pred_1, titanic_data_imp[,"Survived"])
sum(diag(accuracy_1))/sum(accuracy_1)
