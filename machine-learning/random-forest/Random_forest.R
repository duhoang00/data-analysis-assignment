#Import library
library(randomForest)
library(datasets)
library(caret)
library(e1071)

#Load Dataset
data<-iris
str(data)
#Convert Species variables to categorical variables
data$Species <- as.factor(data$Species)
table(data$Species)

#Separate Train and Test
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#Implement Random Forest model
rfm <- randomForest(Species~., data=train, proximity=TRUE) 
print(rfm)

#Prediction & Confusion Matrix – train data
p1 <- predict(rfm, train)
confusionMatrix(p1, train$ Species)

#Prediction & Confusion Matrix – test data
p2 <- predict(rfm, test)
confusionMatrix(p2, test$ Species)

#Error rate of Random Forest model - Visulization
plot(rfm)



