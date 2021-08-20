#Import Library
library(DAAG)
library(party)
library(rpart)
library(rpart.plot)
library(mlbench)
library(caret)
library(pROC)
library(tree)

#Load Data form R in-built data set (Email Spam Detection)
str(spam7)
mydata <- spam7

#Separate Train and Test
set.seed(1234)
ind <- sample(2, nrow(mydata), replace = T, prob = c(0.5, 0.5))
train <- mydata[ind == 1,]
test <- mydata[ind == 2,]

#Tree Classification
tree <- rpart(yesno ~., data = train)
rpart.plot(tree)
printcp(tree)
rpart(formula = yesno ~ ., data = train)
tree <- rpart(yesno ~., data = train,cp=0.07444)

#Confusion Matrix
p <- predict(tree, train, type = 'class')
confusionMatrix(p, train$yesno, positive='y')
