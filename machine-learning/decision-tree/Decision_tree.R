#Import library
library(party)
library(tree)

#Load Dataset
data(iris)
str(iris)
head(iris)

#Split Train, Test
set.seed(1234) #To get reproducible result
ind <- sample(2,nrow(iris), replace=TRUE, prob=c(0.7,0.3))
TrainData <- iris[ind==1,]
TestData <- iris[ind==2,]

#Create model
DecisionTreeModel <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(DecisionTreeModel, data=TrainData)

#Predict Train Data
train_predict <- predict(iris_ctree,TrainData,type="response")

#Confusion matrix and misclassification errors - Train Data
table(train_predict,TrainData$Species)
mean(train_predict != TrainData$Species) * 100

#Predict Test Data
test_predict <- predict(iris_ctree, newdata= TestData,type="response")

#Confusion matrix and misclassification errors - Test Data
table(test_predict, TestData$Species)
mean(test_predict != TestData$Species) * 100

#Print and plot model
print(iris_ctree)
plot(iris_ctree)
plot(iris_ctree, type="simple")