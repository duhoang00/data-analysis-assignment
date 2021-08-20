#Import library
library(randomForest)
library(datasets)
library(caret)
library(e1071)

#Load Dataset
data<-iris
str(data)
data$Species <- as.factor(data$Species)
table(data$Species)

#Separate Train and Test
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

#Implement Random Forest model
rf <- randomForest(Species~., data=train, proximity=TRUE) 
print(rf)

#Prediction & Confusion Matrix – train data
p1 <- predict(rf, train)
confusionMatrix(p1, train$ Species)

#Prediction & Confusion Matrix – test data
p2 <- predict(rf, test)
confusionMatrix(p2, test$ Species)

#Error rate of Random Forest model
plot(rf)
t <- tuneRF(train[,-5], train[,5], stepFactor = 0.5, plot = TRUE, ntreeTry = 150, trace = TRUE, improve = 0.05)

#No. of nodes for the trees
hist(treesize(rf), main = "No. of Nodes for the Trees", col = "green")
varImpPlot(rf, sort = T, n.var = 10, main = "Top 10 - Variable Importance")
importance(rf)

#Partial Dependence Plot
partialPlot(rf, train, Petal.Width, "setosa")

#Multi-dimensional Scaling Plot of Proximity Matrix
MDSplot(rf, train$Species)
