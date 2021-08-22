#Import library
library(caret)
library(e1071)
library(readr)

#Load Dataset
heart_df  <- read.csv("./machine-learning/support-vector-machine/heart_tidy.csv", sep=',', header = FALSE)
str(heart_df)

#Display the top 6 rows of the data set
head(heart_df)
dim(heart_df)

#Split the data into Train, Test
set.seed(2)
intrain <- createDataPartition(y=heart_df$V14, p=0.7, list=F)
training <- heart_df[intrain,]
testing <- heart_df[-intrain,]

#checking the dimensions of training data frame and testing data
dim(training)
dim(testing)

#Checks for any null values
anyNA(heart_df)

#checking the summary of dataset
summary(heart_df)

#convert the V14 variables to categorical variables
training["V14"] = factor(training[["V14"]])

#control all the computational overheads - PreProcessing
trctrl <- trainControl(method = "repeatedcv", number=10, repeats = 3)
svm_linear <- train(V14~. , data=training, method = "svmLinear", trControl = trctrl, preProcess = c("center", "scale"),tuneLength=10)
svm_linear

#predict classes for test set
test_pred <- predict(svm_linear, newdata= testing)
test_pred 
summary(test_pred)

#Confusion Maxtrix
confusionMatrix(table(test_pred, testing$V14))

#build svmLinear classifier
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(V14 ~., data = training, method = "svmLinear",
trControl=trctrl,
preProcess = c("center", "scale"),
tuneGrid = grid,
tuneLength = 10)
svm_Linear_Grid

#Visualzation
plot(svm_Linear_Grid)

#Predict using model svmLinear for test set
test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
test_pred_grid

#Confusion Matrix
confusionMatrix(table(test_pred_grid, testing$V14))