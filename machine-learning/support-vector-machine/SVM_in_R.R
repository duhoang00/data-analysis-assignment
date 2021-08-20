#Import library
library(caTools)
library(e1071)
library(readr)

#Load Dataset
social <- read_csv("./machine-learning/support-vector-machine/social.csv")
social = social[3:5]
social$Purchased = factor(social$Purchased, levels = c(0, 1))

#Splitting the dataset
set.seed(123)
split = sample.split(social$Purchased, SplitRatio = 0.75)
  
training_set = subset(social, split == TRUE)
test_set = subset(social, split == FALSE)
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

#Fitting SVM to the training set
classifier = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'linear')

#Summary
summary(classifier)

#Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])
cm = table(test_set[, 3], y_pred)

#Visualizing the Training set results
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
  
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
  
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine'))
  
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#Visualizing the Test set results
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
  
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
  
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
  
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine'))
  
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
