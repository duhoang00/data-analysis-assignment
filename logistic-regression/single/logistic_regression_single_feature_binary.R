# Load library
library(tidyverse)
library(modelr)
library(broom)
library('ISLR')
#Load Data set
(mydata <- as_tibble(ISLR::Default))
#Checking for NA values
sum(is.na(mydata))
#Creating the Training and Testing data set
sample <- sample(c(TRUE, FALSE), nrow(mydata), replace = T, prob = c(0.6,0.4))
train <- mydata[sample, ]
test <- mydata[!sample, ]
#Fitting a logistic regression model
logmodel <- glm(default ~ balance, family = "binomial", data = train)
#Plotting a graph: Probability of default Vs Balance
mydata %>%
mutate(prob = ifelse(default == "Yes", 1, 0)) %>%
ggplot(aes(balance, prob)) +
geom_point(alpha = .15) +
geom_smooth(method = "glm", method.args = list(family = "binomial")) +
ggtitle("Logistic regression model fit") +
xlab("Balance") +
ylab("Probability of Default")
#Summary of the Logistic Regression Model
summary(logmodel)
#Fitting a logistic regression model on the testing data
logmodel <- glm(default ~ balance, family = "binomial", data = test)
summary(logmodel)
#Predict
predict(logmodel, data.frame(balance = c(2000)), type = "response")