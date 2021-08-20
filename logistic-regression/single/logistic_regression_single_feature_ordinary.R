# - Import Data
library(dplyr)
library(scatterplot)
library(caTools)
library(ROCR)

# - Load Data
ads <- read.csv(file.choose())
head(ads)
# -Replace NA values in dataset
ads[is.na(ads)]=0
head(ads)

colnames(ads)[:-1] <- "Clicked.on.Ad"
head(ads)
# Split train & test
set.seed(100)
split = sample.split(ads$Clicked.on.Ad, SplitRatio = 0.65)
train = subset(ads,split==TRUE)
test = subset(ads,split==FALSE)
head(train)
# - CreateModel
model <- glm(Clicked.on.Ad ~ Age, data = train,  family = binomial)
summary(model)

# - GetResult
coef(model) %>% 
  exp() %>% 
  round(digits = 6)

# - Create Predict base on train data
y_pred <-predict(model,train,type="response")
summary(y_pred)

# - Create Predict base on test data
y_test <-predict(model,test,type="response")
summary(y_test)

# - Confusion Matrxi
table(test$Clicked.on.Ad,y_test > 0.5)

# - Visualize
ROCRpred = prediction(y_pred,train$Clicked.on.Ad)
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,colorize=TRUE)

# - Caculate Accuracy Score
as.numeric(performance(ROCRpred,"auc")@y.values)