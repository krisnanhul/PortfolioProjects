## Importing Dataset to detect Credit Card Fraud
### The Dataset used for this project will contain transactions made by credit cards used

library(ranger)
library(caret)
library(data.table)
creditcard_data <- read.csv("C:/Users/Onlinevirus/Desktop/creditcard.csv")

## The Dataset has been imported to R
### I will begin to explore the dataframe within the creditcard_data by using the head() and tail() functions to display the dataset.

dim(creditcard_data)
head(creditcard_data,6)
tail(creditcard_data,6)

### I will use table(), summary(), names(), and var() functions to display generic values of the dataset based on their variables and object 

table(creditcard_data$Class)
summary(creditcard_data$Amount)
names(creditcard_data)
var(creditcard_data$Amount)

### To caluclate the standard deviation of the dataframe, the function sd() is used to measure the average amount of variability in the dataset

sd(creditcard_data$Amount)

## Manipulating the Dataset
### The data used has an extreme value range that may interfere with the functioning of the projected model, the scale() function will be used to
### structure the data to a specified range; thus clustering the data values, ie; data normalization.

head(creditcard_data)
creditcard_data$Amount=scale(creditcard_data$Amount)
NewData=creditcard_data[,-c(1)]
head(NewData)

## Modeling the Dataset
### The dataset has been feature scaled from the previous functions used. The dataset will now be seperated into two attributes by a 80:20 ratio
### and defined as the train_data and the test_data respectively. Both datasets will be used in future analysis to predict fraud detection.
### The dim() function will be used to set the dimensions of the dataset to their respective ratios.

install.packages("caTools")
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)

## Plotting a Logistic Regression Model of the Dataset
### By plotting the Logistic Regression Model, the probability outcome of either two variables are graphed; whether the dataset predicts the variable
### to either be fraud or not fraud

Logistic_Model=glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)

### The Logistic Regression Model has been summarized and now will be plotted through visual graphs.

plot(Logistic_Model)

### To view the performance of the Logistic Regression model, an ROC curve (Receiver operating Characteristics curce) will graph the performance of the
### model at all classification thresholds.

install.packages("pROC")
library(pROC)
lr.predict <- predict(Logistic_Model,test_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")

## Modeling a Decision Tree Algorithm
### Using a Decision Tree to plot the outcomes of the model that can be used to predict the class or variable of the target; whether the target is 
### fraudulent or not. 

install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , creditcard_data, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard_data, type = 'class')
probability <- predict(decisionTree_model, creditcard_data, type = 'prob')
rpart.plot(decisionTree_model)

## Using the ANN models to learn and determine patterns in the dataset
### ANN (Artificial Neural Network) is modeled after the human nervous system. Allows the algorithm to take historical datasets learned and apply
### the patterns to newly input data. 

install.packages("neuralnet")
library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

## Using Gradient Boosting to form a strong algorithm model
### Gradient Boosting (GBM) will be used to perform classification and regression tasks. This method will take the disision trees and combine them to
### form a stronger model of gradient boosting.

install.packages("gbm")
library(gbm, quietly=TRUE)

### The following function will train the GBM model to learn the classification and regression tasks

system.time(model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))))
gbm.iter = gbm.perf(model_gbm, method = "test")

### The GBM has completed the learning process within a certain time frame.
### The GBM model will now be plotted

model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
plot(model_gbm)

### Bernoulli's deviance is graphed to show the "goodness-of-fit" for the statistical model compared to the amount of iterations the test data is 
### processed.

## AUC is plotted to calcualte the test data
### AUC (Area Under the Receiver Operating Characteristics) represents the degree or measure of separability. This function will tell how much the 
### model is better at distinguishing between fraud or no fraud. The bigger the lower area, the better the model can predict fraud. 

gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")

print(gbm_auc)

### Area under the curve: 0.9555

## SUMMARY
### I've learned how to develop a credit card fraud detection model using different machine learning options to tackle and predict credit card fraud.
### Different algorithms were used to plot models and curves of the dataset provided. I also analyzed and visualized the dataset to determine fraudulent
### credit card transactions. 