# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)
library(ggplot2)
library(OneR)
library("RSQLite")
library(sqldf)
library("data.table")
library("readr")
library(stringr)
library(plyr)
library(reshape2)
library(httr)
library(RCurl)
library(curl)
library(magrittr)

# Function for smote 

smote_data<-function(data,over.val,under.val){
  
  re_somte_data <- SMOTE(Class ~ ., data, perc.over=over.val, perc.under=under.val)
  
  return(re_somte_data)  
}

# Function for ML algorithms
# Testing ML algorithms using the optimized parameters 
# DataSet divided into Test & Training Set.

ML_algorithms<-function(training_data){
  
  # prepare training scheme
  # training_data= max_card_smote
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)

  # Bayesian Generalized Linear Model
  set.seed(12)
  fit.bayes <- train(Class~., data=training_data, method="nb", trControl=control)
  
  # SVM
  set.seed(12)
  fit.svm <- train(Class~., data=training_data, method="svmRadial", trControl=control)
  # kNN
  set.seed(12)
  fit.knn <- train(Class~., data=training_data, method="knn", trControl=control)
  
  # CART
  set.seed(12)
  fit.cart <- train(Class~., data=training_data, method="rpart", trControl=control)
  
  # Gradiant Boosting
  set.seed(12)
  fit.gbm <- train(Class~., data=training_data, method="gbm", trControl=control)

  # Random Forest
  set.seed(12)
  fit.rf <- train(Class~., data=training_data, method="rf", trControl=control)
  
  # Neural Network
  set.seed(12)
  fit.nnet <- train(Class~., data=training_data, method="nnet", trControl=control)
  
  # C4.5
  set.seed(12)
  fit.c45 <- train(Class~., data=training_data, method="J48", trControl=control)
  
  # Logistic Regression
  set.seed(12)
  fit.logic <- train(Class~., data=training_data, method="LogitBoost", trControl=control)

  # collect resamples
  results <- resamples(list(DecisionTree.CART=fit.cart,Bayesian=fit.bayes , SupportVectorMachine.SVM=fit.svm, KNN=fit.knn,RandomForest.RF=fit.rf,NeuralNetwork.NNET=fit.nnet,GradiantBoost.GBM=fit.gbm,DecisionTree.C45=fit.c45, LogisticRegression=fit.logic))
  
  return(results)
}




Get_results <- function(...){
  
  Args <- list(...)
  Model_names <- as.list(sapply(substitute({...})[-1], deparse))
  
  message("Model names:")
  print(Model_names)
  
  # Function for getting max sensitivity
  Max_sens <- function(df, colname = "results"){
    df <- df[[colname]]
    new_df <- df[which.max(df$Sens), ]
    x <- sapply(new_df, is.numeric)
    new_df[, x] <- round(new_df[, x], 2)
    new_df
  }
  
  # Find max Sens for each model
  message("Max sensitivity from model printout:")
  Max_sens_out <- lapply(Args, Max_sens)
  names(Max_sens_out) <- Model_names
  print(Max_sens_out)
  
  # Find predict() result for each model
  message("Results using predict():")
  set.seed(Seed)
  Predict_out <- lapply(Args, function(x) predict(x, trainMinCardSmote))
  Predict_results <- lapply(Predict_out, function(x) confusionMatrix(x, trainMinCardSmote$Class))
  names(Predict_results) <- Model_names
  print(Predict_results)
  
  # Find resamples() results for each model
  
  message("Results using resamples():")
  set.seed(Seed)
  results <- resamples(list(...),modelNames = Model_names)
  # names(results) <- Model_names
  summary(results)
  
}

