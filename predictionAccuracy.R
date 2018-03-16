# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)
library(ggplot2)
library(OneR)

#Zero Rule
ZeroR <- function(X, targetId) {
  # ZeroR Algorithm: Finds the most commonly occuring class
  # 
  # Args:
  # X: data frame or Matrix
  # targetId: response/outcome/target/class feature column number
  
  # Returns:
  # A vector containing the commonly occuring class value and its count 
  if ( is.character(X[, targetId]) | is.factor(X[, targetId]) ) {
    u.x <- unique(X[, targetId])
    u.x.temp <- c()
    for (i in u.x) {
      u.x.temp <- c(u.x.temp, sum(X[, targetId] == i))
    }
    print(u.x.temp)
    accuracy=max(u.x.temp)/length(u.x)
    print(accuracy)
    names(u.x.temp) <- u.x
    return( c(max(u.x.temp), names(u.x.temp)[which.max(u.x.temp)]) ) 
  }
  return(NULL)
}

# Function For loading training dataSets

load_dataSet<-function(name){
  
  data=read.csv(paste("C:/Users/rifat/Desktop/R_milan/githubRepo/RDFShapeInduction/dataset/",name,sep = ""),header = TRUE)
  
  return(data)  
}

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
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  
  
  # Quadratic
  # set.seed(12)
  # fit.qda <- train(Class~., data=training_data, method="qda", trControl=control)
  
  
  # # Logistic regression
  # set.seed(12)
  # fit.lr <- train(Class~., data=training_data, method="multinom", trControl=control)
  
  #Bayesian Generalized Linear Model
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
  #Neural Network
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

# Dataset
filename="3cixty-min-card.csv"
filename="3cixty-max-card.csv"

filename="dbp-min-card.csv"
filename="dbp-max-card.csv"

ml_data<-load_dataSet(filename)
nrow(ml_data)
ml_data$Class=factor(ml_data$Class)

set.seed(3033)

# intrain <- sample(1:nrow(ml_data),size = 0.7*nrow(ml_data)) 
intrain <- createDataPartition(y = ml_data$Class, p= 0.7, list = FALSE)

training <- ml_data[intrain,]

nrow(training)

testing <- ml_data[-intrain,]

nrow(testing)

ml_data<-training

head(ml_data)
ml_data_smote<-smote_data(ml_data,100,200)
ml_data_smote$Class=factor(ml_data_smote$Class)

trainSmote<-ml_data_smote

# write.csv(ml_data_smote,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-max-card-smote.csv",row.names =   FALSE)

prop.table(table(ml_data$Class))
prop.table(table(ml_data_smote$Class))

# Test the original training datasets
# results<-ML_algorithms(ml_data)
# original_data<- summary(results)

# Overall resampling the machine learing algorithms

# with smote data
results<-ML_algorithms(ml_data_smote)
# summarize differences between modes
with_smote_data<- summary(results)
# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))

bwplot(results, scales=scales)
# dot plots of results
dotplot(results)

#------------------
# training$Class=factor(training$Class)
# ml_data_smote<-smote_data(training,100,200)
nrow(testCard)
# Testing for zeroR rule
ZeroR(training,1)
ZeroR(testCard,1)

set.seed(3033)
Train_options<-trainControl(method = "repeatedcv", number = 10, repeats = 3)


# # Logistic Regression
# set.seed(3033)
# fit.svm <- train(Class~., data=trainSmote, method="svmRadial", trControl=Train_options)
# test_pred <- predict( fit.svm  , testCard)
# confusionMatrix(testCard$Class,test_pred)
# set.seed(3033)
# fit.rf <- train(Class~., data=trainSmote, method="rf", trControl=control)
# set.seed(3033)
# # testMinCard$Class=factor(testMinCard$Class)
# 
# test_pred <- predict( fit.svm  , testCard)
# 
# test_pred <- predict( fit.rf   , testCard)
# 
# confusionMatrix(testCard$Class,test_pred)


testCard=testing

seed=3033
training_data=trainSmote

#Bayesian Generalized Linear Model
set.seed(seed)
fit.bayes <- train(Class~., data=training_data, method="nb", trControl=Train_options)
test_pred <- predict( fit.bayes  , testCard)
confusionMatrix(testCard$Class,test_pred)

# SVM
set.seed(seed)
fit.svm <- train(Class~., data=training_data, method="svmRadial", trControl=Train_options)
test_pred <- predict( fit.svm  , testCard)
confusionMatrix(testCard$Class,test_pred)

# kNN
set.seed(seed)
fit.knn <- train(Class~., data=training_data, method="knn", trControl=Train_options)
test_pred <- predict( fit.knn  , testCard)
confusionMatrix(testCard$Class,test_pred)

# CART
set.seed(seed)
fit.cart <- train(Class~., data=training_data, method="rpart", trControl=Train_options)
test_pred <- predict( fit.cart  , testCard)
confusionMatrix(testCard$Class,test_pred)

# Gradiant Boosting
set.seed(seed)
fit.gbm <- train(Class~., data=training_data, method="gbm", trControl=Train_options)
test_pred <- predict( fit.gbm  , testCard)
confusionMatrix(testCard$Class,test_pred)


# Random Forest
set.seed(seed)
fit.rf <- train(Class~., data=training_data, method="rf", trControl=Train_options)
test_pred <- predict( fit.rf  , testCard)
confusionMatrix(testCard$Class,test_pred)

#Neural Network
set.seed(seed)
fit.nnet <- train(Class~., data=training_data, method="nnet", trControl=Train_options)
test_pred <- predict( fit.nnet  , testCard)
confusionMatrix(testCard$Class,test_pred)

# C4.5
set.seed(seed)
fit.c45 <- train(Class~., data=training_data, method="J48", trControl=Train_options)
test_pred <- predict( fit.c45  , testCard)
confusionMatrix(testCard$Class,test_pred)

# Logistic Regression
set.seed(seed)
fit.logic <- train(Class~., data=training_data, method="LogitBoost", trControl=Train_options)
test_pred <- predict( fit.logic  , testCard)
confusionMatrix(testCard$Class,test_pred)




