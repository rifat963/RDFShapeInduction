# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)


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
  control <- trainControl(method="cv", number=10)
  
  
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
  # AdaBoost.M1
  
  # set.seed(12)
  # fit.AdaBoost <- train(Class~., data=training_data, method="AdaBoost.M1", trControl=control)
  
  # Quadratic
  # set.seed(12)
  # fit.qda <- train(Class~., data=training_data, method="qda", trControl=control)
  
  
  # collect resamples
  results <- resamples(list(CART=fit.cart,bayes=fit.bayes , SVM=fit.svm, KNN=fit.knn,RF=fit.rf, NNET=fit.nnet,GBM=fit.gbm,C45=fit.c45, Logistic_R=fit.logic))
  
  return(results)
}

# Various ML algorithms

control <- trainControl(method="cv", number=10, repeats=3)

# Quadratic
cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                       classProbs = TRUE, 
                       summaryFunction = twoClassSummary)

set.seed(849)
test_class_cv_form <- train(Class ~ ., data = ml_data_smote, 
                            method = "qda", 
                            trControl = cctrl1,
                            metric = "ROC", 
                            preProc = c("center", "scale"))


# Test Data

filename="3cixty-min-card.csv"
filename="3cixty-max-card.csv"

filename="dbp-min-card.csv"
filename="dbp-max-card.csv"

ml_data<-load_dataSet(filename)
head(ml_data)
ml_data$Class=factor(ml_data$Class)

ml_data_smote<-smote_data(ml_data,100,200)

ml_data_smote$Class=factor(ml_data_smote$Class)

# write.csv(ml_data_smote,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-max-card-smote.csv",row.names =   FALSE)

prop.table(table(ml_data$Class))

prop.table(table(ml_data_smote$Class))

# Pre process the level for class
results<-ML_algorithms(ml_data)

original_data<- summary(results)

# with smote data
results<-ML_algorithms(ml_data_smote)

# summarize differences between modes
with_smote_data<- summary(results)


# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#--------------------------------------------------

training  <- load_dataSet("dbp-card-training.csv")
attach(training) #attaching data frame to reduce the length of the variable names associated to it.

head(training)

training<-training[,c(1,2,3,4,5,6)]

testing   <-  load_dataSet("dbo_person_test.csv")

head(testing)

# exploring the data

tail(names(training),24)
summary(training$AntMin)

is.factor(OntoClass)
is.factor(AntMax)
str(OntoProperty)

library(ggplot2)
library(caret)
## Loading required package: lattice
#selecting a few of the more promising predictors to be plotted
colSelection<- c("DataMin","AntMin", "AntMax", "DataMax")

#creating a feature plot 
featurePlot(x=training[,colSelection],y = training$AntMin, plot="pairs")

# Min cardinality
qplot(DataMin, DataMax, colour=AntMin, data=training)

training$Class=factor(training$AntMin)

head(training)

training <-training[,colSums(is.na(training))==0]

col_names <- c()
n <- ncol(training)-1
for (i in 1:n) {
  if (is.factor(training[,i])){
    col_names <- c(col_names,i)
  }
}

training <- training[,-col_names]

library(randomForest)
first_seed <- 123355
accuracies <-c()
for (i in 1:3){
  set.seed(first_seed)
  first_seed <- first_seed+1
  trainIndex <- createDataPartition(y=training$Class, p=0.75, list=FALSE)
  trainingSet<- training[trainIndex,]
  testingSet<- training[-trainIndex,]
  modelFit <- randomForest(Class ~., data = trainingSet)
  prediction <- predict(modelFit, testingSet)
  testingSet$rightPred <- prediction == testingSet$Class
  t<-table(prediction, testingSet$Class)
  print(t)
  accuracy <- sum(testingSet$rightPred)/nrow(testingSet)
  accuracies <- c(accuracies,accuracy)
  print(accuracy)
}

mean(accuracies)

modelFit <- randomForest(Class ~., data = training)

# nrow(testing)
# 
# testing<-testing[,c(2,3)]

prediction <- predict(modelFit, testing)


testing$prediction<-prediction

# -------------------------------------------------
# Max cardinality
# -------------------------------------------------

training  <- load_dataSet("dbp-card-training.csv")
attach(training) #attaching data frame to reduce the length of the variable names associated to it.

head(training)

training<-training[,c(1,2,3,4,5,6)]

testing   <-  load_dataSet("dbo_person_test.csv")

head(testing)

# exploring the data

tail(names(training),24)
summary(training$AntMax)

is.factor(OntoClass)
is.factor(AntMax)
str(OntoProperty)

library(ggplot2)
library(caret)
## Loading required package: lattice
#selecting a few of the more promising predictors to be plotted
colSelection<- c("DataMin","AntMin", "AntMax", "DataMax")

#creating a feature plot 
featurePlot(x=training[,colSelection],y = training$AntMin, plot="pairs")

# Min cardinality
qplot(DataMin, DataMax, colour=AntMin, data=training)

training$Class=factor(training$AntMax)

head(training)

training <-training[,colSums(is.na(training))==0]

col_names <- c()
n <- ncol(training)-1
for (i in 1:n) {
  if (is.factor(training[,i])){
    col_names <- c(col_names,i)
  }
}

training <- training[,-col_names]

library(randomForest)
first_seed <- 123355
accuracies <-c()
for (i in 1:3){
  set.seed(first_seed)
  first_seed <- first_seed+1
  trainIndex <- createDataPartition(y=training$Class, p=0.75, list=FALSE)
  trainingSet<- training[trainIndex,]
  testingSet<- training[-trainIndex,]
  modelFit <- randomForest(Class ~., data = trainingSet)
  prediction <- predict(modelFit, testingSet)
  testingSet$rightPred <- prediction == testingSet$Class
  t<-table(prediction, testingSet$Class)
  print(t)
  accuracy <- sum(testingSet$rightPred)/nrow(testingSet)
  accuracies <- c(accuracies,accuracy)
  print(accuracy)
}

mean(accuracies)

modelFit <- randomForest(Class ~., data = training)

# nrow(testing)
# 
# testing<-testing[,c(2,3)]

prediction <- predict(modelFit, testing)


testing$prediction<-prediction


