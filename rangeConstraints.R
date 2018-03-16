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
    accuracy=1-(max(u.x.temp)/length(u.x))
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
  training_data=ml_data_smote
  
 
  
  
  set.seed(12)
  # training_data=ml_data_smote
  fit.nnet <- train(Class~., data=training_data, method="nnet", trControl=control)
  
  pred <- predict(fit.nnet, training_data) 
  
  # cf <- confusionMatrix(pred, fit.nnet$trainingData$.outcome, mode = "everything")
  
  # print(cf)
  
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

# Various ML algorithms

# control <- trainControl(method="cv", number=10, repeats=3)
# 
# # Quadratic
# cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
#                        classProbs = TRUE, 
#                        summaryFunction = twoClassSummary)
# 
# set.seed(849)
# test_class_cv_form <- train(Class ~ ., data = ml_data_smote, 
#                             method = "qda", 
#                             trControl = cctrl1,
#                             metric = "ROC", 
#                             preProc = c("center", "scale"))

# Dataset
# filename="3cixty-min-card.csv"
# filename="3cixty-max-card.csv"
# 
# filename="dbp-min-card.csv"
# filename="dbp-max-card.csv"

#Read Range constraints data

filename="dbo-range-SportsTeam.csv"

filename="3cixty-nice-place-range.csv"

ml_data=read.csv("C:/Users/rifat/Desktop/R_milan/githubRepo/FeatureEngineering/dataset/constraints/3cixty-nice-place-range.csv",header = T)

ml_data<-load_dataSet(filename)
# Pre process

head(ml_data)
tail(ml_data)

nrow(ml_data)

length(unique(ml_data$prop))

# make the label class as the feature vector
unique(ml_data$Label)

# Annotate each property with a no.
propList=data.frame(prop=ml_data$prop)
propList$id=1:nrow(propList)
head(propList)

# Make the dataset with prop and label
head(rangeData)
# Factorize the prop list with the id
rangeData$prop <- propList$id[match(rangeData$prop, propList$prop)]

# check is there any NA in the dataset
rangeData[is.na(propList$prop),]



rangeData=data.frame(Class=ml_data$Label,prop=ml_data$prop)
rangeData$Class <- as.character(rangeData$Class)
rangeData$Class[ml_data$Class=="IRI"] <- "1"
rangeData$Class[ml_data$Class=="LIT"] <- "0"
ml_data=rangeData
# junk$nm[junk$nm == "B"] <- "b"


# make the class factor
ml_data$Class=factor(ml_data$Class)

prop.table(table(ml_data$Class))


nrow(ml_data)

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

# write.csv(ml_data_smote,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-max-card-smote.csv",row.names =   FALSE)

prop.table(table(ml_data$Class))

prop.table(table(ml_data_smote$Class))

# Pre process the level for class
# results<-ML_algorithms(ml_data)

# original_data<- summary(results)

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
# 
# training$Class=factor(training$Class)
# 
# ml_data_smote<-smote_data(training,100,200)

testCard=testing


nrow(testCard)

nrow(testCard[testCard$Class=="IRI",])

nrow(training[training$Class=="LIT",])



ZeroR(training,1)

ZeroR(testCard,1)


set.seed(3033)
Train_options<-trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Logistic Regression
set.seed(3033)
fit.svm <- train(Class~., data=training, method="svmRadial", trControl=Train_options)
set.seed(3033)
fit.rf <- train(Class~., data=training, method="rf", trControl=Train_options)
# testMinCard$Class=factor(testMinCard$Class)
set.seed(3033)
fit.c45 <- train(Class~., data=training, method="J48", trControl=Train_options)

test_pred <- predict( fit.svm  , testCard)

test_pred <- predict( fit.rf   , testCard)

3 <- predict( fit.c45   , testCard)

confusionMatrix(testCard$Class,test_pred)




