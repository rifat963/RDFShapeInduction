# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)
library(ggplot2)
library(OneR)
library(ROCR)
## Load libraries
library(grid)

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
    accuracy=1-max(u.x.temp)/length(u.x)
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
  # set.seed()
  # fit.qda <- train(Class~., data=training_data, method="qda", trControl=control)
  
  
  # # Logistic regression
  # set.seed(12)
  # fit.lr <- train(Class~., data=training_data, method="multinom", trControl=control)
  
  #Bayesian Generalized Linear Model
  training_data= ml_data_smote
  
  set.seed(3033)
  fit.bayes <- train(Class~., data=training_data, method="nb", trControl=control)
  
  pred <- predict(fit.bayes, training_data) 
  
  cf <- confusionMatrix(pred, fit.bayes$trainingData$.outcome, mode = "everything")
  
  print(cf)
  # svmRadial
  # SVM
  set.seed(3033)
  fit.svm <- train(Class~., data=training_data, method="svmRadial", trControl=control)
  pred <- predict(fit.svm, training_data) 
  
  cf <- confusionMatrix(pred, fit.svm$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  # kNN
  set.seed(3033)
  fit.knn <- train(Class~., data=training_data, method="knn", trControl=control)
  
  pred <- predict(fit.knn, training_data) 
  
  cf <- confusionMatrix(pred, fit.knn$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  # CART
  set.seed(12)
  fit.cart <- train(Class~., data=training_data, method="rpart", trControl=control)
  
  pred <- predict(fit.cart, training_data) 
  
  cf <- confusionMatrix(pred, fit.cart$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  # Gradiant Boosting
  set.seed(3033)
  fit.gbm <- train(Class~., data=training_data, method="gbm", trControl=control)
  pred <- predict(fit.gbm, training_data) 
  
  df=training_data
  probs=pred
  
  cf <- confusionMatrix(pred, fit.gbm$trainingData$.outcome, mode = "everything")
  print(cf)
  
  require(precrec)
  # evalss <- evalmod(scores = pred$, labels = pred$Class)
  library(plotROC)
  
  require(PRROC)
  fg <- probs[df$Class == "0"]
  bg <- probs[df$Class == "1"]
  
  # ROC Curve    
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(roc)
  
  # PR Curve
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(pr)
  
  basicplot <- ggplot(probs, aes(d = fg, m = bg)) + geom_roc()
  basicplot
  
  training_data= ml_data_smote
  # Random Forest
  set.seed(3033)
  fit.rf <- train(Class~., data=training_data, method="rf", trControl=control)
  
  pred <- predict(fit.rf, training_data) 
  
  cf <- confusionMatrix(pred, fit.rf$trainingData$.outcome, mode = "everything")
  
  print(cf)


  #Neural Network
  training_data=ml_data_smote
  
  set.seed(3033)
  fit.nnet <- train(Class~., data=training_data, method="nnet", trControl=control)
  
  pred <- predict(fit.nnet, training_data) 
  
  cf <- confusionMatrix(pred, fit.nnet$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  # C4.5
  set.seed(12)
  fit.c45 <- train(Class~., data=training_data, method="J48", trControl=control)
  pred <- predict(fit.c45, testing) 
  
  cf <- confusionMatrix(pred, fit.c45$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  
  # Logistic Regression
  set.seed(3033)
  fit.logic <- train(Class~., data=training_data, method="LogitBoost", trControl=control)
  pred <- predict(fit.logic, training_data) 
  cf <- confusionMatrix(pred, fit.logic$trainingData$.outcome, mode = "everything")
  print(cf)
  
  # collect resamples
  results <- resamples(list(DecisionTree.CART=fit.cart,Bayesian=fit.bayes , SupportVectorMachine=fit.svm, kNN=fit.knn, RandomForest=fit.rf,NeuralNetwork=fit.nnet,GradientBoostingMachines=fit.gbm, DecisionTree.C45=fit.c45, LogisticRegression=fit.logic))
  return(results)
}


# Dataset
# 3cixty
filename="3cixty-min-card.csv"
filename="3cixty-max-card.csv"
filename="lode-Event-property-range.csv"


# English DBpedia
filename="dbp-min-card.csv"
filename="dbp-max-card.csv"
filename="dbo-Place-property-range.csv"

# Spanish DBpedia
filename="dbp-min-cardS.csv"
filename="dbp-max-cardS.csv"
filename="dbo-range-S.csv"

ml_data<-load_dataSet(filename)
nrow(ml_data)
head(ml_data)

#for range
ml_data=data.frame(Class=ml_data$Label,prop=ml_data$prop)
# for range 3city
df2 <- ml_data[sample(nrow(ml_data)),]
ml_data=rbind(ml_data,df2)

# Spnaish DBpedia
# min card
minCard=data.frame(Class=ml_data$Class, prop=ml_data$Class)
# max card
maxCard=data.frame(Class=ml_data$Class.1, label=trimws(ml_data$AntMax))
minCard$Class=ifelse(midCard$label=="M0",0,1)
# max Card
maxCard=data.frame(p=ml_data$OntoProperty,Class=ml_data$AntMax)

ml_data=minCard
ml_data=maxCard

nrow(ml_data)

ml_data$Class=factor(ml_data$Class)

set.seed(3033)

ml_data_smote<-smote_data(ml_data,100,200)

ml_data_smote$Class=factor(ml_data_smote$Class)

ml_data=ml_data_smote

training_data= ml_data

training_data= ml_data_smote

#-------------------------    --------------------------#

# intrain <- sample(1:nrow(ml_data),size = 0.7*nrow(ml_data)) 
intrain <- createDataPartition(y = ml_data$Class, p= 0.7, list = FALSE)

training <- ml_data[intrain,]

nrow(training)

testing <- ml_data[-intrain,]

nrow(testing)

ml_data<-training

head(ml_data)
training_data= ml_data

# write.csv(ml_data_smote,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-max-card-smote.csv",row.names =   FALSE)

prop.table(table(ml_data$Class))

prop.table(table(ml_data_smote$Class))
# Pre process the level for class
# with smote data
results<-ML_algorithms(ml_data_smote)
# summarize differences between modes
with_smote_data<- summary(results)
with_smote_data$models
# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)
# dot plots of results
dotplot(results)


#---------------------------------------------#
