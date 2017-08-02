library(DMwR)
## ML algorithms
# load libraries
library(mlbench)
library(caret)
# load the libraries
library(caret)
library(klaR)
library(randomForest)

# load the dataset
# Original DataSet
min_card<-read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/3cixty-min-card.csv",header = TRUE)
max_card<-read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/3cixty-max-card.csv",header = TRUE)

dbp_card<-read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-card-training.csv",header = TRUE)

head(min_card)
head(max_card)
head(dbp_card)

head(dbp_card)
dbp_card<-dbp_card[,3:ncol(dbp_card)]

dbp_min_card<-dbp_card[,-c(2)]
dbp_max_card<-dbp_card[,-c(1)]

dbp_min_card=plyr::rename(dbp_min_card, c("AntMin"="Class"))
dbp_max_card=plyr::rename(dbp_max_card, c("AntMax"="Class"))
head(dbp_min_card)


write.csv(dbp_min_card,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-min-card.csv",row.names =   FALSE)
write.csv(dbp_max_card,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-max-card.csv",row.names =   FALSE)


prop.table(table(dbp_card$AntMin))

# dbp_card$AntMin[dbp_card$AntMin=="M1 ",]



prop.table(table(min_card$Class))
prop.table(table(max_card$Class))


# Pre process the level for class
min_card$Class=factor(min_card$Class)
max_card$Class=factor(max_card$Class)

dbp_card$AntMin=factor(dbp_card$AntMin)

dbp_card_smote <- SMOTE(AntMin ~ ., dbp_card, perc.over = 100, perc.under=200)

prop.table(table(dbp_card_smote$AntMin))

write.csv(dbp_card_smote,"C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/dbp-card-smote-training.csv",row.names =   FALSE)


#perc.over-- A number that drives the decision of how many extra cases from the minority 
# class are generated (known as over-sampling).
#perc.under--A number that drives the decision of how many extra cases from the majority 
# classes are selected for each case generated from the minority class (known as under-sampling)

min_card_smote <- SMOTE(Class ~ ., min_card, perc.over = 600, perc.under=100)

prop.table(table(min_card_smote$Class))

max_card_smote <- SMOTE(Class ~ ., max_card, perc.over = 600, perc.under=100)

prop.table(table(max_card_smote$Class))


# Testing ML algorithms using the optimized parameters 
# DataSet divided into Test & Training Set.

ML_algorithms<-function(training_data){
  
  # prepare training scheme
  # training_data= max_card_smote
  control <- trainControl(method="repeatedcv", number=10, repeats=3)
  # CART
  set.seed(7)
  fit.cart <- train(Class~., data=training_data, method="rpart", trControl=control)
  # fit.cart$metric
  # LDA
  #set.seed(7)
  #fit.lda <- train(Class~., data=test_data, method="lda", trControl=control)
  
  
  #Bayesian Generalized Linear Model
  set.seed(7)
  fit.bayes <- train(Class~., data=training_data, method="nb", trControl=control)
  
  # SVM
  set.seed(7)
  fit.svm <- train(Class~., data=training_data, method="svmRadial", trControl=control)
  # kNN
  set.seed(7)
  fit.knn <- train(Class~., data=training_data, method="knn", trControl=control)
  # Random Forest
  set.seed(7)
  fit.rf <- train(Class~., data=training_data, method="rf", trControl=control)
  #Neural Network
  set.seed(7)
  fit.nnet <- train(Class~., data=training_data, method="nnet", trControl=control)
  
  # load the package
  # library(nnet)
  # # fit model
  # set.seed(7)
  # 
  # data_sub=min_card_smote[,1:6]
  # fit.nnet <- nnet(Class~., data=data_sub, size=4, decay=0.0001, maxit=500)
  # # summarize the fit
  # summary(fit.nnet)
  # # make predictions
  # 
  #  predictions <- predict(fit.nnet, data, type="class")
  # # # summarize accuracy
  # table(predictions, min_card_smote$Class)
  
  # collect resamples
  results <- resamples(list(CART=fit.cart,bayes=fit.bayes , SVM=fit.svm, KNN=fit.knn,RF=fit.rf, NNET=fit.nnet))
  
  return(results)
}


results<-ML_algorithms(dbp_card)
# summarize differences between modes
summary(results)

# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

################## Test ML algorithms #################


# load the iris dataset
data("iris")
# define an 80%/20% train/test split of the dataset
split=0.80
trainIndex <- createDataPartition(dbp_card$AntMin, p=split, list=FALSE)
data_train <- dbp_card[ trainIndex,]
data_test <- dbp_card[-trainIndex,]
# train a naive bayes model
model <- NaiveBayes(AntMin~., data=data_train)
# make predictions
x_test <- data_test[,2:ncol(dbp_card)]
y_test <- data_test[,1]
predictions <- predict(model, x_test)
# summarize results
confusionMatrix(predictions$class,y_test)

################## Test ML algorithms #################
#Strategy
#When use any sampling technique ( specifically synthetic) you divide your 
#data first and then apply synthetic sampling on the training data only. 
#After you train you use the testing set ( which contains only original samples) to evaluate. 
#The risk if you use your strategy is to have the original sample in training ( testing) and 
#the synthetic sample ( that was created based on this original sample) in the testing ( training) set.

# Split data: training - 80% and test 20%
# split=0.80
# trainIndex <- createDataPartition(min_card$Class, p=split, list=FALSE)
# data_train <- max_card[ trainIndex,]
# data_test <- max_card[-trainIndex,]
# 
# data_train_smote <- SMOTE(Class ~ ., data_train, perc.over = 600, perc.under=100)

library(caret)
max_card<-max_card[,1:5]
set.seed(1234)
splitIndex <- createDataPartition(max_card$Class, p = .50,
                                  list = FALSE)
trainSplit <- max_card[ splitIndex,]
testSplit <- max_card[-splitIndex,]

prop.table(table(trainSplit$Class))

prop.table(table(testSplit$Class))

ctrl <- trainControl(method="cv", number=10, repeats=3)

tbmodel <- train(Class ~ ., data = trainSplit, method = "rpart",
                 trControl = ctrl)

predictors <-names(testSplit)[names(testSplit) != 'Class']

summary(tbmodel$results)

pred <- predict(tbmodel$finalModel, testSplit[,predictors])



as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}

pred=as.data.frame(pred)

as.numeric.factor(pred)
summary(pred)

library(pROC)
auc <- roc(testSplit$Class, pred)
print(auc)

plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

# Apply smote

library(DMwR)
trainSplit$Class <- as.factor(trainSplit$Class)
trainSplit <- SMOTE(Class ~ ., trainSplit, perc.over = 100, perc.under=200)
trainSplit$Class <- as.numeric(trainSplit$Class)

prop.table(table(trainSplit$Class))

# After smote

trainSplit$Class=factor(trainSplit$Class)

tbmodel <- train(Class ~ ., data = trainSplit, method = "rpart",
                 trControl = ctrl)

summary(tbmodel$results)
predictors <- names(testSplit)[names(testSplit) != 'Class']

pred <- predict(tbmodel$finalModel, testSplit[,predictors])
summary(pred)

confusionMatrix (testSplit$Class, pred)

# confusionMatrix(pred,testSplit$Class)

# help("confusionMatrix")

auc <- roc(testSplit$Class, as.numeric.factor(pred))

plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)
print(auc)

######## Test Naive Bayes ########

split=0.50
data_test_max_card<- max_card[,1:5]
trainIndex <- createDataPartition(data_test_max_card$Class, p=split, list=FALSE)
data_train <- data_test_max_card[ trainIndex,]
data_test <- data_test_max_card[-trainIndex,]

prop.table(table(data_train$Class))
prop.table(table(data_test$Class))

# train a naive bayes model
model <- NaiveBayes(Class~., data=data_train)

class(data_test[,4])

# make predictions
x_test <- data_test[,2:5]
y_test <- data_test[,1]
predictions <- predict(model, x_test)
# summarize results
confusionMatrix(predictions$class, y_test)

# Apply smote

data_train <- SMOTE(Class ~ ., data_train, perc.over = 100, perc.under=200)

prop.table(table(data_train$Class))


# train a naive bayes model
model <- NaiveBayes(Class~., data=data_train)

# make predictions
x_test <- data_test[,2:5]
y_test <- data_test[,1]
predictions <- predict(model, x_test)
# summarize results
confusionMatrix(predictions$class, y_test)



