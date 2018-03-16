# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)
library(ggplot2)
library(OneR)
library(ROCR)

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
  # set.seed(12)
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
  
  cf <- confusionMatrix(pred, fit.gbm$trainingData$.outcome, mode = "everything")
  print(cf)
  
  
  
  # Random Forest
  set.seed(3033)
  fit.rf <- train(Class~., data=training_data, method="rf", trControl=control)
  
  pred <- predict(fit.rf, training_data) 
  
  cf <- confusionMatrix(pred, fit.rf$trainingData$.outcome, mode = "everything")000
  
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
  pred <- predict(fit.c45, training_data) 
  
  cf <- confusionMatrix(pred, fit.c45$trainingData$.outcome, mode = "everything")
  
  print(cf)
  
  
  # Logistic Regression
  set.seed(12)
  fit.logic <- train(Class~., data=training_data, method="LogitBoost", trControl=control)
  pred <- predict(fit.logic, training_data) 
  cf <- confusionMatrix(pred, fit.logic$trainingData$.outcome, mode = "everything")
  print(cf)
  
  # collect resamples
  results <- resamples(list(DecisionTree.CART=fit.cart,Bayesian=fit.bayes , SupportVectorMachine=fit.svm, kNN=fit.knn, RandomForest=fit.rf,NeuralNetwork=fit.nnet,GradientBoostingMachines=fit.gbm, DecisionTree.C45=fit.c45, LogisticRegression=fit.logic))
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
filename="3cixty-min-card.csv"
filename="3cixty-max-card.csv"

filename="dbp-min-card.csv"
filename="dbp-max-card.csv"

filename="dbo-range-SportsTeam.csv"

filename="3cixty-nice-place-range.csv"

ml_data<-load_dataSet(filename)

#for range
ml_data=data.frame(Class=ml_data$Label,prop=ml_data$prop)

# for range 3city
df2 <- ml_data[sample(nrow(ml_data)),]
ml_data=rbind(ml_data,df2)

nrow(ml_data)

ml_data$Class=factor(ml_data$Class)

set.seed(3033)

ml_data_smote<-smote_data(ml_data,100,200)

ml_data_smote$Class=factor(ml_data_smote$Class)




ml_data=ml_data_smote

# intrain <- sample(1:nrow(ml_data),size = 0.7*nrow(ml_data)) 
intrain <- createDataPartition(y = ml_data$Class, p= 0.7, list = FALSE)

training <- ml_data[intrain,]



nrow(training)

testing <- ml_data[-intrain,]

nrow(testing)

ml_data<-training

head(ml_data)


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

with_smote_data$models
# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# dot plots of results
dotplot(results)

library(reshape2) 

resamples<-results
plotData <- melt(resamples$values, id.vars = "Resample")

tmp <- strsplit(as.character(plotData$variable), "~", fixed = TRUE)
plotData$Model <- unlist(lapply(tmp, function(x) x[1]))
plotData$Metric <- unlist(lapply(tmp, function(x) x[2]))

plotData <- subset(plotData, Metric == fit.cart$metric)
plotData$variable <- factor(as.character(plotData$variable))
plotData <- split(plotData, plotData$variable)
results <- lapply(plotData, function(x, cl) {
  ttest <- try(t.test(x$value, conf.level = cl), silent = TRUE)
  if (class(ttest)[1] == "htest") {
    out <- c(ttest$conf.int, ttest$estimate)
    names(out) <- c("LowerLimit", "UpperLimit", "Kappa")
  }
  else out <- rep(NA, 3)
  out
}, cl = 0.95)
results <- as.data.frame(do.call("rbind", results))
tmp <- strsplit(rownames(results), "~", fixed = TRUE)
results$Model <- unlist(lapply(tmp, function(x) x[1]))

ggplot(results, aes(x = Model, y = Kappa)) + 
  geom_point() + 
  geom_errorbar(aes(ymin = LowerLimit, ymax = UpperLimit), width = .1) 

p <- ggplot(results, aes(Model, Kappa))
p +  geom_boxplot() + coord_flip()
#------------------
# 
# training$Class=factor(training$Class)
# 
# ml_data_smote<-smote_data(training,100,200)

testCard=testing

nrow(testCard)
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

fit.logic <- train(Class~., data=training, method="LogitBoost", trControl=Train_options)

test_pred <- predict( fit.logic  , testCard)


test_pred1 <- predict( fit.svm  , testCard)

test_pred2 <- predict( fit.rf   , testCard)

confusionMatrix(testCard$Class,test_pred)

class(testCard)

ROCcard=testCard
ROCcard = ifelse(testCard$Class=="M1", 1, 0)
ROcpred = ifelse(test_pred2=="M1", 1, 0) 

plot(roc(testCard$Class, predict(fit.rf, testCard, type = "prob")))

library(pROC)
auc <- roc(ROCcard, ROcpred)
print(auc)

plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

library(plotROC)

basicplot <- ggplot(test, aes(d = D, m = M1)) + geom_roc()

#----------------


fit_glm <- glm(Class~., data=training, family=binomial(link="logit"))

glm_link_scores <- predict(fit_glm, testCard, type="link")

glm_response_scores <- predict(fit_glm, testCard, type="response")

score_data <- data.frame(link=glm_link_scores, 
                         response=glm_response_scores,
                         bad_widget=testCard$Class,
                         stringsAsFactors=FALSE)

score_data %>% 
  ggplot(aes(x=link, y=response, col=bad_widget)) + 
  scale_color_manual(values=c("black", "red")) + 
  geom_point() + 
  geom_rug() + 
  ggtitle("Both link and response scores put cases in the same order")




#-------------Random foreset model


library('randomForest') 


ml_data<-load_dataSet(filename)

nrow(ml_data)

ml_data$Class=factor(ml_data$Class)

ml_data_smote<-smote_data(ml_data,100,200)

ml_data_smote$Class=factor(ml_data_smote$Class)



set.seed(3033)

# intrain <- sample(1:nrow(ml_data),size = 0.7*nrow(ml_data)) 

intrain <- createDataPartition(y = ml_data_smote$Class, p= 0.7, list = FALSE)

training <- ml_data_smote[intrain,]

nrow(training)

testing <- ml_data_smote[-intrain,]

nrow(testing)

ml_data<-training

head(ml_data)



ml_data_smote



rf_model <- randomForest(factor(Class) ~ .,
                         data = training)


plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

prediction <- predict(rf_model, testing)

confusionMatrix(testing$Class,prediction)


# best.guess <- mean(ml_data_smote$DataMin) 
# 
# RMSE.baseline <- sqrt(mean((best.guess-ml_data_smote$DataMin)^2))
# 
# RMSE.rtree <- sqrt(mean((test_pred - testCard$Class)^2))

#--------------------------------------------------

training  <- load_dataSet("dbp-card-training.csv")
unique(OntoProperty)

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

write.csv(training,"C:/Users/rifat/Desktop/R_milan/githubRepo/RDFShapeInduction/RDFShapeInduction/dataset/dbp-min-training.csv",row.names =   FALSE)



# nrow(testing)
# 
# testing<-testing[,c(2,3)]

prediction <- predict(modelFit, testing)


testing$PredMin<-prediction

write.csv(testing,"C:/Users/rifat/Desktop/R_milan/githubRepo/RDFShapeInduction/RDFShapeInduction/dataset/dbp-person-min-cardinality.csv",row.names =   FALSE)


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

write.csv(training,"C:/Users/rifat/Desktop/R_milan/githubRepo/RDFShapeInduction/RDFShapeInduction/dataset/dbp-max-training.csv",row.names =   FALSE)


# nrow(testing)
# 
# testing<-testing[,c(2,3)]

prediction <- predict(modelFit, testing)


testing$PredMax<-prediction


write.csv(testing,"C:/Users/rifat/Desktop/R_milan/githubRepo/RDFShapeInduction/RDFShapeInduction/dataset/dbp-person-max-cardinality.csv",row.names =   FALSE)


#---------

library(caret)
train <- ml_data_smote

tc <- trainControl("cv",10)

rpart.grid <- expand.grid(.cp=0.2)

# Convert variable interpreted as integer to factor

(train.rpart <- train(  Class~., 
                        data=train, 
                        method="rf",
                        trControl=tc,
                        na.action = na.omit,
                        tuneGrid=rpart.grid))

control <- trainControl(method="cv", number=10)

fit.rf <- train(Class~., data=train, method="rf", trControl=control)

pred <- predict(fit.rf, train) 

cf <- confusionMatrix(pred, fit.rf$trainingData$.outcome, mode = "everything")

print(cf)


# Predict
pred <- predict(train.rpart, train) 

# Produce confusion matrix from prediction and data used for training
cf <- confusionMatrix(pred, train.rpart$trainingData$.outcome, mode = "everything")
print(cf)

fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)


nnetGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

nnetFit <- train(Class ~ ., 
                 data = train,
                 method = "nnet",
                 metric = "ROC",
                 trControl = fitControl,
                 tuneGrid = nnetGrid,
                 verbose = FALSE)

pred <- predict(nnetFit, train) 

cf <- confusionMatrix(pred, nnetFit$trainingData$.outcome, mode = "everything")
print(cf)


# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

