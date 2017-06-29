library(DMwR)
## ML algorithms
# load libraries
library(mlbench)
library(caret)
# load the dataset

min_card<-read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/3cixty-min-card.csv",header = TRUE)

max_card<-read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/3cixty-max-card.csv",header = TRUE)

head(min_card)
head(max_card)

min_card$Class=factor(min_card$Class)

max_card$Class=factor(max_card$Class)

prop.table(table(min_card$Class))

prop.table(table(max_card$Class))


#perc.over-- A number that drives the decision of how many extra cases from the minority 
# class are generated (known as over-sampling).
#perc.under--A number that drives the decision of how many extra cases from the majority 
# classes are selected for each case generated from the minority class (known as under-sampling)

min_card_smote <- SMOTE(Class ~ ., min_card, perc.over = 600, perc.under=100)

prop.table(table(min_card_smote$Class))

max_card_smote <- SMOTE(Class ~ ., max_card, perc.over = 600, perc.under=100)

prop.table(table(max_card_smote$Class))

head(min_card)

ML_algorithms<-function(training_data){

# prepare training scheme ~ 10-fold cross validation
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
 

results<-ML_algorithms(min_card_smote)
# summarize differences between modes
summary(results)

# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


