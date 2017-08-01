# Random forest based cardinality estimation
#

# load libraries
library(DMwR)
library(mlbench)
library(caret)
library(klaR)
library(randomForest)

# Function for smote 

smote_data<-function(data,over.val,under.val){
  
  re_somte_data <- SMOTE(Class ~ ., data, perc.over=over.val, perc.under=under.val)
  
  return(re_somte_data)  
}

training  <- load_dataSet("dbp-card-training.csv")
attach(training) #attaching data frame to reduce the length of the variable names associated to it.
training<-training[,c(2,3,5)]
head(training)

testing   <-  load_dataSet("dbo_person_test.csv")

testing<-testing[,c(2,3,5)]

head(testing)

# exploring the data

tail(names(training),24)
summary(training$AntMin)

training$Class=factor(training$AntMin)

# training<-smote_data(training,100,200)

summary(training$Class)


is.factor(OntoClass)
#is.factor(AntMax)
#str(OntoProperty)

# library(ggplot2)
# library(caret)
# ## Loading required package: lattice
# #selecting a few of the more promising predictors to be plotted
# colSelection<- c("DataMin","OntoProperty")
# 
# #creating a feature plot 
# featurePlot(x=training[,colSelection],y = training$AntMin, plot="pairs")
# 
# # Min cardinality
# qplot(DataMin, DataMax, colour=AntMin, data=training)

#-----


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
  # modelFit<-  rpart(Class ~., data = trainingSet)
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
# modelFit<-  rpart(Class ~., data = training)

nrow(testing)

# testing<-testing[,c(2,3)]

prediction <- predict(modelFit, testing)


testing$prediction<-prediction

minCardinality<-testing
