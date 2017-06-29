library(DMwR)
library(readxl)

# Specify sheet with a number or name
test_data<-read_excel("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/3cixtyData.xlsx", sheet = "Sheet3")

test_data=as.data.frame(test_data)

write.csv(test_data, file = "C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty.csv",row.names = FALSE)

head(test_data,10)

table(test_data$ClassMax)

print(prop.table(table(test_data$ClassMax)))

test_data$ClassMax=factor(test_data$ClassMax)

test_data[,6:ncol(test_data)] <- lapply(test_data[,6:ncol(test_data)], function (x) as.factor(as.numeric(x)))

sapply(test_data,class)

head(test_data)

class(test_data$Class)

#perc.over-- A number that drives the decision of how many extra cases from the minority 
# class are generated (known as over-sampling).
#perc.under--A number that drives the decision of how many extra cases from the majority 
# classes are selected for each case generated from the minority class (known as under-sampling)

newData <- SMOTE(ClassMax ~ ., test_data, perc.over = 600, perc.under=100)

table(newData$ClassMax)

print(prop.table(table(newData$Class)))

write.csv(newData, file = "C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty-max-smote.csv",row.names = FALSE)



# Data from rapid miner

test_data<- read.csv("C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty.csv", header = TRUE)
head(test_data)

test_sample_data<-test_data[,-c(1,3)]
head(test_sample_data)

# Same data with max samaple 
write.csv(test_sample_data, file = "C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty-max-card.csv",row.names = FALSE)

test_sample_data$ClassMax=factor(test_sample_data$ClassMax)

test_sample_data[,6:ncol(test_sample_data)] <- lapply(test_sample_data[,6:ncol(test_sample_data)], function (x) as.factor(as.numeric(x)))
sapply(test_sample_data,class)

table(test_sample_data$ClassMax)

# Apply SMOTE sampling

test.smote <- SMOTE(ClassMax ~ ., test_sample_data, perc.over = 600, perc.under=100)

table(test.smote$ClassMax)

write.csv(test.smote, file = "C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty-max-card-smote.csv",row.names = FALSE)

# ROSE Sampling
library(ROSE)
test.rose <- ROSE(ClassMax ~., data = test_sample_data, seed = 1)$data
table(test.rose$ClassMax)

write.csv(test.rose, file = "C:/Users/rifat/Desktop/R_milan/KB_Integrity_Constraints/rapidminer/data/3cixty-max-card-rose.csv",row.names = FALSE)




