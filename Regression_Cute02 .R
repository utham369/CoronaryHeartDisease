rm(list=ls())

#Importing libraries
library(ROSE)
library(ggplot2)
library(DMwR)
library(ROCR) 
#library(car)
library(caret)
library(MASS)
library(corrplot)
library(glmnet)

#read data
chd_data <- read.csv("data_20180408.csv" , header=T , na.strings = c(NA,'-99'))
#Structure of chd_data
str(chd_data)
colnames(chd_data)
summary(chd_data)


#Solutions :  Synthetic Points(SMOTE) , Under Sampling , Over Sampling(repeating)
prop.table(table(chd_data$Target))

#bar plot with target variable
ggplot(chd_data,aes(x=Target))+ geom_bar(fill='grey' ,color ="blue") +
 geom_text(stat="count",aes(label=..count..))+ theme_classic() +  ggtitle('Class Distribution(Target)')


#Plot to notify the missing IDs in the dataset
ggplot(chd_data,aes(x=ID , y = Target))+ geom_point(fill='grey' ,color ="blue") +
  ggtitle('Class Distribution(Target)')



chd_data$Target <- as.factor(chd_data$Target)
sum(is.na(chd_data))
colSums(is.na(chd_data))

#ID column has no use in our model building
chd_data$ID <- NULL
str(chd_data)
#Removing A11 column as it contains same value for all observations , No use formodel building
range_chd <- apply(chd_data , 2 , function(x){ range(x) })
chd_data$A11 <- NULL



### Splitting the data into Train/Test Manually
set.seed(123)
rows = seq(1,nrow(chd_data),1)
trainRows = sample(rows,(70*nrow(chd_data))/100)
train_set = chd_data[trainRows,]
validation_set = chd_data[-trainRows,]
str(train_set)


# #Imputed train set with its respective mean
train_set$A2[which(is.na(train_set$A2))]   <- mean(train_set$A2 , na.rm = T)
train_set$A15[which(is.na(train_set$A15))] <- mean(train_set$A15 , na.rm = T)
train_set$A16[which(is.na(train_set$A16))] <- mean(train_set$A16 , na.rm = T)

#Imputed test set with train mean
validation_set$A2[which(is.na(validation_set$A2))]   <- mean(train_set$A2 , na.rm = T)
validation_set$A15[which(is.na(validation_set$A15))] <- mean(train_set$A15 , na.rm = T)
validation_set$A16[which(is.na(validation_set$A16))] <- mean(train_set$A16 , na.rm = T)

colSums(is.na(train_set))
colSums(is.na(validation_set))
str(train_set)

#After splitting , Checking target variable proportionality 
#in train and validation set
prop.table(table(train_set$Target))       
prop.table(table(validation_set$Target)) #proportionality looks fine same as original data set


# SMOTE the data
train_set <- SMOTE(Target ~ ., train_set , perc.over = 100 , perc.under=200)
#trainSplit$target <- as.numeric(trainSplit$target)
print(prop.table(table(train_set$Target)))
table(train_set$Target)

train_Catr <-  subset(train_set , select = c(A13 , A17 , A18 , A19, A20 , A22))
train_Num  <-  subset(train_set , select = -c(A13 , A17 , A18 , A19, A20 , A22))
str(train_Num)
train_Catr <- data.frame(apply(train_Catr , 2 , function(x){as.factor(x)}))
train_Num$Target <- NULL
train_Num  <- data.frame(apply(train_Num , 2 , function(x){as.numeric(x)}))
str(train_Num)
str(train_Catr)
train_target     <- train_set$Target
str(train_Num)

#Validation
val_Catr <-  subset(validation_set , select = c(A13 , A17 , A18 , A19, A20 , A22))
val_Num  <-  subset(validation_set , select = -c(A13 , A17 , A18 , A19, A20 , A22))
val_Catr <- data.frame(apply(val_Catr , 2 , function(x){as.factor(x)}))
val_Num$Target <- NULL
val_Num  <- data.frame(apply(val_Num , 2 , function(x){as.numeric(x)}))
str(val_Num)
str(val_Catr)
sum(is.na(val_Num))
colSums(is.na(val_Num))
val_target     <- validation_set$Target

#To check the corelation b/w independent variables 
corelation_data <- round(cor(train_Num) ,2)


#### Remove unneccesary objects
#rm(list = setdiff(ls(),c("train","validation" , "train_Num" , "val_Num", "train_Catr" , "val_Catr")))
rm(list = setdiff(ls(),c("trainRows","train_set","validation_set" , "train_Num" , "val_Num", "train_Catr" , "val_Catr")))

############  generalized Logistic Regression Model #####################
log_reg <- glm(Target~ ., data = train_set, family = binomial)
summary(log_reg)

# By default if no dataset is mentioned, training data is used
prob_train <- predict(log_reg, train_set , type="response")

pred <- prediction(prob_train, train_set$Target)
perf <- performance(pred, measure="tpr", x.measure="fpr")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
auc <- round(auc  , 4)
auc = auc * 100
print(auc)
legend(0.6 ,0.4 , auc, title = " AUC" , cex = 1.2)

pred_class <- ifelse(prob_train > 0.55, 1, 0)
table(train_set$Target,pred_class)


prob_val <- predict(log_reg, validation_set, type = "response")
table(validation_set$Target , prob_val > 0.55)
table(validation_set$Target)

preds_val <- ifelse(prob_val > 0.55, 1, 0)
preds_val <- as.factor(preds_val)
table(preds_val)


conf_matrix <- table(validation_set$Target, preds_val)
print(conf_matrix)
specificity <- conf_matrix[1, 1]/sum(conf_matrix[1, ])
sensitivity <- conf_matrix[2, 2]/sum(conf_matrix[2, ])
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
accuracy


confusionMatrix(preds_val, validation_set$Target ,positive="1")
precision =  conf_matrix[2, 2] / sum(conf_matrix[,2])
precision
Fmeasure <- 2 * precision * sensitivity / (precision + sensitivity)
Fmeasure

####################### STEP - AIC #######################
log_reg_aic = stepAIC(log_reg, direction = "backward")
summary(log_reg_aic)


# By default if no dataset is mentioned, training data is used
prob_train <- predict(log_reg_aic, train_set , type="response")
table(train_set$Target , prob_train > 0.5)
table(train_set$Target)

pred <- prediction(prob_train, train_set$Target)
perf <- performance(pred, measure="tpr", x.measure="fpr")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
auc <- round(auc  , 4)
auc = auc * 100
print(auc)
legend(0.6 ,0.4 , auc, title = " AUC" , cex = 1.2)

pred_class <- ifelse(prob_train > 0.5, 1, 0)
table(train_set$Target,pred_class)
table(train_set$Target)

prob_val <- predict(log_reg_aic, validation_set, type = "response")
table(validation_set$Target , prob_val > 0.5)
table(validation_set$Target)

preds_val <- ifelse(prob_val > 0.5, 1, 0)
preds_val <- as.factor(preds_val)
table(preds_val)


conf_matrix <- table(validation_set$Target, preds_val)

print(conf_matrix)
specificity <- conf_matrix[1, 1]/sum(conf_matrix[1, ])
specificity
sensitivity <- conf_matrix[2, 2]/sum(conf_matrix[2, ])
sensitivity
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
accuracy


confusionMatrix(preds_val, validation_set$Target ,positive="1")
precision =  conf_matrix[2, 2] / sum(conf_matrix[,2])
Fmeasure <- 2 * precision * sensitivity / (precision + sensitivity)




##################### VIF - Model #########################
ggpairs(data=train_set, columns=1:5,title="Coronary Heart Disease Dataset")
corrplot(train_set , method = "circle")

#Use vif to find any multi-collinearity
log_reg_vif = vif(log_reg)
log_reg_vif


#for loop to get all vif values gretaer than 10
drops = c()
for(i in 1 : length(log_reg_vif)){
  if(log_reg_vif[i] > 10){
    drops[i] = names(log_reg_vif[i])
  }
  
}
drops <- na.omit(drops)
drops

train  <- train_set[,!(names(train_set)%in% drops)]
validation    <- validation_set[,!(names(validation_set) %in% drops)]


log_reg <- glm(Target~., data = train, family = binomial)
summary(log_reg)

log_reg_aic = stepAIC(log_reg, direction = "backward")
summary(log_reg_aic)

# By default if no dataset is mentioned, training data is used
prob_train <- predict(log_reg, train , type="response")
table(train$Target , prob_train > 0.5)
table(train$Target)

pred <- prediction(prob_train, train$Target)
perf <- performance(pred, measure="tpr", x.measure="fpr")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
auc <- round(auc  , 4)
auc = auc * 100
print(auc)
legend(0.6 ,0.4 , auc, title = " AUC" , cex = 1.2)

pred_class <- ifelse(prob_train > 0.35, 1, 0)
table(train$Target,pred_class)
table(train$Target)

prob_val <- predict(log_reg, validation, type = "response")
table(validation$Target , prob_val > 0.35)
table(validation$Target)

preds_val <- ifelse(prob_val > 0.35, 1, 0)
preds_val <- as.factor(preds_val)
table(preds_val)


conf_matrix <- table(validation$Target, preds_val)

print(conf_matrix)
specificity <- conf_matrix[1, 1]/sum(conf_matrix[1, ])
specificity
sensitivity <- conf_matrix[2, 2]/sum(conf_matrix[2, ])
sensitivity
accuracy <- sum(diag(conf_matrix))/sum(conf_matrix)
accuracy


confusionMatrix(preds_val, validation$Target ,positive="1")
precision =  conf_matrix[2, 2] / sum(conf_matrix[2,])
Fmeasure <- 2 * precision * sensitivity / (precision + sensitivity)

