setwd("C:\\Users\\Emfsupport.kwb1\\Downloads\\Analytics Vidya\\Loan_prediction")
train_data=read.csv("train_ctrUa4K.csv",stringsAsFactors = F)
test_data=read.csv("test_lAUu6dG.csv",stringsAsFactors = F)
library(dplyr)
glimpse(train_data)
train_data$data="train"
test_data$data="test"
test_data$Loan_status=NA
test_data=test_data %>% 
  select(-Loan_status)
test_data$Loan_Status=NA
loan=rbind(train_data,test_data)
#######################################data preparation starts######################
glimpse(loan)
summary(loan)
table(loan$Gender)
loan$Gender=ifelse(loan$Gender=="Male",1,0)
table(loan$Married)
loan$Married=ifelse(loan$Married=="Yes",1,0)
table(loan$Dependents)
loan1=loan %>% 
  mutate(Dependents=ifelse(Dependents=="3+",3,Dependents),
         Dep_1=as.numeric(Dependents %in% 1),
         Dep_2=as.numeric(Dependents %in% 2),
         Dep_0=as.numeric(Dependents %in% 0),
         Self_Employed=ifelse(Self_Employed=="Yes",1,0)) %>% 
  select(-Dependents)
glimpse(loan1)
table(loan1$Education)
loan1$Education=ifelse(loan1$Education=="Graduate",1,0)
table(loan1$Property_Area)
loan2=loan1 %>% 
  mutate(Area=as.numeric(Property_Area %in% c("Semiurban")),
         Area1=as.numeric(Property_Area %in% c("Urban"))) %>% 
  select(-Property_Area)
table(loan2$Loan_Amount_Term)
loan2=loan2 %>% 
  mutate(Loan_TERM=as.numeric(Loan_Amount_Term %in% 360)) %>% 
  select(-Loan_Amount_Term)

glimpse(loan2)
table(loan2$Credit_History)
loan2$Loan_Status=ifelse(loan2$Loan_Status=="Y",1,0)
lapply(loan2,function(x) sum(is.na(x)))
##we will treat missing values after seperating the train data and test data#######
train1=loan2 %>% 
  filter(data=="train") %>% 
  select(-data)
test1=loan2 %>% 
  filter(data=="test") %>% 
  select(-data,-Loan_Status)
glimpse(train1)
glimpse(test1)
#####################will treat the missing values###################

summary(train1)
lapply(train1, function(x) sum(is.na(x)))
train1$LoanAmount[is.na(train1$LoanAmount)]=median(train1$LoanAmount,na.rm = T)
train1$Credit_History[is.na(train1$Credit_History)]=median(train1$Credit_History,na.rm = T)
lapply(train1, function(x) sum(is.na(x)))
lapply(test1,function(x) sum(is.na(x)))
test1$LoanAmount[is.na(test1$LoanAmount)]=median(test1$LoanAmount,na.rm = T)
test1$Credit_History[is.na(test1$Credit_History)]=median(test1$Credit_History,na.rm = T)
#####now we create the validation dataset ########################
set.seed(3)
s=sample(1:nrow(train1),0.7*nrow(train1))
train_data1=train1[s,]
train_val=train1[-s,]
nrow(train_data1)
nrow(train_val)
summary(train_data1)
glimpse(train_data1)
#############apply linear regression#########################
fit=lm(Loan_Status~.-Loan_ID,data = train1)
summary(fit)
step(fit)
sort(vif(fit),decreasing = T)[1:3]

fit=lm(Loan_Status ~ Married + Education + CoapplicantIncome + Credit_History + 
         Dep_1 + Area,data = train1)
summary(fit)
library(car)
sort(vif(fit),decreasing = T)[1:3]

library(randomForest)
randfit=randomForest(as.factor(Loan_Status) ~ Married + Education + CoapplicantIncome + Credit_History + 
                       Dep_1 + Area,data = train1)
rand_predict=predict(randfit,newdata = train1,type = "response")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(as.factor(train1$Loan_Status),rand_predict,positive = levels(as.factor(train1$Loan_Status))[2])
###apply cv tunning####################
train1$Loan_Status=as.factor(train1$Loan_Status)
library(cvTools)
library(robustbase)
param_list=list(mtry=c(3,4),
                ntree=c(500,550),
                nodesize=c(4,5),
                maxnodes=c(9,11))
param_list_comb=expand.grid(param_list)

my_cost_function=function(y_test,Y_pred){
  roc_res=pROC::roc(y_test,Y_pred,quiet=TRUE)
  auc_result=pROC::auc(roc_res)
  return(auc_result)
}
init_auc_score=-999
for(i in 1:16){
  print(i)
  param_list_used=param_list_comb[i,]
  rf_fit=cvTuning(randomForest,Loan_Status ~.-Loan_ID ,data = train1,
                  tuning = param_list_used,
                  folds = cvFolds(nrow(train1),K=5,type = "random"),
                  cost = my_cost_function,
                  predictArgs = list(type = c("prob")))
  current_auc=rf_fit$cv[,2]
  if(current_auc>init_auc_score){
    init_auc_score=current_auc
    print(param_list_used)
  }
}
init_auc_score

##########applying randomforest with the best combinations received through#############################
randfit1=randomForest(Loan_Status ~.-Loan_ID ,data = train1,mtry=3,ntree=550,nodesize=5,maxnodes=9)
randpredict1=predict(randfit1,newdata = test1,type = "response")
confusionMatrix(train1$Loan_Status,randpredict1,positive = levels(train1$Loan_Status)[2])
randpredict1=ifelse(randpredict1==1,"Y","N")
write.csv(randpredict1,"newprediction.csv",row.names = F)
