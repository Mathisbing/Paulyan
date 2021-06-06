##-----------数据处理--------------
voice <- read.csv("voice.csv", stringsAsFactors = F)
table(voice$label)

##热力图
library("corrplot")
library("ggplot2")
library("caret")
library("tidyr")
voice_cor <- cor(voice[, 1:20])
##20个特征之间的关系
corrplot.mixed(voice_cor, tl.col='black', tl.pos = "lt", tl.cex = 0.8, number.cex=0.45)

##不同性别之间的每个特征具有的差异
plotdata <- gather(voice, key="variable", value="value", c(-label))
ggplot(plotdata, aes(fill = label))+
        theme_bw()+ 
        geom_density(aes(value),alpha = 0.5)+
        facet_wrap(~variable, scales = "free")
                                   

##---------------逻辑回归----------------
voice$label[which(voice$label =='female')] <- 0
voice$label[which(voice$label =='male')] <- 1
n <- nrow(voice)
ntrain<-round(2*n/3)
ntest<-n-ntrain
train<-sample(n,ntrain)
data_train <- voice[train,]
data_test <- voice[-train,]
voice_lm <- glm(as.factor(label)~. ,data = data_train, family = binomial)
summary(voice_lm)

##分析多重线性性导致的数据奇异性
kappa(voice_lm)
alias(voice_lm)

##逐步回归筛选变量
voice_step <- step(voice_lm, direction = "both")
summary(voice_step)
kappa(voice_step)

##逐步回归可视化
stepanova <- voice_step$anova
stepanova$Step <- as.factor(stepanova$Step)
ggplot(stepanova, aes(x = reorder(Step, -AIC), y = AIC)) +
  theme_bw(base_size = 12)+
  geom_point(colour = "red", size =2)+
  geom_text(aes(y=AIC-1, label = round(AIC, 2)))+
  theme(axis.text.x = element_text(angle = 30, size = 12))+
  labs(x = "deleted features")

##在测试集上预测
pred_voice <- predict(voice_lm, newdata=data_test, type='response')
perf <- table(data_test$label,pred_voice>0.5)
print(perf)
err_logreg <- 1-sum(diag(perf))/ntest  # error rate
print(err_logreg)

##pred_voice2 <- as.factor(ifelse(pred_voice > 0.5, 1, 0))
pred_voice3 <- predict(voice_step, newdata = data_test, type = 'response')
perf3 <- table(data_test$label, pred_voice3 > 0.5)
print(perf3)
err_logreg3 <- 1-sum(diag(perf3))/ntest  # error rate

##-----------------决策树--------------------
library("rpart")
library("rpart.plot")
voice_dt<- rpart(label~. ,data = data_train,  method="class", cp = 0)
rpart.plot(voice_dt,type = 2, extra = "auto",cex = 0.7, under = T, fallen.leaves = F,
           main="Decision Tree")
pred_voice_dt <- predict(voice_dt,newdata=data_test,type='class')
y.test <- data_test[,"label"]
table(y.test,pred_voice_dt)
err_dt1<-1-mean(y.test==pred_voice_dt)
print(err_dt1)

##决策树剪枝
plotcp(voice_dt)
bestcp <- voice_dt$cptable[which.min(voice_dt$cptable[, "xerror"]), "CP"]
print(bestcp)
voice_dt_pruned <- prune(voice_dt, cp = bestcp)
rpart.plot(voice_dt_pruned,type = 2, extra = "auto",cex = 0.7, under = T, fallen.leaves = F,
           main="Pruned Tree")

pred_voice_dt2 <- predict(voice_dt_pruned,newdata=data_test,type='class')
y.test <- data_test[,"label"]
table(y.test,pred_voice_dt2)
err_dt2<-1-mean(y.test==pred_voice_dt2)
print(err_dt2)


##------------------随机森林---------------------
library(randomForest)
voice_RF<-randomForest(as.factor(label)~. , data = data_train , mtry=3, ntree=300,
                      importance=TRUE)

trainerror <- as.data.frame(plot(voice_RF))
colnames(trainerror) <- paste("error", colnames(trainerror), sep = "")
trainerror$ntree <- 1:nrow(trainerror)
trainerror <- gather(trainerror, key = "Type", value = "Error", 2:3)

plot(voice_RF)
legend("topright",legend=c("err0","err1"),col=c("red","blue"),lty=1,lwd=2) 

ggplot(trainerror, aes(x = ntree, y=Error))+
  geom_line(aes(linetype = Type, colour = Type))

varImpPlot(voice_RF, pch = 20, main = "importance")

pred_voice_RF<-predict(voice_RF,newdata=data_test,type="class")
y.test <- data_test[,"label"]
err_RF<-1-mean(y.test==pred_voice_RF)
print(err_RF)


##------------------错误比较---------------------
##绘制ROC曲线
# Plot of ROC curves
library(ROCR)
library(pROC)

roc_logreg<-roc(data_test$label,as.vector(pred_voice))
plot(roc_logreg)

roc_step<-roc(data_test$label,as.vector(pred_voice3))
plot(roc_step,add=TRUE,col='green')

roc_dt<-roc(data_test$label,as.numeric(pred_voice_dt))
plot(roc_dt,add=TRUE,col='red')

roc_pt<-roc(data_test$label,as.numeric(pred_voice_dt2))
plot(roc_pt,add=TRUE,col='blue')

roc_rf<-roc(data_test$label,as.numeric(pred_voice_RF))
plot(roc_rf,add=TRUE,col='yellow')
legend("topright",legend=c("logreg","step","tree", "prTree","RF")
       ,col=c("black","green", "red", "blue", "yellow"),lty=1,lwd=2,cex=0.5) 

auc_log <- auc(data_test$label, as.vector(pred_voice))
auc_step <- auc(data_test$label, as.vector(pred_voice3))
auc_dt <- auc(data_test$label, as.numeric(pred_voice_dt))
auc_dt2 <- auc(data_test$label, as.numeric(pred_voice_dt2))
auc_RF <- auc(data_test$label, as.numeric(pred_voice_RF))


g1 <- ggroc(roc_logreg, # roc()函数创建的对象
      alpha = 0.5, # 设置曲线透明度
      colour = "black",  # 设置曲线颜色
      linetype = 1, size = 1)


roc <- rbind.data.frame(roc_logreg,roc_step,roc_dt,roc_pt,roc_rf)



## 10 folds cross valitation
k <- 10
folds <- sample(1:k, ntrain,replace = TRUE ) 
ERR<-matrix(0,k,5)
for(i in (1:k)){
  ##print(i)
  ##train<-sample(n,ntrain)
  ##data_train<-voice[train,]
  ##data_test<-voice[-train,]
  fit_logreg <- glm(as.factor(label)~. ,data = data_train[folds != i,],family = binomial(link = "logit"),trace=FALSE)
  ##pred_logreg <- predict(fit_logreg, newdata=data_train[folds = i], type='response')
  ##perf <- table(data_test$label,pred_voice>0.5)
  pred_logreg<-predict(fit_logreg,newdata=data_train[folds == i,],type='response')
  pred_logreg =ifelse(pred_logreg>0.5,1,0)
  ERR[i,1]<-sum(pred_logreg != data_train$label[folds == i])
  ##ERR[i,1] <- 1-sum(diag(perf))/ntest
  ##ERR[i,1] <- mean(data_test$label != (pred_logreg>0.5))
  
  
  fit_step <- step(fit_logreg, direction = "both")
  ##pred_step <- predict(voice_step, newdata = data_test, type = 'response')
  ##perf2 <- table(data_test$label,pred_step>0.5)
  ##ERR[i,2] <- 1-sum(diag(perf2))/ntest
  ##ERR[i,2] <- mean(data_test$label != (pred_step>0.5))
  pred_step <- predict(voice_step, newdata = data_train[folds==i,], type = "response")
  pred_step =ifelse(pred_step>0.5,1,0)
  ERR[i,2]<-sum(pred_step!=data_train$label[folds==i])
  
  
  ##fit_dt <- rpart(label~. ,data = data_train, method="class", cp = 0)
  ##pred_dt <- predict(fit_dt, newdata=data_test, type='class')
  ##ERR[i,3]<-mean(data_test$label !=pred_dt)
  fit_dt <- rpart(label~., data = data_train[folds != i,], method = "class", cp=0)
  pred_dt <- predict(fit_dt, newdata=data_train[folds==i,], type = "class")
  ERR[i,3]<-sum(pred_dt!=data_train$label[folds==i])
  
  
  bestcp <- fit_dt$cptable[which.min(fit_dt$cptable[, "xerror"]), "CP"]
  fit_pt <- prune(fit_dt, cp = bestcp)
  ##pred_pt <- predict(fit_pt ,newdata=data_test ,type='class')
  ##ERR[i,4]<-mean(data_test$label !=pred_pt)
  pred_pt <- predict(fit_pt, newdata = data_train[folds == i,], type = "class")
  ERR[i,4]<-sum(pred_pt!=data_train$label[folds==i])
  
  
  ##fit_RF <- randomForest(as.factor(label)~. , data = data_train , mtry=3, ntree=820,
                        ## importance=TRUE)
  ##pred_RF <- predict(fit_RF,newdata=data_test, type="class")
  ##ERR[i,5]<-mean(data_test$label != pred_RF)
  fit_RF <- randomForest(as.factor(label)~. ,data = data_train[folds != i,],
                         mtry=3, ntree =300, improtance=TRUE)
  pred_RF <-predict(fit_RF, newdata = data_train[folds ==i, ], type = "class")
  ERR[i,5]<-sum(pred_RF!=data_train$label[folds==i])
}
print(ERR)
Err_cv <- colSums(ERR)/(ntrain)
print(Err_cv)

## 10 replication

M<-10
ERR_1<-matrix(0,M,5)
for(i in 1:M){
  print(i)
  train<-sample(n,ntrain)
  data_train<-voice[train,]
  data_test<-voice[-train,]
  fit_logreg <- glm(as.factor(label)~. ,data = data_train,family = binomial)
  pred_logreg <- predict(fit_logreg, newdata=data_test, type='response')
  perf <- table(data_test$label,pred_logreg>0.5)
  ERR_1[i,1] <- 1-sum(diag(perf))/ntest
  
  fit_step <- step(fit_logreg, direction = "both")
  pred_step <- predict(fit_step, newdata = data_test, type = 'response')
  perf2 <- table(data_test$label,pred_step>0.5)
  ERR_1[i,2] <- 1-sum(diag(perf2))/ntest
  
  fit_dt <- rpart(label~. ,data = data_train, method="class", cp = 0)
  pred_dt <- predict(fit_dt, newdata=data_test, type='class')
  ERR_1[i,3]<-mean(data_test$label !=pred_dt)
  
  bestcp <- fit_dt$cptable[which.min(fit_dt$cptable[, "xerror"]), "CP"]
  fit_pt <- prune(fit_dt, cp = bestcp)
  pred_pt <- predict(fit_pt ,newdata=data_test ,type='class')
  ERR_1[i,4]<-mean(data_test$label !=pred_pt)
  
  fit_RF <- randomForest(as.factor(label)~. , data = data_train , mtry=3, ntree=820,
   importance=TRUE)
  pred_RF <- predict(fit_RF,newdata=data_test, type="class")
  ERR_1[i,5]<-mean(data_test$label != pred_RF)
  
}
boxplot(ERR_1,ylab="Test error rate",names=c("Logreg","Step","Tree","PTree","Forest"))



##--------------k-means聚类------------------
data_train[, 1:20] <- scale(data_train[, 1:20])
x<-data_train[, 1:20]

c<-2
voice_km <- kmeans(x, centers = c, nstart = 10)

library(cluster)
clusplot(x, voice_km$cluster, main = "kmean")



plot(data_train, col = voice_km$cluster, pch = voice_km$cluster)
points(voice_km$centers, pch = 15)