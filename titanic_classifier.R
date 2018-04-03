library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('gbm')
library('caret')
library('pROC')
library('e1071')
require(gbm)
require(MASS)


#Notes
# Titanic Deck Plans: https://www.encyclopedia-titanica.org/titanic-deckplans/b-deck.html


train <- read.csv('./train.csv', stringsAsFactors = F)
test  <- read.csv('./test.csv', stringsAsFactors = F)
full  <- bind_rows(train, test) # bind training & test data



#### Title Engineering ####

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# Visualize the relationship between title & survival
ggplot(full[1:891,], aes(x = Title, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge')


#### Family Size Feature Engineering ####

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size')

# Discretize family size
full$FsizeD[full$Fsize == 1] <- 'single'
full$FsizeD[full$Fsize == 2] <- 'small'
full$FsizeD[full$Fsize >= 3 & full$Fsize <= 4] <- 'medium'
full$FsizeD[full$Fsize >= 5] <- 'large'

# Family Size (bucketed) vs. Survival
ggplot(full[1:891,], aes(x = FsizeD, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Family Size')

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Family', 'FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

#### Fare Calculation ####

# Replace N/A and 0 Fares:
full$Fare[full$Fare == 0] <- NA

mice_fare_mod <- mice(full[, names(full) %in% c('Embarked', 'PClass', 'Fare')], method='cart') 
mice_fare_output <- complete(mice_fare_mod)

full$Fare <- mice_fare_output$Fare

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

full$cabinCount <- sapply(full$Cabin, function(x) max(1,length(strsplit(x, " ")[[1]])))

full <- full %>% mutate(normalizedTicketCost = (Fare / cabinCount))
full <- full %>% mutate(logTicketCost = log(Fare))

ggplot(full, aes(x = normalizedTicketCost)) +
  geom_histogram() +
  theme_few()


#### Deck Imputation ####
# Perform mice imputation, excluding certain less-than-useful variables for Deck.
mice_deck_mod <- mice(full[, names(full) %in% c('normalizedTicketCost', 'Embarked', 'Deck', 'PClass', 'FsizeD')], method='cart') #, m = 10, maxit=10) 

# Save the complete output 
mice_deck_output <- complete(mice_deck_mod)

full["Deck_Predict"] <- mice_deck_output$Deck

deck_plot <- full[c("Deck", "Deck_Predict", "normalizedTicketCost", "Fare")]
deck_plot$predicted <- is.na(deck_plot$Deck)

ggplot(deck_plot, aes(x = Deck_Predict, y = normalizedTicketCost, fill = predicted)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()


#### Age Engineering ####


# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, names(full) %in% c('Age', 'SibSP', 'Parch', 'Pclass')], method='rf') 

# Save the complete output 
mice_output <- complete(mice_mod)

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

#Bucket to age groups
full$AgeGroup[full$Age <= 16] <- 'Child'
full$AgeGroup[full$Age > 16 & full$Age <= 30] <- 'Young Adult'
full$AgeGroup[full$Age > 30 & full$Age <= 45] <- 'Adult'
full$AgeGroup[full$Age > 45] <- 'Elderly'
full$AgeGroup  <- factor(full$AgeGroup)

# Show counts
table(full$AgeGroup, full$Survived)


#### Modeling Preparation ####

train <- full[1:891,]
test <- full[892:1309,]
set.seed(754)

#### Random Forest Model ####
# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + AgeGroup + SibSp + Parch + Title + FsizeD + normalizedTicketCost + Deck_Predict,
                         data = train)
  
# Show model error
plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Predict using the test set
prediction_rf <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction_rf)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)




#### GBM with Caret Package ####
# GBM Documentation: https://www.rdocumentation.org/packages/gbm/versions/2.1.1/topics/gbm
# https://www.r-bloggers.com/predicting-titanic-deaths-on-kaggle-ii-gbm/
# https://amunategui.github.io/binary-outcome-modeling/

set.seed(190)

train$SurvivedFactor <- ifelse(train$Survived==1, 'yes', 'no')
train$SurvivedFactor <- as.factor(train$SurvivedFactor)

smp_size <- floor(0.85 * nrow(train))
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

train_train <- train[train_ind, ]
train_test <- train[-train_ind, ]

predictor_names <- c("Pclass", "Sex", "Age", "Title", "FsizeD", "logTicketCost", 'Embarked')

#Attempting GBM via Caret package
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
grid <- expand.grid(n.trees = seq(500,10000,500), interaction.depth = 3, shrinkage = .001, n.minobsinnode = 20)
objModel <- train(train_train[predictor_names], 
                  train_train$SurvivedFactor, 
                  method='gbm', 
                  trControl=objControl,
                  tuneGrid = grid,
                  metric = "ROC",
                  preProc = c("center", "scale"))

#plot(objModel)

predictions <- predict(object=objModel, train_test[predictor_names], type='raw')

print(postResample(pred=predictions, obs=as.factor(train_test$SurvivedFactor)))




# Predict using the test set
prediction_caret_gbm <- predict(object=objModel, test[predictor_names], type='raw')

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction_caret_gbm)
solution$Survived <- ifelse(solution$Survived == "yes", 1, 0)
# Write the solution to file
write.csv(solution, file = 'gbm_caret_Solution.csv', row.names = F)









#### GBM with GBM package ####

gbm_train <- train_train[c("Survived", "Pclass", "Sex", "AgeGroup", "SibSp", "Parch", "Title", "FsizeD", "normalizedTicketCost", "Deck_Predict")]
gbm_train$Survived <- as.factor(gbm_train$Survived)

# Build the model (note: not all possible variables are used)
# gbm_model <- gbm(Survived ~ Pclass + Sex + AgeGroup + SibSp + Parch + Title + FsizeD + normalizedTicketCost + Deck_Predict,
#                          data = gbm_train,
#                  var.monotone=c(0,0,0,0,0,0,0,0,0),distribution="bernoulli",n.trees=3000,shrinkage=0.01,
# 
#                  interaction.depth=3,         # 1: additive model, 2: two-way interactions, etc.
#                  bag.fraction = 0.5,          # subsampling fraction, 0.5 is probably best
#                  train.fraction = 0.8,        # fraction of data for training,
#                  n.minobsinnode = 10,         # minimum total weight needed in each node
#                  cv.folds = 3,                # do 3-fold cross-validation
#                  keep.data=TRUE,              # keep a copy of the dataset with the object
#                  verbose=FALSE,               # don't print out progress
#                  n.cores=1)

gbm_model <- gbm(Survived ~ Pclass + Sex + AgeGroup + SibSp + Parch + Title + FsizeD + normalizedTicketCost + Deck_Predict,
                         data = gbm_train,
                 var.monotone=c(0,0,0,0,0,0,0,0,0),distribution="gaussian",n.trees=3000,shrinkage=0.01,

                 interaction.depth=3,         # 1: additive model, 2: two-way interactions, etc.
                 bag.fraction = 0.5,          # subsampling fraction, 0.5 is probably best
                 train.fraction = 0.8,        # fraction of data for training,
                 n.minobsinnode = 10,         # minimum total weight needed in each node
                 cv.folds = 3,                # do 3-fold cross-validation
                 keep.data=TRUE,              # keep a copy of the dataset with the object
                 verbose=FALSE,               # don't print out progress
                 n.cores=1)


# check performance using an out-of-bag estimator
# OOB underestimates the optimal number of iterations
best.iter <- gbm.perf(gbm_model,method="OOB")
print(best.iter)

# check performance using a 50% heldout test set
best.iter <- gbm.perf(gbm_model,method="test")
print(best.iter)

# check performance using 5-fold cross-validation
best.iter <- gbm.perf(gbm_model,method="cv")
print(best.iter)

#Evaluate the model:
validate_prediction <- predict(gbm_model, train_test, best.iter)
target <- c(0,1)[train_test$Survived]
sapply(seq(0,2,.1),function(step) c(step, sum(ifelse(validate_prediction))))


# Predict using the test set
prediction_gbm <- predict(gbm_model, test,best.iter)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution_gbm <- data.frame(PassengerID = test$PassengerId, Survived = prediction_gbm)

write.csv(solution_gbm, file = 'gbm_mod_Solution.csv', row.names = F)




