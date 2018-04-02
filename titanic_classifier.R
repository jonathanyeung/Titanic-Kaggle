library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('gbm')
require(gbm)
require(MASS)

#Notes
# Titanic Deck Plans: https://www.encyclopedia-titanica.org/titanic-deckplans/b-deck.html


train <- read.csv('./train.csv', stringsAsFactors = F)
test  <- read.csv('./test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data

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

# Show title counts by sex again
table(full$Sex, full$Title)

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
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# Show family size by survival using a mosaic plot
#mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

full$Fare[is.na(full$Fare)] <- with(full, ave(Fare,
                                          FUN = function(x) median(x, na.rm = TRUE)))[is.na(full$Fare)]

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

full$cabinCount <- sapply(full$Cabin, function(x) max(1,length(strsplit(x, " ")[[1]])))

full <- full %>% mutate(normalizedTicketCost = Fare / cabinCount)

deck_predict_C <- full %>% 
  filter(Embarked == "C")

ggplot(deck_predict_C, aes(x = Deck, y = normalizedTicketCost)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()


deck_predict_S <- full %>% 
  filter(Embarked == "S")

ggplot(deck_predict_S, aes(x = Deck, y = normalizedTicketCost)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()


deck_predict_Q <- full %>% 
  filter(Embarked == "Q")

ggplot(deck_predict_Q, aes(x = Deck, y = normalizedTicketCost)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()




ggplot(full, aes(x = Deck, y = normalizedTicketCost, fill = factor(Embarked))) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

ggplot(full, aes(x = Deck, y = normalizedTicketCost, fill = factor(Pclass))) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()



deck_graph <- full[1:891,] %>% 
  filter(!is.na(Deck) & Sex == "male")


ggplot(deck_graph, aes(x = Deck, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Deck') +
  theme_few()


age_male_graph <- full[1:891,] %>% 
  filter(!is.na(Age) & Sex == "male")

age_neutral_graph <- full[1:891,] %>% 
  filter(!is.na(Age) & Pclass == 3)

ggplot(age_male_graph, aes(x = Age, fill = factor(Survived))) +
  geom_histogram(binwidth = 4, position="identity",  alpha=.5)

# Fill in missing ages:

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Family', 'FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Survived', 'cabinCount', 'Deck')], method='rf') 

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


# Perform mice imputation, excluding certain less-than-useful variables for Deck.
mice_deck_mod <- mice(full[, names(full) %in% c('normalizedTicketCost', 'Embarked', 'Deck', 'PClass', 'FsizeD', 'Title')], method='cart') #, m = 10, maxit=10) 

# Save the complete output 
mice_deck_output <- complete(mice_deck_mod)

full["Deck_Predict"] <- mice_deck_output$Deck


deck_full_predict_Q <- full #%>% filter(Embarked == "S")

plot1 <- ggplot(deck_full_predict_Q, aes(x = Deck, y = normalizedTicketCost)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

plot2 <- ggplot(deck_full_predict_Q, aes(x = Deck_Predict, y = normalizedTicketCost)) +
  geom_boxplot() +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

require(gridExtra)
grid.arrange(plot1, plot2, ncol=2)


#Bucket to age groups
full$AgeGroup[full$Age <= 16] <- 'Child'
full$AgeGroup[full$Age > 16 & full$Age <= 30] <- 'Young Adult'
full$AgeGroup[full$Age > 30 & full$Age <= 45] <- 'Adult'
full$AgeGroup[full$Age > 45] <- 'Elderly'
full$AgeGroup  <- factor(full$AgeGroup)

# Show counts
table(full$AgeGroup, full$Survived)

train <- full[1:891,]
test <- full[892:1309,]

set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + AgeGroup + SibSp + Parch + Title + FsizeD + normalizedTicketCost + Deck_Predict,
                         data = train)
  
# Show model error
plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


set.seed(190)

# GBM Documentation: https://www.rdocumentation.org/packages/gbm/versions/2.1.1/topics/gbm

gbm_train <- train[c("Survived", "Pclass", "Sex", "AgeGroup", "SibSp", "Parch", "Title", "FsizeD", "normalizedTicketCost", "Deck_Predict")]
# Build the model (note: not all possible variables are used)
gbm_model <- gbm(factor(Survived) ~ Pclass + Sex + AgeGroup + SibSp + Parch + Title + FsizeD + normalizedTicketCost + Deck_Predict,
                         data = gbm_train,
                 var.monotone=c(0,0,0,0,0,0,0,0,0),distribution="gaussian",n.trees=1000,shrinkage=0.05,

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


# Predict using the test set
prediction_gbm <- predict(gbm_model, test,best.iter)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution_gbm <- data.frame(PassengerID = test$PassengerId, Survived = prediction_gbm)

write.csv(solution_gbm, file = 'gbm_mod_Solution.csv', row.names = F)




# Predict using the test set
prediction_rf <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction_rf)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)


