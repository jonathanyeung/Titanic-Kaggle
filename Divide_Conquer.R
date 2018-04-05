library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('gbm')
library('caret')
library('e1071')

# define function to compute log likelihood of a/(1-a)
logl <- function(a) {
  a <- max(a,0.1); # avoids log(0)
  a <- min(a,0.9); # avoids division by 0
  return (log(a/(1-a)));
}


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
                 'Title','Family', 'FsizeD', 'Survived')

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


#### Group Assignment ####
full$FareFac <- factor(full$Fare);

full$TFreq <- ave(seq(nrow(full)), full$Ticket,  FUN=length)
full$FFreq <- ave(seq(nrow(full)), full$FareFac, FUN=length)
full$CFreq <- ave(seq(nrow(full)), full$Cabin,   FUN=length)


full$GID <- rep(NA, nrow(full))
maxgroup <- 12 # maximum size of a group
for(i in as.numeric(full$PassengerId)) {
  if(full$SibSp[i] + full$Parch[i] > 0) 
    { # Check first if ID has relatives 
    full$GID[i] <- paste0(full$Surname[i], full$SibSp[i] + full$Parch[i]); # SibSp and Parch are used to potentially differentiate different families with the same Surname.    
  } else {
    if(full$TFreq[i] > 1 & is.na(full$GID[i])) { # Next if shares ticket number with others 
      full$GID[i] <- as.character(full$Ticket[i]);
    } else {
      if(full$CFreq[i] > 1 & full$CFreq[i]<maxgroup & is.na(full$GID[i])) { # Next if shares cabin with others
        full$GID[i] <- as.character(full$Cabin[i]);
      }
      else {
        if(full$FFreq[i] > 1 & full$FFreq[i]<maxgroup & is.na(full$GID[i])) { # Next if shares Fare value with others
          full$GID[i] <- as.character(full$FareFac[i]);
        } else { 
          full$GID[i] <- "Single"; # Individual doesn't belong to any group
        }
      }
    }   
  }
}
full$GID <- factor(full$GID)



#### Modeling Preparation ####
for (sex in c("male", "female")) 
{
  for (class in c(1:3))
  {
    full$SLogL[full$Sex==sex & full$Pclass==class] <- 
      logl(nrow(full %>% dplyr::filter(Survived==1, Sex==sex, Pclass==class))/nrow(train %>% dplyr::filter(Sex==sex, Pclass==class)));
  }
}

# Examining SLogL for PClass
ggplot(train, aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived))

#### Random Optimizations ####

ticket_stats <- full %>% group_by(Ticket) %>% summarize(l = length(Survived), na = sum(is.na(Survived)), c = sum(as.numeric(Survived)-1, na.rm=T));

for ( i in 1:nrow(ticket_stats)) {
  plist <- which(full$Ticket==ticket_stats$Ticket[i]);
  if(ticket_stats$na[i] > 0 & ticket_stats$l[i] > 1 & ticket_stats$c[i] > 0) {
    full$SLogL[plist] <- full$SLogL[plist] + 3;
  }
}

sconst <- -2.1;
full$SLogL[full$GID=="Single"] <- full$SLogL[full$GID=="Single"] - sconst;

full$SLogL[full$TFreq ==  7] <- full$SLogL[full$TFreq == 7]  - 3;
full$SLogL[full$TFreq ==  8] <- full$SLogL[full$TFreq == 8]  - 1;
full$SLogL[full$TFreq == 11] <- full$SLogL[full$TFreq == 11] - 3;

full$SLogL[full$Minor==1] <- 8;


train <- full[1:891,]
test <- full[892:1309,]

set.seed(2017);
trControl <- trainControl(method="repeatedcv", number=7, repeats = 5); 
fms <- formula("Survived ~ SLogL"); 
model_m <- train(fms, 
                 data = train,
                 metric="Accuracy", 
                 trControl = trControl, 
                 method = "knn"); 
full$Pred <- predict(model_m, full);
print(model_m$results)


prediction_dac <- predict(model_m, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution_dac <- data.frame(PassengerID = test$PassengerId, Survived = prediction_dac)

write.csv(solution_dac, file = 'dac_mod_Solution.csv', row.names = F)


ggplot(train, aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + 
  facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")


