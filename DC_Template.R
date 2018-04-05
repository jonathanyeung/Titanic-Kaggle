

## Load libraries


library(plyr);      # load plyr prior to dplyr to avoid warnings
library(caret);     # A nice machine learning wrapper library 
library(dplyr);     # data manipulation
library(gridExtra); # multiple plots in one page
#library(rpart.plot);# nice plots for rpart trees 


## Load data


inpath  <- "../data/";
outpath <- "../data/";
kaggle <- F;
if(!dir.exists(inpath)) {
        kaggle  <- T;
	inpath  <- "../input/"; # changing path to Kaggle's environment
	outpath <- "";
}
train <- read.csv('./train.csv', stringsAsFactors = F)
test  <- read.csv('./test.csv', stringsAsFactors = F)

## Combine data


test$Survived <- NA;
test$Set  <- "Test";
train$Set <- "Train";
comb <- rbind(train, test);
test_index <- comb %>% dplyr::filter(Set=="Test") %>% .$PassengerId # test PassengerId 


## Let us fix a couple of errors in the data set

comb$SibSp[comb$PassengerId==280] = 0
comb$Parch[comb$PassengerId==280] = 2
comb$SibSp[comb$PassengerId==1284] = 1
comb$Parch[comb$PassengerId==1284] = 1


## Transform Survived to a categorical target for binary classification

comb$Survived <- factor(comb$Survived);

# Completing missing data

## Completing Fare 

trControl <- trainControl(method="repeatedcv", number=7, repeats=5); 
faremiss <- which(is.na(comb$Fare)); # missing fares (only PassengerId = 1044 is missing)
model_f   <- train( Fare ~ Pclass + Sex + Embarked + SibSp + Parch,
                  data = comb %>% dplyr::filter(!is.na(Fare)),
                   trControl = trControl, method="rpart", na.action = na.pass, tuneLength = 5);
comb$Fare[faremiss] = predict(model_f, comb[faremiss,]); 
comb$FareFac <- factor(comb$Fare); 

# rpart.plot(model_f$finalModel)

print(model_f$results)


## Completing Embarked

ggplot(comb, aes(Pclass,fill=!is.na(Age))) + geom_bar(position="dodge") + labs(title="Passenger Has Age",fill="Has Age")


### How Age impacts Pclass 1 and 2

ggplot(comb %>% dplyr::filter(Set=="Train", Pclass!=3), aes(Age)) + 
  geom_density(alpha=0.5, aes(fill=factor(Survived))) + labs(title="Survival density per Age for Pclass 1 and 2");

child <- 14;
comb$Minor <- ifelse(comb$Age<child&comb$Pclass!=3, 1, 0);
comb$Minor <- ifelse(is.na(comb$Minor), 0, comb$Minor);


## Compute frequencies


comb$TFreq <- ave(seq(nrow(comb)), comb$Ticket,  FUN=length);
comb$FFreq <- ave(seq(nrow(comb)), comb$FareFac, FUN=length);
comb$CFreq <- ave(seq(nrow(comb)), comb$Cabin,   FUN=length);


## Family

comb$Surname <- sapply(as.character(comb$Name), FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]});

## Fares


print(comb %>% dplyr::filter(Fare=="6.75")) # A young couple?



comb %>% dplyr::filter(Sex=="female",Pclass==3,SibSp==0,Parch==0,Age>15, Age<20)

## Finding groups


comb$GID <- rep(NA, nrow(comb));
maxgroup <- 12; # maximum size of a group
for ( i in comb$PassengerId) {
	if(comb$SibSp[i] + comb$Parch[i] > 0) { # Check first if ID has relatives 
		comb$GID[i] <- paste0(comb$Surname[i], comb$SibSp[i] + comb$Parch[i]); # SibSp and Parch are used to potentially differentiate different families with the same Surname.	
	} else {
		if(comb$TFreq[i] > 1 & is.na(comb$GID[i])) { # Next if shares ticket number with others 
			comb$GID[i] <- as.character(comb$Ticket[i]);
		} else {
			if(comb$CFreq[i] > 1 & comb$CFreq[i]<maxgroup & is.na(comb$GID[i])) { # Next if shares cabin with others
					comb$GID[i] <- as.character(comb$Cabin[i]);
			}
			else {
				if(comb$FFreq[i] > 1 & comb$FFreq[i]<maxgroup & is.na(comb$GID[i])) { # Next if shares Fare value with others
					comb$GID[i] <- as.character(comb$FareFac[i]);
				} else { 
					comb$GID[i] <- "Single"; # Individual doesn't belong to any group
				}
			}
		}	
	}
}
comb$GID <- factor(comb$GID);


# Engineer SLogL feature

comb$SLogL <- rep(0,nrow(comb));
# define function to compute log likelihood of a/(1-a)
logl <- function(a) {
	a <- max(a,0.1); # avoids log(0)
	a <- min(a,0.9); # avoids division by 0
	return (log(a/(1-a)));
}

p <- list();
item <- 1;
ylim <- 300;
for(class in c(1:3)){
	for(sex in c("male", "female")) {
		p[[item]] <- ggplot(comb %>% dplyr::filter(Set=="Train", Pclass==class, Sex==sex), aes(x=Survived)) + 
		  geom_bar(aes(fill=Survived)) + scale_y_continuous(limits=c(0,ylim)) + 
		  theme(legend.position="none") + labs(title=paste('Pclass=', as.character(class), sex));
		item <- item + 1;
	}
}
do.call(grid.arrange, p)

for (sex in c("male", "female")) {
	for (class in c(1:3)) {
		comb$SLogL[comb$Sex==sex& comb$Pclass==class] <- 
			logl(nrow(comb %>% dplyr::filter(Survived==1, Sex==sex, Pclass==class))/nrow(comb %>% dplyr::filter(Set=="Train", Sex==sex, Pclass==class)));
	}
}

ggplot(comb %>% dplyr::filter(Set=="Train"), aes(x=FFreq, y=TFreq, color=factor(Survived))) +
  geom_density_2d() + labs(title="Ticket Frequency and Fare Frequency Density");

pf <- ggplot(comb %>% dplyr::filter(Set=="Train",Sex=='female'), aes(x=FFreq, y=TFreq, color=Survived)) +
  geom_density_2d() + labs(title="TFreq and FFreq Density (female)");
pm <- ggplot(comb %>% dplyr::filter(Set=="Train",Sex=="male"), aes(x=FFreq, y=TFreq, color=Survived)) +
  geom_density_2d() + labs(title="TFreq and FFreq Density (male)");
grid.arrange(pf, pm, ncol=2);

ggplot(comb %>% dplyr::filter(Set=="Train"), aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + 
  facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")


# Adjusting SLogL according to groups

ggplot(comb %>% dplyr::filter(Set=="Train"), aes(x=FFreq, y=TFreq, color=Survived, alpha=0.5)) +
  geom_count(position=position_dodge(width=5)) + labs(title="Ticket and Fare Frequencies");



## Next we reward or penalize groups of people

ticket_stats <- comb %>% group_by(Ticket) %>% summarize(l = length(Survived), na = sum(is.na(Survived)), c = sum(as.numeric(Survived)-1, na.rm=T));


### By increasing the log likelihood score for groups larger than one that have survivors

for ( i in 1:nrow(ticket_stats)) {
	plist <- which(comb$Ticket==ticket_stats$Ticket[i]);
	if(ticket_stats$na[i] > 0 & ticket_stats$l[i] > 1 & ticket_stats$c[i] > 0) {
			comb$SLogL[plist] <- comb$SLogL[plist] + 3;
	}
}


### The sconst variable was used to penalize singles prior to version 30

sconst <- -2.1;
comb$SLogL[comb$GID=="Single"] <- comb$SLogL[comb$GID=="Single"] - sconst;


### By penalizing large group sizes (See TFreq vs Pclass vs SLogL graph)

comb$SLogL[comb$TFreq ==  7] <- comb$SLogL[comb$TFreq == 7]  - 3;
comb$SLogL[comb$TFreq ==  8] <- comb$SLogL[comb$TFreq == 8]  - 1;
comb$SLogL[comb$TFreq == 11] <- comb$SLogL[comb$TFreq == 11] - 3;


### By promoting high likelihood of survival for Minors in Pclass 1 and Pclass 2.


comb$SLogL[comb$Minor==1] <- 8;


## Plot SLogL after adjustments


ggplot(comb %>% dplyr::filter(Set=="Train"), aes(x=Pclass, y=SLogL)) + geom_jitter(aes(color=Survived)) + 
  facet_grid(  . ~ TFreq,  labeller=label_both) + labs(title="SLogL vs Pclass vs TFreq")

set.seed(2017);
trControl <- trainControl(method="repeatedcv", number=7, repeats = 5); 
fms <- formula("Survived ~ SLogL"); 
model_m <- train(fms, data = comb %>% dplyr::filter(Set=="Train"),
	 metric="Accuracy", trControl = trControl, method = "knn"); 
comb$Pred <- predict(model_m, comb);
print(model_m$results)

df_final <- data.frame(PassengerId = comb$PassengerId[test_index], Survived=comb$Pred[test_index]);
write.csv(df_final, paste0(outpath,"pred_divconq.csv"), row.names =F, quote=F)




 
