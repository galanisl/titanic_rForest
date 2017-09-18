
library(readr)
library(dplyr)
library(tidyr)

# Recursive Partitioning and Regression Trees, part of Base R
library(rpart) 

# Need to be installed
library(rattle) # Do sudo apt-get install libgtk2.0-dev before installation
library(rpart.plot)
library(RColorBrewer)


# Import data ------------------------------------------------

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")


# Feature engineering -----------------------------------------------------

# Combine the train and test sets to obtain all engineered features for 
# both cases
test <- mutate(test, Survived = NA)
combi <- rbind(train, test)

# Let's see if name titles help with prediction of survival
combi <- separate(combi, Name, into = c("LName", "Title", "FName"), 
                  sep = "[,.]")
combi <- mutate(combi, Title = trimws(Title),
                LName = trimws(LName),
                FName = trimws(FName))

# Group redundant titles into more general ones
combi <- mutate(combi, Title = replace(Title, 
                                       Title %in% c("Mme", "Mlle"), "Mlle"))
combi <- mutate(combi, Title = replace(Title, 
                                       Title %in% c("Capt", "Don", "Major", "Sir"), 
                                       "Sir"))
combi <- mutate(combi, Title = replace(Title, 
                                       Title %in% c("Dona", "Lady", "the Countess", 
                                                           "Jonkheer"), 
                                       "Lady"))

# Big families could have had troubles surviving, so let's sum number of 
# siblings/spouses, parents/children and add the passenger himself
combi <- mutate(combi, FamilySize = SibSp + Parch + 1)

# Maybe specific families had more troubles to survive than others, let's create
# a family ID by combining Surnames and family size
combi <- mutate(combi, FamilyID = paste0(FamilySize, LName))

# Given the hypothesis that large families might have had trouble sticking 
# together in the panic, let's knock out any family size of two or less and 
# call it a 'small' family
combi <- mutate(combi, FamilyID = replace(FamilyID, FamilySize <= 2, "Small"))

# If we explore the data with table(combi$FamilyID), we see cases like family
# '3Beckwith', which is supposed to have 3 members, but only has 2 in our data:
table(combi$FamilyID)

# To solve this problem, we label family IDs with frequencies below 3 as "small"
combi <- group_by(combi, FamilyID) %>% 
  mutate(Freq = n()) %>% ungroup() %>% 
  mutate(FamilyID = replace(FamilyID, Freq <= 2, "Small"))

# Transform Title and FamilyID into factors
combi$Title <- as.factor(combi$Title)
combi$FamilyID <- as.factor(combi$FamilyID)

# Let's get back to train and test sets with engineered features
train_fe <- filter(combi, !is.na(Survived))
test_fe <- filter(combi, is.na(Survived))

# Save combi for further experiments
save(combi, file = "data/combined_train_test.RData")

# Decision tree construction ----------------------------------------------

fit <- rpart(Survived ~ Pclass + Sex + Age + 
               SibSp + Parch + Fare + Embarked + 
               Title + FamilySize + FamilyID,
             data = train_fe, method = "class")

fancyRpartPlot(fit)

# Prepare tibble for submission to Kaggle
test_fe <- select(test_fe, -Survived)
pred <- predict(fit, test_fe, type = "class")
submission <- select(test_fe, PassengerId) %>% mutate(Survived = pred)
write_csv(submission, path = "results/dec_tree_feat_eng.csv")
