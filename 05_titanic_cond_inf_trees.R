
library(dplyr)

# Forest of conditional inference trees
library(party)

# Import data -------------------------------------------------------------

load("data/combined_train_test_imputed.RData")


# Forest of conditional inference trees -----------------------------------

train <- filter(combi, !is.na(Survived))
test <- filter(combi, is.na(Survived))
test <- select(test, -Survived)

set.seed(415)

# Since this library allows more than 32 levels, we go back to the original 
# FamilyID variable
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyID,
               data = train, 
               controls = cforest_unbiased(ntree = 2000, # Number of trees
                                           mtry = 3)) # Features to sample

# Prepare tibble for submission to Kaggle
pred <- predict(fit, test, OOB=TRUE, type = "response")
submission <- select(test, PassengerId) %>% mutate(Survived = pred)
write_csv(submission, path = "results/forest_cond_inf_trees.csv")
