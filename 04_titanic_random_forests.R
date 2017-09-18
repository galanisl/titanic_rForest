
library(dplyr)

# Recursive Partitioning and Regression Trees, part of Base R
library(rpart)

# Need to be installed
library(randomForest)

# Import data -------------------------------------------------------------

load("data/combined_train_test.RData")


# Imputation of missing values --------------------------------------------

# randomForest package does not accept NAs:

# Around 20% of the age values are missing
summary(combi$Age)

# We can use a Decision Tree to predict the age of the samples with missing ages
# Note that method is now 'anova' and not 'class', because we're trying to pre-
# dict continuous values
age_fit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + 
                   Embarked + Title + FamilySize, 
                 data = combi[!is.na(combi$Age), ], # We train with aged samples
                 method = "anova")
combi <- mutate(combi, Age = replace(Age, is.na(Age), 
                                     predict(age_fit, combi[is.na(combi$Age), ])))

# There are two passengers with no Port of Embarkation
summary(factor(combi$Embarked))

# Since most passengers embarked in Southampton, we impute the 
# missing values with S
combi <- mutate(combi, Embarked = replace(Embarked, is.na(Embarked), "S"))
combi$Embarked <- as.factor(combi$Embarked)

# There is one passenger with missing Fare
summary(combi$Fare)

# Let's impute the missing Fare with the median
combi <- mutate(combi, Fare = replace(Fare, is.na(Fare), median(Fare, na.rm = T)))

# randomForest package limits factors to 32 levels:

# The FamilyID column has almost 61 levels, we solve this problem with a new
# family ID strategy. This results in 22 levels
length(levels(combi$FamilyID))
combi <- mutate(combi, FamilyID2 = as.character(FamilyID)) %>% 
  mutate(FamilyID2 = as.factor(replace(FamilyID2, FamilySize <= 3, "Small")))
length(levels(combi$FamilyID2))


# Random forest construction ----------------------------------------------

# Let's get back to train and test sets with engineered features and 
# imputed values
combi$Sex <- as.factor(combi$Sex)

# Save combi for further experiments
save(combi, file = "data/combined_train_test_imputed.RData")

train <- filter(combi, !is.na(Survived))
test <- filter(combi, is.na(Survived))
test <- select(test, -Survived)

# Random forests have two sources of randomness: sample and feature sampling.
# So, we set a seed for reproducibility
set.seed(415)

# Make sure that all categorical variables are factors!!!
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                      Parch + Fare + Embarked + Title + 
                      FamilySize + FamilyID2,
                    data = train, 
                    importance = TRUE, # Compute feature importance?
                    ntree = 2000) # Number of trees in the forest

# Look at variable importance
varImpPlot(fit)

# Prepare tibble for submission to Kaggle
pred <- predict(fit, test, type = "class")
submission <- select(test, PassengerId) %>% mutate(Survived = pred)
write_csv(submission, path = "results/random_forest.csv")
