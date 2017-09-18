
library(readr)
library(dplyr)

# Recursive Partitioning and Regression Trees, part of Base R
library(rpart) 

# Need to be installed
library(rattle) # Do sudo apt-get install libgtk2.0-dev before installation
library(rpart.plot)
library(RColorBrewer)


# Import data ------------------------------------------------

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")


# Decision tree with default parameters -----------------------------------

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train,
             method = "class")

# Base R visualisation
plot(fit)
text(fit)

# Fancy visualisation
fancyRpartPlot(fit)

# Prepare tibble for submission to Kaggle
pred <- predict(fit, test, type = "class")
submission <- select(test, PassengerId) %>% mutate(Survived = pred)
write_csv(submission, path = "results/dec_tree_defaults.csv")


# Decision tree with overfitting ------------------------------------------

# cp stops splits that aren't deemed important enough
# minsplit governs how many samples must sit in a bucket before even looking 
# for a split
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train,
             method = "class", 
             control = rpart.control(minsplit = 2, cp = 0))

# Prepare tibble for submission to Kaggle
pred <- predict(fit, test, type = "class")
submission <- select(test, PassengerId) %>% mutate(Survived = pred)
write_csv(submission, path = "results/dec_tree_overfit.csv")


# Decision tree interactive -----------------------------------------------

# The following commands run an interactive version of the decision tree fit to
# the data for manual pruning

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train,
             method = "class")
new.fit <- prp(fit,snip=TRUE)$obj
fancyRpartPlot(new.fit)

