
library(readr)
library(dplyr)


# Import data ------------------------------------------------

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")


# Preliminary exploration -------------------------------------------------

str(train)
table(train$Survived)
# Note that more than 60% of the passengers died
# Base R
prop.table(table(train$Survived))

# dplyr
train %>% summarise(lived = mean(Survived == 0), died = mean(Survived == 1))

# Initial prediction ------------------------------------------------------

# Since most people died according to the training set, let's assume everyone
# died in the test set
test <- mutate(test, Survived = 0)

# Prepare tibble for submission to Kaggle
submission <- select(test, PassengerId, Survived)
write_csv(submission, path = "results/they_all_perish.csv")


# Gender-Class Model ------------------------------------------------------

# How many female and male passengers? ---
# Base R
summary(factor(train$Sex))

# dplyr
train %>% group_by(Sex) %>% count(Sex) #or
train %>% group_by(Sex) %>% summarise(count = n())

# Proportion of female and male passengers that survived from total? ---
# Base R
prop.table(table(train$Sex, train$Survived))

# dplyr
train %>% 
  group_by(Sex) %>% 
  summarise(lived = sum(Survived == 0)/nrow(train), 
            died = sum(Survived == 1)/nrow(train))

# Proportion of female and male passengers that survived within sex? ---
# Base R (you have to indicate the dimension for proportion computation)
prop.table(table(train$Sex, train$Survived), 1)

# dplyr
train %>% 
  group_by(Sex) %>% 
  summarise(lived = mean(Survived == 0), died = mean(Survived == 1))

# Since 74% of the females survived, we could predict that all females survived
# in the test set
test <- mutate(test, Survived = 0) %>% 
  mutate(Survived = replace(Survived, Sex == "female", 1))

# Prepare tibble for submission to Kaggle
submission <- select(test, PassengerId, Survived)
write_csv(submission, path = "results/only_males_perish.csv")

# Let's now see if looking at passenger age improves prediction accuracy
summary(train$Age)

train <- mutate(train, Child = 0) %>% 
  mutate(Child = replace(Child, Age < 18, 1))

# Number of female and male passengers that survived by age group
train %>% group_by(Sex, Child) %>% summarise(lived = sum(Survived))

# Proportion of female and male passengers that survived within age group
# Base R
aggregate(Survived ~ Child + Sex, data=train, 
          FUN = function(x) {sum(x)/length(x)})

# dplyr
train %>% group_by(Sex, Child) %>% summarise(lived = mean(Survived))

# Since not much is gained by looking at age groups, let's see if class and 
# ticket fare help
train <- mutate(train, Fare2 = "30+") %>% 
  mutate(Fare2 = replace(Fare2, Fare < 30 & Fare >= 20, "20-30")) %>%
  mutate(Fare2 = replace(Fare2, Fare < 20 & Fare >= 10, "10-20")) %>% 
  mutate(Fare2 = replace(Fare2, Fare < 10, "<10"))
  
# Proportion of passengers that survived based on Sex, Class and Fare group
# Base R
aggregate(Survived ~ Fare2 + Pclass + Sex, data = train, 
          FUN = function(x) {sum(x)/length(x)})
# dplyr
train %>% group_by(Sex, Pclass, Fare2) %>% 
  summarise(lived = mean(Survived))

# We notice that most of the class 3 women who paid more than $20 for their 
# ticket actually also miss out on a lifeboat. Let's make a new prediction based
# on these insights
test <- mutate(test, Survived = 0) %>% 
  mutate(Survived = replace(Survived, Sex == "female", 1)) %>% 
  mutate(Survived = replace(Survived, 
                            Sex == "female" & Pclass == 3 & Fare >= 20, 0))

# Prepare tibble for submission to Kaggle
submission <- select(test, PassengerId, Survived)
write_csv(submission, path = "results/expensive_class3_females_perish.csv")
