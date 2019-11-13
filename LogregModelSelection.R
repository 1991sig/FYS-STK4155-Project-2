library(readr)

#####################################################
# Read in training data
#####################################################
CCDataCleanTrain <- read_csv("CCDataCleanTrain.csv")
colnames(CCDataCleanTrain)

# Removing the ID-column
CCDataCleanTrain <- CCDataCleanTrain[,-1]

#####################################################
# Fit GLM-model
#####################################################

# Full model containing all columns
mod.full <- glm(default.payment.next.month ~ ., family = binomial(link = "logit"), data = CCDataCleanTrain)
summary(mod.full)

# Predicted probabilities on training set
mu_hat <- predict.glm(mod.full, CCDataCleanTrain, type = "response")

# Predicted labels
y_hat <- ifelse(mu_hat > 0.5, 1, 0)

# Accuracy on training data
mean(y_hat == CCDataCleanTrain$default.payment.next.month)

#####################################################
# Model Selection
#####################################################

## Backward elimination
mod.optim1 <- step(mod.full, direction="backward")
summary(mod.optim1)

# Prediction and accuracy evaluation on training set
mu_hat1 <- predict.glm(mod.optim1, CCDataCleanTrain, type = "response")
y_hat1 <- ifelse(mu_hat1 > 0.5, 1, 0)
mean(y_hat1 == CCDataCleanTrain$default.payment.next.month)

## Forward Selection

# Null model containing only an intercept term
mod.null <- glm(default.payment.next.month ~ 1, 
                family = binomial(link = "logit"), 
                data = CCDataCleanTrain)

# The forward selection model starting from the
# null model.
mod.optim2 <- step(mod.null, 
                   scope = list(lower = formula(mod.null),
                                upper = formula(mod.full)), 
                   direction="forward")

summary(mod.optim2)

# Prediction with the resulting model, and evalutation
# of accuracy on training set
mu_hat2 <- predict.glm(mod.optim2, CCDataCleanTrain, type = "response")
y_hat2 <- ifelse(mu_hat2 > 0.5, 1, 0)
mean(y_hat2 == CCDataCleanTrain$default.payment.next.month)


formula(mod.optim1)
formula(mod.optim2)

# The backward elimination model includes:
# BILL_AMT1, BILL_AMT2, BILL_AMT5, and PAY_AMT5
# These variables are not in the forward selection model

mean(y_hat == CCDataCleanTrain$default.payment.next.month)
mean(y_hat1 == CCDataCleanTrain$default.payment.next.month)
mean(y_hat2 == CCDataCleanTrain$default.payment.next.month)

# Model mod.optim2 performs best on the training set

#####################################################
# Test Set Evaluation
#####################################################
CCDataCleanTest <- read_csv("CCDataCleanTest.csv")
CCDataCleanTest <- CCDataCleanTest[,-1]

# Predicting with the optimal model, mod.optim2
mu_hat.test <- predict.glm(mod.optim2, newdata = CCDataCleanTest, type = "response")
y_hat.test <- ifelse(mu_hat.test > 0.5, 1, 0)
mean(y_hat.test == CCDataCleanTest$default.payment.next.month)

# Checking the other models as well
mu_hat.test1 <- predict.glm(mod.optim1, newdata = CCDataCleanTest, type = "response")
y_hat.test1 <- ifelse(mu_hat.test1 > 0.5, 1, 0)
mean(y_hat.test1 == CCDataCleanTest$default.payment.next.month)

mu_hat.test2 <- predict.glm(mod.full, newdata = CCDataCleanTest, type = "response")
y_hat.test2 <- ifelse(mu_hat.test2 > 0.5, 1, 0)
mean(y_hat.test2 == CCDataCleanTest$default.payment.next.month)

# Conclusion: mod.optim2 is the best on the test set also

















