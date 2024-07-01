# Load necessary libraries
library(tidyverse)
install.packages("caret")
library(caret)
library(e1071)
library(rpart)
install.packages('rpart.plot')
library(rpart.plot)

# Set working directory and load the data
setwd("/Users/kirthanshaker/Desktop/SCMA 631 Data Files ")
df <- read.csv("/Users/kirthanshaker/Desktop/SCMA 631 Data Files /Credit Card Defaulter Prediction.csv")

# Data Cleaning and EDA
# Remove spaces in column names
names(df) <- gsub(" ", "", names(df))

# Check for imbalance in the dataset
table(df$default)

# Drop the 'ID' column
df <- df %>% select(-ID)

# Encode categorical variables
cat_features <- df %>% select_if(is.character)

# Data Encoding for categorical variables
df <- df %>% 
  mutate(across(where(is.character), as.factor)) %>% 
  mutate(across(where(is.factor), as.integer))

# Final data
f_data <- df %>% select(-c(SEX, EDUCATION, MARRIAGE))

# Test-Train Split
set.seed(42)
trainIndex <- createDataPartition(f_data$default, p = .8, 
                                  list = FALSE, 
                                  times = 1)
x_train <- f_data[trainIndex, -ncol(f_data)]
x_test <- f_data[-trainIndex, -ncol(f_data)]
y_train <- f_data[trainIndex, ncol(f_data)]
y_test <- f_data[-trainIndex, ncol(f_data)]

# Scaling the data
scaler <- preProcess(x_train, method = c("range"))
x_train <- predict(scaler, x_train)
x_test <- predict(scaler, x_test)

# Feature selection using mutual information
mutual_info_scores <- varImp(FeatureSelection(y = y_train, x = x_train, method = "mutual_info"))
feature_scores_df <- mutual_info_scores %>% 
  arrange(desc(Overall))

# Select top 15 features
selected_features <- head(feature_scores_df$var, 15)
feature_selection_train <- x_train[, selected_features]
feature_selection_test <- x_test[, selected_features]

# Logistic Regression
logreg <- glm(y_train ~ ., data = feature_selection_train, family = binomial)
y_pred <- predict(logreg, feature_selection_test, type = "response")
y_pred_class <- ifelse(y_pred > 0.5, 1, 0)

# Classification report for Logistic Regression
logreg_cm <- confusionMatrix(as.factor(y_pred_class), as.factor(y_test))
print(logreg_cm)

# ROC Curve and AUC for Logistic Regression
pred <- prediction(y_pred, y_test)
perf <- performance(pred, "tpr", "fpr")
roc_auc <- performance(pred, "auc")@y.values[[1]]

# Plot ROC Curve
plot(perf, col = "darkorange", lwd = 2, main = "ROC Curve for Logistic Regression")
abline(a = 0, b = 1, col = "navy", lwd = 2, lty = 2)
text(0.5, 0.5, paste("AUC =", round(roc_auc, 2)), pos = 4, col = "darkorange")

# Confusion Matrix for Logistic Regression
conf_matrix <- table(Predicted = y_pred_class, Actual = y_test)
print(conf_matrix)

# Decision Tree Classifier
dt_classifier <- rpart(y_train ~ ., data = feature_selection_train, method = "class")
y_pred_dt <- predict(dt_classifier, feature_selection_test, type = "class")

# Classification report for Decision Tree
dtree_cm <- confusionMatrix(as.factor(y_pred_dt), as.factor(y_test))
print(dtree_cm)

# ROC Curve and AUC for Decision Tree
y_pred_proba <- predict(dt_classifier, feature_selection_test, type = "prob")[, 2]
pred_dt <- prediction(y_pred_proba, y_test)
perf_dt <- performance(pred_dt, "tpr", "fpr")
roc_auc_dt <- performance(pred_dt, "auc")@y.values[[1]]

# Plot ROC Curve
plot(perf_dt, col = "darkorange", lwd = 2, main = "ROC Curve for Decision Tree")
abline(a = 0, b = 1, col = "navy", lwd = 2, lty = 2)
text(0.5, 0.5, paste("AUC =", round(roc_auc_dt, 2)), pos = 4, col = "darkorange")

# Confusion Matrix for Decision Tree
conf_matrix2 <- table(Predicted = y_pred_dt, Actual = y_test)
print(conf_matrix2)

# Parse Classification Report function
parse_classification_report <- function(cm) {
  data.frame(
    class = rownames(cm$byClass),
    precision = cm$byClass[, "Precision"],
    recall = cm$byClass[, "Recall"],
    f1_score = cm$byClass[, "F1"],
    support = cm$table[, "Support"]
  )
}

df1 <- parse_classification_report(dtree_cm)
df2 <- parse_classification_report(logreg_cm)

# Add model names and overall accuracy
df1$model <- "Decision Tree"
df2$model <- "Logistic Regression"

# Concatenate the two dataframes
comparison_df <- rbind(df1, df2)

# Reorder columns
comparison_df <- comparison_df %>% select(model, class, precision, recall, f1_score, support)

# Display the comparison table
print(comparison_df)
