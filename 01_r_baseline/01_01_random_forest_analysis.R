# ==============================================================================
# MEDICAL ABSTRACT CLASSIFICATION: ENSEMBLE LEARNING (RANDOM FOREST)
# ==============================================================================
# Objective: Establish a quick performance floor to ensure a valid submission.
# Strategy: This script focuses on speed and simplicity, allowing more time 
#           to be allocated to Transformer-based models (BioBERT).
# Accuracy achieved: 0.434
# ==============================================================================

# 1. Load Libraries
library(tidyverse)
library(tidytext)
library(caret)
library(SnowballC)
library(tm)
library(randomForest)

# 2. Data Loading
# Note: Dataset files should be in the working directory.
train_data <- read.csv("medical_abstracts_train.csv") 
train_data$condition_label <- as.factor(train_data$condition_label)

# 3. Text Pre-processing
# Tokenization, Stemming, and Stopword removal
data_tokens <- train_data %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, medical_abstract) %>%
  anti_join(stop_words) %>%
  mutate(word = wordStem(word, language = "en"))

# 4. Feature Engineering (TF-IDF)
dtm_tfidf <- data_tokens %>%
  count(line, word) %>%
  bind_tf_idf(word, line, n) %>%
  cast_dtm(line, word, tf_idf)

# 5. Dimensionality Reduction
# Keep terms appearing in at least 2% of documents to ensure fast training.
dtm_small <- removeSparseTerms(dtm_tfidf, sparse = 0.98)
data_sparse <- as.data.frame(as.matrix(dtm_small))

# 6. Label Synchronization
present_lines <- as.numeric(rownames(data_sparse))
data_sparse$condition_label <- train_data$condition_label[present_lines]

# 7. Model Training (Random Forest)
set.seed(123)
trainIndex <- createDataPartition(data_sparse$condition_label, p = .8, list = FALSE)
train_set <- data_sparse[trainIndex, ]

x_train <- train_set %>% select(-condition_label)
y_train <- train_set$condition_label

# Using 50 trees for a rapid baseline execution
model_rf <- randomForest(x = x_train, y = y_train, ntree = 50)

# 8. Validation Data Loading
# Loading the 2,000 unlabeled samples for the challenge
validation_data <- read.csv("medical_abstracts_validation.csv")

# 9. Validation Set Pre-processing
# Process text to match the feature space (columns) of the training set
validation_tokens <- validation_data %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, medical_abstract) %>%
  filter(word %in% colnames(x_train)) %>%
  anti_join(stop_words) %>%
  mutate(word = wordStem(word, language = "en"))

x_validation <- as.data.frame(as.matrix(cast_dtm(validation_tokens, line, word, n)))

# 10. Column Alignment
# Ensure the validation dataframe has the exact same columns and order as x_train
missing_cols <- setdiff(colnames(x_train), colnames(x_validation))
if(length(missing_cols) > 0) x_validation[missing_cols] <- 0
x_validation <- x_validation[, colnames(x_train)]

# 11. Final Inference & Export
# Format requirement: 2,000 rows, one predicted class per line, no headers.
final_predictions <- predict(model_rf, x_validation)

write.table(final_predictions, 
            file = "challenge_predictions_rf.txt", 
            row.names = FALSE, 
            col.names = FALSE, 
            quote = FALSE)

print("Baseline process completed. Output: 'challenge_predictions_rf.txt' (2,000 rows).")
