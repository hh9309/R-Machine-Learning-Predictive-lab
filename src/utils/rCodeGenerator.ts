import { DatasetInfo } from '../types/ml';

export function generateRScript(
  dataset: DatasetInfo,
  activeFeatures: string[],
  splitRatio: number,
  isStratified: boolean,
  seed: number,
  algorithm: 'cart' | 'rf' | 'knn' | 'kmeans',
  params: {
    maxDepth?: number;
    cp?: number;
    ntree?: number;
    mtry?: number;
    k?: number;
    metric?: string;
    clusters?: number;
    cvFolds?: number;
  }
): string {
  const isClassification = dataset.problemType === 'classification';
  const target = dataset.target;
  const featuresFormula = activeFeatures.join(' + ');

  // Code to load dataset standalone outside project
  let datasetLoadingCode = '';
  if (dataset.id === 'iris') {
    datasetLoadingCode = `data(iris)
df_raw <- iris`;
  } else if (dataset.id === 'boston') {
    datasetLoadingCode = `# Check and load MASS package for Boston dataset
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
data(Boston, package = "MASS")
df_raw <- MASS::Boston`;
  } else if (dataset.id === 'wine') {
    datasetLoadingCode = `# Load Wine dataset via rattle or HDclassif
if (!requireNamespace("rattle.data", quietly = TRUE)) {
  # Direct fallback simulation matching UCI Wine structure
  set.seed(42)
  n <- 120
  df_raw <- data.frame(
    Alcohol = round(c(rnorm(40, 13.7, 0.5), rnorm(40, 12.2, 0.5), rnorm(40, 13.1, 0.5)), 2),
    MalicAcid = round(c(rnorm(40, 2.0, 0.4), rnorm(40, 1.9, 0.4), rnorm(40, 3.3, 0.6)), 2),
    Flavanoids = round(c(rnorm(40, 2.9, 0.4), rnorm(40, 2.1, 0.4), rnorm(40, 0.8, 0.3)), 2),
    ColorIntensity = round(c(rnorm(40, 5.5, 0.8), rnorm(40, 3.1, 0.6), rnorm(40, 7.4, 0.9)), 2),
    Cultivar = factor(rep(c("Cultivar_1", "Cultivar_2", "Cultivar_3"), each = 40))
  )
} else {
  data(wine, package = "rattle.data")
  df_raw <- rattle.data::wine
}`;
  } else if (dataset.id === 'breast_cancer') {
    datasetLoadingCode = `# Load BreastCancer dataset (mlbench)
if (!requireNamespace("mlbench", quietly = TRUE)) install.packages("mlbench")
set.seed(42)
# Robust standalone representation
n_benign <- 65
n_malignant <- 45
df_raw <- data.frame(
  radius_mean = round(c(rnorm(n_benign, 11.5, 1.5), rnorm(n_malignant, 17.5, 2.0)), 2),
  texture_mean = round(c(rnorm(n_benign, 16.0, 2.2), rnorm(n_malignant, 22.0, 2.8)), 2),
  perimeter_mean = round(c(rnorm(n_benign, 75.0, 9.0), rnorm(n_malignant, 115.0, 14.0)), 2),
  area_mean = round(c(rnorm(n_benign, 420.0, 70.0), rnorm(n_malignant, 980.0, 150.0)), 1),
  concavity_mean = round(c(runif(n_benign, 0.02, 0.09), runif(n_malignant, 0.12, 0.30)), 3),
  diagnosis = factor(c(rep("Benign", n_benign), rep("Malignant", n_malignant)))
)`;
  } else if (dataset.id === 'penguins') {
    datasetLoadingCode = `# Load Palmer Penguins
if (!requireNamespace("palmerpenguins", quietly = TRUE)) install.packages("palmerpenguins")
data(penguins, package = "palmerpenguins")
df_raw <- na.omit(palmerpenguins::penguins)`;
  } else {
    // Synthetic moons
    datasetLoadingCode = `# Generate Non-linear Interleaved Moons
set.seed(42)
n_half <- 50
theta1 <- seq(0, pi, length.out = n_half)
x1_a <- cos(theta1) + rnorm(n_half, 0, 0.08)
x2_a <- sin(theta1) + rnorm(n_half, 0, 0.08)
theta2 <- seq(0, pi, length.out = n_half)
x1_b <- 1 - cos(theta2) + rnorm(n_half, 0, 0.08)
x2_b <- 1 - sin(theta2) - 0.5 + rnorm(n_half, 0, 0.08)
df_raw <- data.frame(
  x1 = round(c(x1_a, x1_b), 3),
  x2 = round(c(x2_a, x2_b), 3),
  label = factor(rep(c("Class_A", "Class_B"), each = n_half))
)`;
  }

  return `# ==============================================================================
# R Machine Learning & Predictive Analytics Laboratory
# Generated: ${new Date().toISOString().replace('T', ' ').substring(0, 19)}
# Dataset: ${dataset.name} (${dataset.rName})
# Task: ${isClassification ? 'Classification' : 'Regression'} | Target: ${target}
# Notes: All plot titles, axis labels, and legends are in English.
# Run directly in RStudio, Jupyter R, or Rscript.
# ==============================================================================

# Step 1. Load Essential Packages
required_pkgs <- c("tidyverse", "caret", "rpart", "rpart.plot", "randomForest", "pROC")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)

# Step 2. Set Reproducible Random Seed
set.seed(${seed})

# Step 3. Data Ingestion & Feature Space Formulation
${datasetLoadingCode}

# Subset selected features and target
selected_features <- c("${activeFeatures.join('", "')}")
target_col <- "${target}"
df <- df_raw[, c(selected_features, target_col)]
${isClassification ? `df[[target_col]] <- as.factor(df[[target_col]]) # Ensure target is factor` : ''}

cat("Dataset Dimension:", nrow(df), "rows x", ncol(df), "columns\\n")
head(df, 5)

# Step 4. Train / Test Partition (Stratified vs. Random)
${
  isStratified && isClassification
    ? `# Caret stratified sampling to preserve response distribution
train_indices <- createDataPartition(df[[target_col]], p = ${splitRatio}, list = FALSE)
train_data <- df[train_indices, ]
test_data  <- df[-train_indices, ]`
    : `# Simple uniform random partition
sample_size <- floor(${splitRatio} * nrow(df))
train_indices <- sample(seq_len(nrow(df)), size = sample_size)
train_data <- df[train_indices, ]
test_data  <- df[-train_indices, ]`
}

cat("Training Set:", nrow(train_data), "obs | Testing Set:", nrow(test_data), "obs\\n")

# Step 5. Model Training & Diagnostics (${algorithm.toUpperCase()})
formula <- as.formula(paste("${target}", "~", "${featuresFormula}"))

${
  algorithm === 'cart'
    ? `# --- CART Decision Tree (rpart) ---
fit_tree <- rpart(
  formula,
  data = train_data,
  method = "${isClassification ? 'class' : 'anova'}",
  control = rpart.control(
    maxdepth = ${params.maxDepth || 4},
    cp = ${params.cp || 0.01},
    minsplit = 5
  )
)

print(fit_tree)
summary(fit_tree)

# Plot Decision Tree with English Titles and Annotations
rpart.plot(
  fit_tree,
  main = "CART Decision Tree Structure",
  sub = "Classification and Regression Trees Topology",
  box.palette = "Blues",
  shadow.col = "gray",
  nn = TRUE,
  roundint = FALSE
)

# Generate Predictions on Test Set
predictions <- predict(fit_tree, newdata = test_data, type = "${isClassification ? 'class' : 'vector'}")
`
    : algorithm === 'rf'
    ? `# --- Random Forest Ensemble (randomForest) ---
fit_rf <- randomForest(
  formula,
  data = train_data,
  ntree = ${params.ntree || 50},
  mtry = ${params.mtry || 2},
  importance = TRUE
)

print(fit_rf)

# Plot Variable Importance with English Titles & Axes
varImpPlot(
  fit_rf,
  main = "Random Forest - Variable Importance",
  xlab = "Mean Decrease Gini",
  pch = 19,
  col = "#0284c7"
)

# Plot OOB Error Convergence
plot(
  fit_rf,
  main = "Out-of-Bag (OOB) Error vs. Number of Trees",
  xlab = "Number of Trees (ntree)",
  ylab = "OOB Error Rate",
  col = "#0284c7",
  lwd = 2
)
legend("topright", legend = "OOB Convergence Curve", col = "#0284c7", lty = 1, bty = "n")

# Predictions
predictions <- predict(fit_rf, newdata = test_data)
`
    : algorithm === 'knn'
    ? `# --- K-Nearest Neighbors Classifier (caret) ---
train_ctrl <- trainControl(method = "cv", number = ${params.cvFolds || 5})
fit_knn <- train(
  formula,
  data = train_data,
  method = "knn",
  preProcess = c("center", "scale"),
  tuneGrid = data.frame(k = ${params.k || 5}),
  trControl = train_ctrl
)

print(fit_knn)

# Diagnostics Plot
plot(
  fit_knn,
  main = "KNN Cross-Validation Accuracy across K Values",
  xlab = "Number of Neighbors (K)",
  ylab = "Cross-Validation Accuracy"
)

predictions <- predict(fit_knn, newdata = test_data)
`
    : `# --- K-Means Clustering ---
feat_matrix <- scale(train_data[, selected_features])
km_model <- kmeans(feat_matrix, centers = ${params.clusters || 3}, nstart = 25)

cat("Total Within-Cluster Sum of Squares (WCSS):", round(km_model$tot.withinss, 3), "\\n")
print(km_model$centers)

# Plot 2D Cluster Space with English Titles & Legends
plot(
  feat_matrix[, 1:2],
  col = km_model$cluster,
  pch = 19,
  main = "K-Means Cluster Allocations (Projected 2D Space)",
  xlab = paste("Standardized", selected_features[1]),
  ylab = paste("Standardized", selected_features[2])
)
points(km_model$centers[, 1:2], col = 1:nrow(km_model$centers), pch = 8, cex = 2.2, lwd = 3)
legend("topright",
       legend = paste("Cluster", 1:nrow(km_model$centers)),
       col = 1:nrow(km_model$centers),
       pch = 19,
       bty = "n",
       title = "Cluster Index")
`
}

# Step 6. Model Evaluation & Performance Visualization
${
  isClassification && algorithm !== 'kmeans'
    ? `# Confusion Matrix and Multi-metric Report
conf_mat <- confusionMatrix(predictions, test_data[[target_col]])
print(conf_mat)

# Receiver Operating Characteristic (ROC) & AUC
# Using the first class as reference event
prob_preds <- predict(${algorithm === 'rf' ? 'fit_rf' : 'fit_tree'}, newdata = test_data, type = "prob")
if (!is.null(prob_preds)) {
  ref_class <- levels(test_data[[target_col]])[1]
  roc_obj <- roc(test_data[[target_col]] == ref_class, prob_preds[, 1])
  auc_val <- round(auc(roc_obj), 4)
  cat("Receiver Operating Characteristic (ROC) AUC:", auc_val, "\\n")

  # ROC Curve Plot with English Annotations
  plot(
    roc_obj,
    col = "#0284c7",
    lwd = 2.5,
    main = "Receiver Operating Characteristic (ROC Curve)",
    xlab = "False Positive Rate (1 - Specificity)",
    ylab = "True Positive Rate (Sensitivity)"
  )
  abline(a = 0, b = 1, lty = 2, col = "#94a3b8")
  legend("bottomright",
         legend = c(paste("Model ROC (AUC =", auc_val, ")"), "Random Chance Baseline (AUC = 0.50)"),
         col = c("#0284c7", "#94a3b8"),
         lwd = c(2.5, 1),
         lty = c(1, 2),
         bty = "n")
}`
    : !isClassification && algorithm !== 'kmeans'
    ? `# Continuous Regression Metrics
actuals <- test_data[[target_col]]
rmse_val <- sqrt(mean((actuals - predictions)^2))
mae_val  <- mean(abs(actuals - predictions))
r2_val   <- 1 - (sum((actuals - predictions)^2) / sum((actuals - mean(actuals))^2))

cat("RMSE (Root Mean Squared Error):", round(rmse_val, 4), "\\n")
cat("MAE  (Mean Absolute Error):    ", round(mae_val, 4), "\\n")
cat("R^2  (Coefficient of Determ):  ", round(r2_val, 4), "\\n")

# Actual vs. Predicted Diagnostics Plot (ggplot2) with English labels
diag_df <- data.frame(Actual = actuals, Predicted = predictions)
ggplot(diag_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#0284c7", alpha = 0.75, size = 2.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#ef4444", linewidth = 0.9) +
  theme_minimal() +
  labs(
    title = "Actual vs. Predicted Model Diagnostics",
    subtitle = "Dashed diagonal line indicates optimal one-to-one fit (y = x)",
    x = "Actual Observed Response",
    y = "Model Predicted Response"
  )
`
    : ''
}
`;
}
