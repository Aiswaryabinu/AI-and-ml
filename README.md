# Machine Learning Interview Quick Guide

This guide covers key concepts and interview-ready answers for **Data Preprocessing**, **Exploratory Data Analysis (EDA)**, **Linear Regression**, **Logistic Regression**, and **K-Nearest Neighbors (KNN)**. Each section includes definitions, examples, and common interview questions.

---

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Linear Regression](#linear-regression)
4. [Logistic Regression](#logistic-regression)
5. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)

---

## 1. Data Preprocessing

### 1.1 Types of Missing Data
- **MCAR (Missing Completely at Random):**  
  *Definition:* Missingness is unrelated to any data.  
  *Example:* Sensor randomly fails to record data.
- **MAR (Missing at Random):**  
  *Definition:* Missingness is related to observed data, not the missing value.  
  *Example:* Income missing more often for younger respondents.
- **MNAR (Missing Not at Random):**  
  *Definition:* Missingness depends on the value of the missing data itself.  
  *Example:* People with high income less likely to report it.

### 1.2 Handling Categorical Variables
- **Label Encoding:** Assigns an integer to each category.  
  *Example:* [Red, Green, Blue] → [0, 1, 2]
- **One-Hot Encoding:** Creates binary columns for each category.  
  *Example:* [Red, Green, Blue] → [1,0,0], [0,1,0], [0,0,1]

### 1.3 Normalization vs Standardization
- **Normalization:** Scale data to [0, 1].  
  *Example:* [10, 20, 30] → [0, 0.5, 1]
- **Standardization:** Scale data to mean 0, std 1.  
  *Example:* [10, 20, 30] → [-1.22, 0, 1.22]

### 1.4 Outlier Detection
- **Statistical:** Z-score, IQR.  
  *Example:* 100 in [10, 12, 11, 13, 100] is an outlier.
- **Visualization:** Box plots, scatter plots.

### 1.5 Importance of Preprocessing
- Clean and consistent data improves model performance.
- *Example:* Imputing missing values increases accuracy.

### 1.6 One-Hot Encoding vs Label Encoding
- **One-Hot:** Binary columns for each category.
- **Label:** Integer for each category.

### 1.7 Handling Imbalanced Data
- **Resampling:** Over/under-sample classes.
- **SMOTE:** Synthetic samples.
- **Class weights:** Adjust model penalty.

### 1.8 Effect on Model Accuracy
- Good preprocessing improves accuracy.
- *Example:* Scaling improves KNN; not handling missing values hurts models.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Purpose of EDA
- Summarize, visualize, and understand data before modeling.
- *Example:* Plotting histograms for feature distribution.

### 2.2 Boxplots in EDA
- Visualize distribution, median, quartiles, and outliers.
- *Example:* Boxplot of salaries shows high earners as outliers.

### 2.3 Correlation
- Measures linear relationship between variables.
- *Example:* Height and weight are positively correlated.

### 2.4 Detecting Skewness
- Skewness shows asymmetry (e.g. histogram tail to the right = right-skewed).

### 2.5 Multicollinearity
- High correlation between predictors (e.g. age and years of experience).

### 2.6 EDA Tools
- Pandas, Matplotlib, Seaborn, Plotly, R, Tableau.

### 2.7 Example: EDA Revealing Problems
- EDA revealed duplicated sales records causing a data spike.

### 2.8 Role of Visualization
- Helps detect patterns, outliers, relationships, and communicate results.
- *Example:* Heatmaps for correlations, scatter plots for relationships.

---

## 3. Linear Regression

### 3.1 Assumptions
- **Linearity:** Relationship between features and target is linear.
- **Independence:** Observations independent.
- **Homoscedasticity:** Constant error variance.
- **Normality:** Errors are normally distributed.
- **No multicollinearity:** Predictors not highly correlated.

### 3.2 Interpreting Coefficients
- Each coefficient: Expected change in target for one-unit increase in feature, holding others constant.

### 3.3 R² Score
- Proportion of variance explained by the model.
- *Example:* R²=0.8 → 80% explained.

### 3.4 MSE vs MAE
- **MSE:** Penalizes large errors more (outlier-sensitive).
- **MAE:** Treats all errors equally (robust to outliers).

### 3.5 Detecting Multicollinearity
- **VIF (Variance Inflation Factor):** VIF > 5 or 10 is a problem.
- **Correlation matrix:** Look for high correlations.

### 3.6 Simple vs Multiple Regression
- **Simple:** One predictor.
- **Multiple:** Multiple predictors.

### 3.7 Can Linear Regression be used for Classification?
- No; use logistic regression for classification.

### 3.8 Consequences of Violating Assumptions
- Biased, unreliable results, invalid inference.

---

## 4. Logistic Regression

### 4.1 Logistic vs Linear Regression
- **Linear regression:** Predicts continuous values.
- **Logistic regression:** Predicts probability of class membership.

### 4.2 Sigmoid Function
- Maps any real number to [0, 1]; used to model probability.
- **Formula:** sigmoid(x) = 1 / (1 + exp(-x))
- *Example:* Input 0 → output 0.5.

### 4.3 Precision vs Recall
- **Precision:** Proportion of predicted positives that are correct.  
  Precision = TP / (TP + FP)
- **Recall:** Proportion of actual positives correctly identified.  
  Recall = TP / (TP + FN)

### 4.4 ROC-AUC Curve
- **ROC curve:** Plots True Positive Rate vs False Positive Rate at different thresholds.
- **AUC:** Area under ROC; higher AUC = better model.

### 4.5 Confusion Matrix
- **Definition:** Table showing TP, FP, TN, FN counts.
- **Example:**  
  |   | Predicted No | Predicted Yes |  
  |---|--------------|---------------|  
  | Actual No | TN | FP |  
  | Actual Yes | FN | TP |

### 4.6 Handling Imbalanced Classes
- Model may favor majority class, poor minority class detection.
- **Solution:** Resampling, class weights, or alternative metrics like F1-score.

### 4.7 Choosing the Threshold
- Threshold determines cutoff for classifying as positive.
- Choose based on maximizing F1-score, precision, recall, or ROC curve.

### 4.8 Multi-class Logistic Regression
- Use techniques like one-vs-rest (OvR) or multinomial logistic regression.
- *Example:* Classifying images into three categories using OvR.

---

## 5. K-Nearest Neighbors (KNN)

### 5.1 How KNN Works
1. Choose the value of K (number of neighbors).
2. Calculate the distance from the new data point to all training points.
3. Pick the K nearest neighbors.
4. The most common class among those neighbors is the prediction.

### 5.2 Choosing the Right K
- Try different K values and use the one that gives the best validation accuracy.
- Too small K can be noisy; too large K can make the model less sensitive to patterns.

### 5.3 Importance of Normalization in KNN
- KNN relies on distance; features with bigger scales can dominate.
- Normalization ensures all features contribute fairly.

### 5.4 Time Complexity of KNN
- For every prediction, KNN checks all training points.
- Time complexity: O(n × d), where n = number of training points, d = number of features.

### 5.5 Pros & Cons of KNN
**Pros:**  
- Simple, easy to understand  
- No training phase  
- Works for multi-class problems

**Cons:**  
- Slow with large data  
- Needs feature scaling  
- Sensitive to noisy or irrelevant data

### 5.6 Sensitivity to Noise
- KNN can be affected by noisy and incorrect data points, especially if K is small.

### 5.7 Handling Multi-class Problems
- KNN works naturally for multi-class classification by picking the most common class among the K nearest neighbors.

### 5.8 Role of Distance Metrics
- Distance metrics (like Euclidean or Manhattan) decide how “closeness” is measured and affect the final prediction.

---

## Reference

Inspired by [GeeksforGeeks – K-Nearest Neighbours](https://www.geeksforgeeks.org/k-nearest-neighbours/)
