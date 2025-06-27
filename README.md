# Data Preprocessing, EDA, Linear Regression & Logistic Regression: Key Concepts

This document summarizes essential questions and answers on data preprocessing, exploratory data analysis (EDA), linear regression, and logistic regression in machine learning—with clear definitions and examples.

---

## Data Preprocessing

### 1. What are the different types of missing data?
- **MCAR (Missing Completely at Random):**  
  Definition: Missingness is unrelated to any data.  
  Example: Sensor randomly fails to record data.
- **MAR (Missing at Random):**  
  Definition: Missingness is related to observed data, not the missing value.  
  Example: Income missing more often for younger respondents.
- **MNAR (Missing Not at Random):**  
  Definition: Missingness depends on the value of the missing data itself.  
  Example: People with high income less likely to report it.

### 2. How do you handle categorical variables?
- **Label Encoding:** Assigns an integer to each category.  
  Example: [Red, Green, Blue] → [0, 1, 2]
- **One-Hot Encoding:** Creates binary columns for each category.  
  Example: [Red, Green, Blue] → [1,0,0], [0,1,0], [0,0,1]

### 3. What is the difference between normalization and standardization?
- **Normalization:** Scale data to [0, 1].  
  Example: [10, 20, 30] → [0, 0.5, 1]
- **Standardization:** Scale data to mean 0, std 1.  
  Example: [10, 20, 30] → [-1.22, 0, 1.22]

### 4. How do you detect outliers?
- **Statistical:** Z-score, IQR.  
  Example: 100 in [10, 12, 11, 13, 100] is an outlier.
- **Visualization:** Box plots, scatter plots.

### 5. Why is preprocessing important in ML?
- **Definition:** Clean and consistent data improves model performance.
- **Example:** Imputing missing values increases accuracy.

### 6. What is one-hot encoding vs label encoding?
- **One-Hot:** Binary columns for each category.
- **Label:** Integer for each category.

### 7. How do you handle data imbalance?
- **Resampling:** Over/under-sample classes.
- **SMOTE:** Synthetic samples.
- **Class weights:** Adjust model penalty.

### 8. Can preprocessing affect model accuracy?
- **Definition:** Yes, good preprocessing improves accuracy.
- **Example:** Scaling improves KNN; not handling missing values hurts models.

---

## Exploratory Data Analysis (EDA)

### 1. What is the purpose of EDA?
- **Definition:** Summarize, visualize, and understand data before modeling.
- **Example:** Plotting histograms for feature distribution.

### 2. How do boxplots help in understanding a dataset?
- **Definition:** Visualize distribution, median, quartiles, and outliers.
- **Example:** Boxplot of salaries shows high earners as outliers.

### 3. What is correlation and why is it useful?
- **Definition:** Measures linear relationship between variables.
- **Example:** Height and weight are positively correlated.

### 4. How do you detect skewness in data?
- **Definition:** Skewness shows asymmetry.  
  Example: Histogram tail to the right = right-skewed.

### 5. What is multicollinearity?
- **Definition:** High correlation between predictors.
- **Example:** Age and years of experience highly correlated.

### 6. What tools do you use for EDA?
- **Examples:** Pandas, Matplotlib, Seaborn, Plotly, R, Tableau.

### 7. Can you explain a time when EDA helped you find a problem?
- **Example:** EDA revealed duplicated sales records causing a data spike.

### 8. What is the role of visualization in ML?
- **Definition:** Helps detect patterns, outliers, relationships, and communicate results.
- **Example:** Heatmaps for correlations, scatter plots for relationships.

---

## Linear Regression

### 1. What assumptions does linear regression make?
- **Linearity:** Relationship between features and target is linear.
- **Independence:** Observations independent.
- **Homoscedasticity:** Constant error variance.
- **Normality:** Errors are normally distributed.
- **No multicollinearity:** Predictors not highly correlated.

### 2. How do you interpret the coefficients?
- Each coefficient: Expected change in target for one-unit increase in feature, holding others constant.

### 3. What is R² score and its significance?
- **Definition:** Proportion of variance explained by the model.
- **Example:** R²=0.8 → 80% explained.

### 4. When would you prefer MSE over MAE?
- **MSE:** Penalizes large errors more (outlier-sensitive).
- **MAE:** Treats all errors equally (robust to outliers).

### 5. How do you detect multicollinearity?
- **VIF (Variance Inflation Factor):** VIF > 5 or 10 is a problem.
- **Correlation matrix:** Look for high correlations.

### 6. What is the difference between simple and multiple regression?
- **Simple:** One predictor.
- **Multiple:** Multiple predictors.

### 7. Can linear regression be used for classification?
- **Definition:** Not suitable; use logistic regression for classification.

### 8. What happens if you violate regression assumptions?
- **Effect:** Biased, unreliable results, invalid inference.

---

## Logistic Regression

### 1. How does logistic regression differ from linear regression?
- **Linear regression:** Predicts continuous values.
- **Logistic regression:** Predicts probability of class membership (classification).

### 2. What is the sigmoid function?
- **Definition:** Maps any real number to [0, 1]; used to model probability.
- **Formula:**  
  sigmoid(x) = 1 / (1 + exp(-x))
- **Example:** Input 0 → output 0.5.

### 3. What is precision vs recall?
- **Precision:** Proportion of predicted positives that are correct.  
  Precision = TP / (TP + FP)
- **Recall:** Proportion of actual positives correctly identified.  
  Recall = TP / (TP + FN)

### 4. What is the ROC-AUC curve?
- **ROC curve:** Plots True Positive Rate vs False Positive Rate at different thresholds.
- **AUC:** Area under ROC; higher AUC = better model.

### 5. What is the confusion matrix?
- **Definition:** Table showing TP, FP, TN, FN counts.
- **Example:**  
  |   | Predicted No | Predicted Yes |  
  |---|--------------|---------------|  
  | Actual No | TN | FP |  
  | Actual Yes | FN | TP |

### 6. What happens if classes are imbalanced?
- **Effect:** Model may favor majority class, poor minority class detection.
- **Solution:** Resampling, class weights, or alternative metrics like F1-score.

### 7. How do you choose the threshold?
- **Definition:** Threshold determines cutoff for classifying as positive.
- **Method:** Choose based on maximizing F1-score, precision, recall, or ROC curve.

### 8. Can logistic regression be used for multi-class problems?
- **Definition:** Yes, using techniques like one-vs-rest (OvR) or multinomial logistic regression.
- **Example:** Classifying images into three categories using OvR.

---
