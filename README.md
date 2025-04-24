# fraud-detection

## [Live Demo](https://fraud-detection-analysis.streamlit.app/)
## Data Analysis Report. 
### Introduction 

Financial fraud is a pervasive problem, and detecting it early is crucial for businesses and individuals alike. In this blog post, we'll embark on a data-driven journey to analyze a dataset of financial transactions and build models to predict fraudulent activities.

---

### Step 1: Data Loading and Initial Exploration

- **Raw data** loaded from data/raw/transactions.csv.

#### Raw features in the dataset:
- **step:** Represents a unit of time (1 step = 1 hour). Total steps: 744 (30 days).
- **type:** Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
- **amount:** Transaction amount.
- **nameOrig / nameDest:** Originator and recipient IDs.
- **oldbalanceOrg / newbalanceOrig:** Originator’s balances.
- **oldbalanceDest / newbalanceDest:** Recipient’s balances.

##### Target Feature:
- **isFraud:** 1 = fraudulent, 0 = legitimate

---

### Step 2: Unveiling the Target Variable

**Distribution of target variable:**

I analyzed the distribution of the target variable 'isFraud' to understand the prevalence of fraudulent transactions.

![Target Distribution](https://github.com/parvpareek/fraud-detection/blob/main/app/res/target_distribution.png)

> The dataset is heavily imbalanced.  
> A ratio of **119474:161** →  
> **99.863%** of transactions are non-fraud  
> **0.134%** are fraudulent

---

### Step 3: Data Cleaning - Taming Outliers

![Outlier Handling](https://github.com/parvpareek/fraud-detection/blob/main/app/res/outlier_handling.png)

- A few missing values in newBalanceOrig and oldBalanceDest were dropped.
- Numerical outliers were capped at the 99th percentile.

---

### Step 4: Data Analysis

#### Fraudulent Transactions per Step

![Fraud per Step](https://github.com/parvpareek/fraud-detection/blob/main/app/res/fraud_per_step.png)  
> Fraud happens randomly — no clear periodicity.

#### Cumulative Fraudulent Transactions over Time

![Cumulative Fraud](https://github.com/parvpareek/fraud-detection/blob/main/app/res/cumulative_fraud_over_time.png)  
> Cumulative fraud steadily increases — ongoing risk.

#### Distribution of Features: Fraud vs Non-Fraud

![Feature Distributions](https://github.com/parvpareek/fraud-detection/blob/main/app/res/distribution_features_fraud.png)

> Fraudulent transactions:
> - Tend to be larger in amount
> - Originate from and go to larger accounts

#### Fraud by Transaction Type

![Fraud by Type](https://github.com/parvpareek/fraud-detection/blob/main/app/res/Fraud%20by%20Type.png)  
> **TRANSFER** and **CASH_OUT** are most fraud-prone

#### Correlation Matrix

![Correlation Matrix](res/correlation_matrix.png)

> - oldbalanceOrg ~ newbalanceOrig  
> - oldbalanceDest ~ newbalanceDest  
> - No raw numeric feature has strong linear correlation with fraud

---

### Overall Insights

1. **Severe imbalance** demands stratified sampling or cost-sensitive learning  
2. **Relative features** (e.g. amount ÷ oldbalanceOrg) are crucial  
3. **Temporal patterns** exist, albeit subtly  
4. **Collinear features** should be combined or dropped

---

### Step 5: Feature Engineering - Crafting New Insights

**Created features:**
- Balance differences: balance_diff_orig, balance_diff_dest
- Interactions: amount_x_oldbalanceOrg, etc.
- One-hot encoded transaction type
- Flag for amount > 10% of oldbalanceOrg

---

### Step 6: Model Training and Evaluation

Models trained:
- **Random Forest** – accuracy: **89.97%**
- **Logistic Regression** – accuracy: **93.21%**
- **XGBoost** – accuracy: **99.97%**

![Confusion Matrix](https://github.com/parvpareek/fraud-detection/blob/main/app/res/confusion_matrix_xgboost.png)

#### XGBoost Feature Importance


![XGBoost Importances](https://github.com/parvpareek/fraud-detection/blob/main/app/res/feature_importance_xgboost.png)

Additional ROC chart from external source:

---

### Step 8: Hyperparameter Tuning

**Best Parameters**:

json
{
"learning_rate": 0.1,
"max_depth": 5,
"n_estimators": 100,
"subsample": 0.8
}   


## Summary:

### 1. Q&A


* **What is the distribution of fraudulent transactions in the dataset?** The dataset is highly imbalanced, with only 0.129% of transactions being fraudulent.
* **Are there any outliers in the numerical features?** Yes, there are outliers in several numerical features, notably 'amount' and balance columns.  These were handled by capping values at the 99th percentile.
* **How do different features relate to the target variable (fraud)?** Several features show correlation with fraud, most notably the transaction amount.  Transaction type also appears to be an important factor.
* **Which model performs best at predicting fraudulent transactions?** While all three models (Random Forest, Logistic Regression, and XGBoost) achieve high accuracy, precision, recall and F1-scores, the XGBoost models generally have a slightly better AUC-ROC score.
* **Can the XGBoost model be improved with hyperparameter tuning?** Yes, hyperparameter tuning using GridSearchCV improved the XGBoost's performance marginally, although some metrics decreased slightly.

### 2. Data Analysis Key Findings

* **Class Imbalance:** The target variable 'isFraud' is highly imbalanced, with only 129 (0.129%) fraudulent transactions out of 100,000. This imbalance needs to be addressed in modeling.
* **Outlier Handling:** Outliers in numerical features were capped at the 99th percentile.
* **Feature Importance:**  `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` are found to be important features for fraud detection.
* **Model Performance:**  All three models (Random Forest, Logistic Regression, and XGBoost) exhibit high accuracy (around 0.9997), precision (around 0.9997), recall (around 0.9997), and F1-score (around 0.9996).  XGBoost models generally perform better with AUC-ROC scores around 0.9996.
* **XGBoost Optimization:** Tuning the XGBoost model with GridSearchCV resulted in a slightly lower AUC-ROC score (0.9991) but comparable or better performance on other metrics compared to the initial XGBoost model.


### 3. Insights or Next Steps

* **Focus on AUC-ROC for model selection:**  While accuracy is high for all models, AUC-ROC may be a more reliable metric for this imbalanced dataset, highlighting the models' ability to distinguish between fraud and non-fraud effectively. Further analyze the specific business needs and determine which metric is the most important in this use-case.
* **Investigate other techniques for imbalanced datasets:** Explore techniques like oversampling (SMOTE), undersampling, or cost-sensitive learning to improve the models' ability to detect the minority class.
