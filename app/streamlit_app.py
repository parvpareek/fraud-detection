import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="centered", page_title="Fraud Detection App")

MODEL_PATH    = os.path.join('models','fraud_detector.joblib')
SAMPLES_PATH = os.path.join('data','samples','samples.csv')

@st.cache_data()
def load_samples():
    df = pd.read_csv(SAMPLES_PATH)
    return df

# Load artifacts
model   = joblib.load(MODEL_PATH)
samples = load_samples()

page = st.sidebar.radio("🔀 Navigate", ["Fraud Detection", "Analysis Report"])

if page == "Fraud Detection":
    st.title("⚠️ Fraud Detection Demo")

    # Centering CSS
    st.markdown(
        """
        <style>
          .stDataFrame > div { margin-left:auto; margin-right:auto; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Show samples
    st.subheader("🔍 Six Random Samples")
    st.dataframe(samples.drop(columns=['sample_id']), width=900, height=280)

    # Pick one
    choice = st.selectbox("➡️ Choose a sample to score", samples['sample_id'].tolist())
    row = samples[samples.sample_id == choice].drop(columns=['sample_id'])

    st.subheader("📋 Selected Transaction Features")
    st.dataframe(row, width=900, height=100)

    # Predict button
    if st.button("▶️ Predict Fraud"):
        X = row.drop('isFraud', axis=1)
        prob   = model.predict_proba(X)[0,1]
        actual = int(row['isFraud'].iloc[0])
        st.write(f"**Actual Label:** {actual}")
        st.write(f"**Predicted Fraud Probability:** {prob:.2%}")

elif page == "Analysis Report":
    st.title("📊 Analysis Report")
    st.markdown("""
    ## Introduction

    Financial fraud is a pervasive problem, and detecting it early is crucial for businesses and individuals alike. In this blog post, we'll embark on a data-driven journey to analyze a dataset of financial transactions and build models to predict fraudulent activities.

    ---

    ### Step 1: Data Loading and Initial Exploration

    - **Raw data** loaded from `data/raw/transactions.csv`.

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

    - A few missing values in `newBalanceOrig` and `oldBalanceDest` were dropped.
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

    > - `oldbalanceOrg` ~ `newbalanceOrig`  
    > - `oldbalanceDest` ~ `newbalanceDest`  
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
    - Balance differences: `balance_diff_orig`, `balance_diff_dest`
    - Interactions: `amount_x_oldbalanceOrg`, etc.
    - One-hot encoded transaction `type`
    - Flag for `amount > 10% of oldbalanceOrg`

    ---

    ### Step 6: Model Training and Evaluation

    Models trained:
    - **Random Forest** – accuracy: **89.97%**
    - **Logistic Regression** – accuracy: **93.21%**
    - **XGBoost** – accuracy: **99.97%**

    #### XGBoost Feature Importance

    ![XGBoost Importances](https://github.com/parvpareek/fraud-detection/blob/main/app/res/feature_importance_xgboost.png)

    Additional ROC chart from external source:


    ---

    ### Step 8: Hyperparameter Tuning

    **Best Parameters**:
    ```json
    {
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 100,
    "subsample": 0.8
    }   
""")