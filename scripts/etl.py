"""
Extract & Transform script:
- Reads raw CSV from data/raw/
- Cleans missing/outlier values
- Writes cleaned CSV to data/processed/
"""
import os
import pandas as pd

RAW_PATH = os.path.join('data', 'raw', 'transactions.csv')
CLEAN_PATH = os.path.join('data', 'processed', 'cleaned.csv')
NUM_FEATURES = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']


def run_etl():
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df = df.dropna()

    # Cap outliers at 99th percentile
    for col in NUM_FEATURES:
        threshold = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=threshold)

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data written to {CLEAN_PATH}")


if __name__ == '__main__':
    run_etl()