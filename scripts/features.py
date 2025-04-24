"""
Feature Engineering script:
- Loads cleaned data
- Generates aggregate & interaction features
- Writes features CSV to data/features/
"""
import os
import pandas as pd

CLEAN_PATH = os.path.join('data', 'processed', 'cleaned.csv')
FEAT_PATH = os.path.join('data', 'features', 'transactions_features.csv')


def build_features():
    df = pd.read_csv(CLEAN_PATH)

    # Balance diffs
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Interaction terms
    for col in ['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
        df[f'amount_x_{col}'] = df['amount'] * df[col]

    # One-hot encode transaction type
    dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df.drop('type', axis=1), dummies], axis=1)

    # Flag if amount >10% of original balance
    df['amount_gt_10pct_orig'] = (df['amount'] > 0.1 * df['oldbalanceOrg']).astype(int)

    os.makedirs(os.path.dirname(FEAT_PATH), exist_ok=True)
    df.to_csv(FEAT_PATH, index=False)
    print(f"Features written to {FEAT_PATH}")


if __name__ == '__main__':
    build_features()