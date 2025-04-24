"""
Model Training script:
- Loads features
- Splits into train/test
- Trains a RandomForest + GridSearch
- Evaluates & serializes best model
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

FEAT_PATH = os.path.join('data', 'features', 'transactions_features.csv')
MODEL_PATH = os.path.join('models', 'fraud_detector.joblib')


def train_model():
    df = pd.read_csv(FEAT_PATH)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators':[100,200],'max_depth':[5,10]}
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Classification report on test set:")
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train_model()