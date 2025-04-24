#!/usr/bin/env bash
set -e

# 1. Clone the repo containing the sampled fraud data
echo "Cloning fraud-detection repo..."
git clone https://github.com/Mo-Khalifa96/Transaction-Fraud-Detection.git temp_repo

# 2. Create raw data directory and move file
mkdir -p data/raw
echo "Moving dataset to data/raw/"
mv temp_repo/FraudData_sampled.csv data/raw/transactions.csv

# 3. Clean up
echo "Removing temporary repo"
rm -rf temp_repo

echo "Setup complete. Raw data available at data/raw/transactions.csv"