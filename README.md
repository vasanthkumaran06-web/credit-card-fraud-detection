# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using supervised classification techniques.

The project focuses on handling highly imbalanced transaction data, comparing multiple classification models, and evaluating fraud detection performance using precision, recall, F1-score, ROC-AUC, and Average Precision.

## Project Overview

Credit card fraud detection is an imbalanced classification problem where fraudulent transactions represent a very small percentage of all transactions.

This project performs:

- Data loading and analysis
- Exploratory Data Analysis (EDA)
- Feature preprocessing
- Train/test splitting with stratification
- Feature scaling
- Logistic Regression training
- Random Forest training
- Model comparison
- Confusion matrix analysis
- ROC curve analysis
- Precision-Recall curve analysis
- Model selection
- Model serialization using Joblib
- Transaction prediction

## Dataset

The project uses the Credit Card Fraud Detection dataset containing:

- 284,807 transactions
- 30 input features
- 1 target variable
- 492 fraudulent transactions
- 284,315 legitimate transactions

The target column is:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

Fraudulent transactions represent approximately 0.17% of the dataset, making this a highly imbalanced classification problem.

### Dataset Source

The dataset can be obtained from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The raw dataset is intentionally not included in this repository because the CSV file exceeds GitHub's recommended repository file size limits.

After downloading the dataset, place the file here:

```text
data/creditcard.csv
