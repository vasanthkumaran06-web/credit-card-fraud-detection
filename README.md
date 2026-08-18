# Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using supervised classification techniques.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of all transactions.

This project explores the dataset, prepares the features, trains multiple machine learning models, evaluates their performance using fraud-focused metrics, and saves the best-performing model for future predictions.

## Dataset

The dataset contains:

- 284,807 transactions
- 30 input features
- 1 target variable
- 492 fraudulent transactions
- 284,315 legitimate transactions

The target column is:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.17% of all transactions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Git
- GitHub

## Machine Learning Models

Two classification models were evaluated:

### Logistic Regression

ROC-AUC:

`0.9722`

Average Precision:

`0.7159`

Fraud Recall:

`91.84%`

Fraud F1-score:

`11.44%`

### Random Forest

ROC-AUC:

`0.9697`

Average Precision:

`0.8451`

Fraud Precision:

`89.41%`

Fraud Recall:

`77.55%`

Fraud F1-score:

`83.06%`

## Best Model

Random Forest was selected as the final model based on Average Precision and fraud-class performance.

### Final Results

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9697 |
| Average Precision | 0.8451 |
| Fraud Precision | 89.41% |
| Fraud Recall | 77.55% |
| Fraud F1-score | 83.06% |
| Accuracy | 99.95% |

## Why Accuracy Is Not Enough

The dataset is extremely imbalanced.

Only approximately 0.17% of transactions are fraudulent.

Because of this imbalance, accuracy alone can be misleading.

Therefore, this project focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

These metrics provide a better understanding of how effectively the model detects fraudulent transactions.

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── amount_by_class.png
│   ├── model_comparison.csv
│   ├── roc_curve_comparison.png
│   ├── precision_recall_curve.png
│   └── confusion matrices
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── test_fraud.py
│
├── requirements.txt
└── README.md
