import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/creditcard.csv"


def load_artifacts():
    """Load trained model and scaler."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python3 src/train.py"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "Run: python3 src/train.py"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def preprocess_transaction(transaction, scaler):
    """
    Prepare one transaction for the trained model.

    The training pipeline scaled only:
    - Time
    - Amount

    V1-V28 are already PCA-transformed features.
    """

    df = pd.DataFrame([transaction])

    scale_columns = ["Time", "Amount"]

    df[scale_columns] = scaler.transform(
        df[scale_columns]
    )

    return df


def predict_transaction(transaction):
    """Predict whether a transaction is fraudulent."""

    model, scaler = load_artifacts()

    processed = preprocess_transaction(
        transaction,
        scaler
    )

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(
        processed
    )[0][1]

    return prediction, probability


def get_sample_transaction():
    """
    Load a real transaction from the dataset
    for demonstration purposes.
    """

    df = pd.read_csv(DATA_PATH)

    # Select a known legitimate transaction.
    legitimate = df[df["Class"] == 0].iloc[0]

    transaction = legitimate.drop("Class").to_dict()

    return transaction


def print_prediction(transaction):
    """Display prediction in a readable format."""

    prediction, probability = predict_transaction(
        transaction
    )

    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 60)

    if prediction == 1:
        print("\nPrediction: FRAUD")
    else:
        print("\nPrediction: LEGITIMATE")

    print(
        f"Fraud Probability: {probability * 100:.2f}%"
    )

    print(
        f"Legitimate Probability: "
        f"{(1 - probability) * 100:.2f}%"
    )

    print("\n" + "-" * 60)

    if prediction == 1:
        print(
            "⚠️  WARNING: This transaction has been "
            "classified as potentially fraudulent."
        )
    else:
        print(
            "✓ This transaction has been classified "
            "as legitimate."
        )

    print("-" * 60)


def main():
    print("\nLoading trained model...")

    transaction = get_sample_transaction()

    print("Using a real transaction from the dataset")
    print("for prediction demonstration.")

    print_prediction(transaction)


if __name__ == "__main__":
    main()
