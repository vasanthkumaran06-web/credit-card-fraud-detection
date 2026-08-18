from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

def load_artifacts():
    """Load the trained model and scaler."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python3 src/train.py"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "Run: python3 src/train.py"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ============================================================
# DEMO TRANSACTION
# ============================================================

def get_demo_transaction(model):
    """
    Create a demonstration transaction without requiring
    the original training dataset.
    """

    feature_names = list(model.feature_names_in_)

    transaction = {
        feature: 0.0
        for feature in feature_names
    }

    transaction["Time"] = 1000.0
    transaction["Amount"] = 50.0

    return transaction


# ============================================================
# PREPROCESS TRANSACTION
# ============================================================

def preprocess_transaction(transaction, scaler):
    """
    Prepare one transaction for prediction.

    Only Time and Amount were scaled during training.
    """

    df = pd.DataFrame([transaction])

    scale_columns = ["Time", "Amount"]

    df[scale_columns] = scaler.transform(
        df[scale_columns]
    )

    return df


# ============================================================
# PREDICTION
# ============================================================

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


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_prediction(transaction):
    """Display prediction in a readable format."""

    prediction, probability = predict_transaction(
        transaction
    )

    legitimate_probability = 1 - probability

    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 60)

    if prediction == 1:
        print("\nPrediction: FRAUD")
    else:
        print("\nPrediction: LEGITIMATE")

    print(
        f"Fraud Probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Legitimate Probability: "
        f"{legitimate_probability * 100:.2f}%"
    )

    print("\n" + "-" * 60)

    if prediction == 1:
        print(
            "WARNING: This transaction has been "
            "classified as potentially fraudulent."
        )
    else:
        print(
            "This transaction has been classified "
            "as legitimate."
        )

    print("-" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading trained model...")

    model, _ = load_artifacts()

    print(
        "Using a built-in demonstration transaction."
    )

    print(
        "The original training dataset is not required "
        "for prediction."
    )

    transaction = get_demo_transaction(model)

    print_prediction(transaction)


if __name__ == "__main__":
    main()
