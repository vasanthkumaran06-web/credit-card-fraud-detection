import pandas as pd

from predict import predict_transaction


DATA_PATH = "data/creditcard.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    fraud_transaction = (
        df[df["Class"] == 1]
        .iloc[0]
        .drop("Class")
        .to_dict()
    )

    prediction, probability = predict_transaction(
        fraud_transaction
    )

    print("\n" + "=" * 60)
    print("FRAUD TRANSACTION TEST")
    print("=" * 60)

    print("\nActual Class: FRAUD")

    if prediction == 1:
        print("Model Prediction: FRAUD")
    else:
        print("Model Prediction: LEGITIMATE")

    print(
        f"Fraud Probability: {probability * 100:.2f}%"
    )

    print(
        f"Legitimate Probability: "
        f"{(1 - probability) * 100:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
