import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================

print("\n[2/8] Analysing dataset...")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[3/8] Creating EDA plots...")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/class_distribution.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(10, 5))
sns.histplot(
    data=df,
    x="Amount",
    bins=100,
    kde=True
)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/amount_distribution.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)
plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Amount")
plt.ylim(0, df["Amount"].quantile(0.99))
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/amount_by_class.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

print("\n[4/8] Preparing features...")

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Features: {X.shape}")
print(f"Target: {y.shape}")


# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

print("\n[5/8] Scaling features...")

scaler = StandardScaler()

# Scale Time and Amount.
# V1-V28 are already PCA-transformed and standardized.
scale_columns = ["Time", "Amount"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

X_test_scaled[scale_columns] = scaler.transform(
    X_test[scale_columns]
)

print("Scaled columns:", scale_columns)


# ============================================================
# 6. TRAIN MODELS
# ============================================================

print("\n[6/8] Training machine learning models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
}

results = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    results[name] = {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

print("\n[7/8] Evaluating models...")

for name, result in results.items():

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            result["predictions"],
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            digits=4
        )
    )

    cm = confusion_matrix(
        y_test,
        result["predictions"]
    )

    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Legitimate",
            "Fraud"
        ],
        yticklabels=[
            "Legitimate",
            "Fraud"
        ]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ROC CURVES
# ============================================================

plt.figure(figsize=(9, 6))

for name, result in results.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        result["probabilities"]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {result['roc_auc']:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/roc_curve_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(9, 6))

for name, result in results.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        result["probabilities"]
    )

    ap = result["average_precision"]

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.4f})"
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/precision_recall_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

print("\n[8/8] Selecting best model...")

best_model_name = max(
    results,
    key=lambda name: results[name]["average_precision"]
)

best_model = results[best_model_name]["model"]

print(f"\nBest model: {best_model_name}")

print(
    f"Best Average Precision: "
    f"{results[best_model_name]['average_precision']:.4f}"
)

print(
    f"Best ROC-AUC: "
    f"{results[best_model_name]['roc_auc']:.4f}"
)


# ============================================================
# SAVE MODEL + SCALER
# ============================================================

model_path = f"{MODEL_DIR}/fraud_detection_model.pkl"
scaler_path = f"{MODEL_DIR}/scaler.pkl"

joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print("\nSaved model:")
print(model_path)

print("\nSaved scaler:")
print(scaler_path)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

comparison = []

for name, result in results.items():

    report = classification_report(
        y_test,
        result["predictions"],
        output_dict=True
    )

    comparison.append({
        "Model": name,
        "ROC_AUC": result["roc_auc"],
        "Average_Precision": result["average_precision"],
        "Precision_Fraud": report["1"]["precision"],
        "Recall_Fraud": report["1"]["recall"],
        "F1_Fraud": report["1"]["f1-score"],
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    by="Average_Precision",
    ascending=False
)

comparison_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(comparison_df.to_string(index=False))


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("PROJECT TRAINING COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print("  models/fraud_detection_model.pkl")
print("  models/scaler.pkl")
print("  outputs/class_distribution.png")
print("  outputs/amount_distribution.png")
print("  outputs/amount_by_class.png")
print("  outputs/model_comparison.csv")
print("  outputs/roc_curve_comparison.png")
print("  outputs/precision_recall_curve.png")
print("  outputs/*_confusion_matrix.png")

print("\nDone!")
