# credit_model_local.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = os.path.join("datasets", "credit_data.csv")  # your saved CSV
TARGET_COL = "creditworthy"

# ----------------------------
# LOAD DATA
# ----------------------------
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {DATA_PATH}, shape: {data.shape}")
print("Columns:", data.columns.tolist())

# Features and target
if TARGET_COL not in data.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

X = data.drop(TARGET_COL, axis=1)
y = data[TARGET_COL]

# ----------------------------
# SCALE FEATURES
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# SPLIT DATA (stratified to maintain class balance)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------
# DEFINE MODELS
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# ----------------------------
# TRAIN AND EVALUATE
# ----------------------------
plt.figure(figsize=(8,6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Try predict_proba for ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred, zero_division=0))

    if y_proba is not None and len(set(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {auc:.2f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    else:
        print("ROC-AUC not available (only one class in test or predict_proba missing).")

# ----------------------------
# PLOT ROC CURVE
# ----------------------------
if plt.gca().has_data():
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No ROC curves to show.")
