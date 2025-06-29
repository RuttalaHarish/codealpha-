import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (simulated for demonstration)
data = pd.DataFrame({
    'income': [40000, 60000, 35000, 80000, 20000, 120000, 50000, 30000, 75000, 62000],
    'debt': [10000, 5000, 12000, 7000, 15000, 4000, 8000, 13000, 6000, 7000],
    'payment_history': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],  # 1 = good, 0 = bad
    'age': [25, 45, 30, 35, 22, 55, 40, 28, 38, 33],
    'creditworthy': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # Target variable
})

# Feature matrix and target vector
X = data.drop('creditworthy', axis=1)
y = data['creditworthy']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

# Plot ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
