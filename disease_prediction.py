import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Load dataset - you can change to breast cancer or heart data
def load_dataset(name="diabetes"):
    if name == "diabetes":
        df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    elif name == "breast_cancer":
        df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv")
        df.drop(['Id'], axis=1, inplace=True)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    elif name == "heart":
        df = pd.read_csv("https://raw.githubusercontent.com/anshupandey/Machine_Learning_Training/master/heart.csv")
    else:
        raise ValueError("Unsupported dataset")
    return df


# Train and evaluate model
def train_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n----- {model_name} -----")
    print(f"Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def main():
    dataset_name = "diabetes"  # Change to 'breast_cancer' or 'heart'
    df = load_dataset(dataset_name)

    if dataset_name == "breast_cancer":
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]
    elif dataset_name == "diabetes":
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
    elif dataset_name == "heart":
        X = df.drop("target", axis=1)
        y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (SVC(kernel="linear"), "SVM"),
        (RandomForestClassifier(n_estimators=100), "Random Forest"),
        (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), "XGBoost")
    ]

    for clf, name in models:
        train_model(X_train, X_test, y_train, y_test, clf, name)


if __name__ == "__main__":
    main()
