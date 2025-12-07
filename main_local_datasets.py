import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_dataset(name="diabetes", datasets_dir="datasets"):
    """Load dataset from local CSV files in datasets_dir.

    Arguments:
        name: 'diabetes', 'breast_cancer', or 'heart'
        datasets_dir: folder where CSV files are stored

    Returns:
        pandas.DataFrame
    """
    # map dataset names to local filenames
    files = {
        "diabetes": "diabetes.csv",
        "breast_cancer": "BreastCancer.csv",
        "heart": "heart.csv",
    }

    if name not in files:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: {list(files.keys())}")

    path = os.path.join(datasets_dir, files[name])

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}\nMake sure you saved the CSV into the '{datasets_dir}' folder.")

    df = pd.read_csv(path)

    # small cleaning steps so the rest of your code works
    if name == "breast_cancer":
        # some versions have column 'Id' and diagnosis 'M'/'B'
        if 'Id' in df.columns:
            df.drop(['Id'], axis=1, inplace=True)
        if 'diagnosis' in df.columns and df['diagnosis'].dtype == object:
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def train_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n----- {model_name} -----")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))


def main():
    # Choose the dataset here: 'diabetes', 'breast_cancer', or 'heart'
    dataset_name = "diabetes"
    datasets_dir = "datasets"

    try:
        df = load_dataset(dataset_name, datasets_dir=datasets_dir)
    except Exception as e:
        print("ERROR loading dataset:", e)
        sys.exit(1)

    print(f"Loaded dataset '{dataset_name}' with shape: {df.shape}")
    print(df.head())

    # Prepare X and y according to dataset
    if dataset_name == "breast_cancer":
        if 'diagnosis' not in df.columns:
            raise KeyError("'diagnosis' column not found in breast cancer CSV")
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]
    elif dataset_name == "diabetes":
        if 'Outcome' not in df.columns:
            raise KeyError("'Outcome' column not found in diabetes CSV")
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
    elif dataset_name == "heart":
        # heart dataset commonly uses 'target' column for label
        if 'target' not in df.columns:
            # try common alternative column names
            if 'Target' in df.columns:
                df.rename(columns={'Target': 'target'}, inplace=True)
            else:
                raise KeyError("'target' column not found in heart CSV")
        X = df.drop("target", axis=1)
        y = df["target"]

    # Split and scale
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
