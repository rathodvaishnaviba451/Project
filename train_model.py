import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# ====================================
# LOAD DATA
# ====================================

print("Loading dataset...")

df = pd.read_csv(
    "data/creditcard.csv"
)

print(df.head())


# ====================================
# PREPROCESSING
# ====================================

print("\nPreprocessing...")

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]

scaler = StandardScaler()

X["Time"] = scaler.fit_transform(
    X["Time"].values.reshape(-1, 1)
)

X["Amount"] = scaler.fit_transform(
    X["Amount"].values.reshape(-1, 1)
)


# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ====================================
# BALANCING
# ====================================

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42
)

smote = SMOTE(
    random_state=42,
    k_neighbors=1
)
# ====================================
# ISOLATION FOREST
# ====================================

print("\nRunning Isolation Forest...")

iso = IsolationForest(
    contamination=0.01,
    random_state=42
)

iso.fit(
    X_train
)

iso_pred = iso.predict(
    X_test
)

iso_pred = np.where(
    iso_pred == -1,
    1,
    0
)

print(
    classification_report(
        y_test,
        iso_pred
    )
)


# ====================================
# LOCAL OUTLIER FACTOR
# ====================================

print("\nRunning LOF...")

lof = LocalOutlierFactor(
    contamination=0.01,
    novelty=True
)

lof.fit(
    X_train
)

lof_pred = lof.predict(
    X_test
)

lof_pred = np.where(
    lof_pred == -1,
    1,
    0
)

print(
    classification_report(
        y_test,
        lof_pred
    )
)


# ====================================
# XGBOOST
# ====================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)


# ====================================
# PREDICTION
# ====================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ====================================
# CLASSIFICATION REPORT
# ====================================

print("\nXGBoost Results:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ====================================
# CONFUSION MATRIX
# ====================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


# ====================================
# ROC CURVE
# ====================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_score = auc(
    fpr,
    tpr
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.show()


# ====================================
# SAVE MODEL
# ====================================

joblib.dump(
    model,
    "saved_model.pkl"
)

print(
    "\nModel saved successfully!"
)