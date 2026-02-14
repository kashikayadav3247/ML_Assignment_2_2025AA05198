import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("ML Assignment 2 - Classification Models")

st.write("Upload test dataset (CSV format).")

# Upload test dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Model selection dropdown
model_choice = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(data.head())

    X = data.drop("target", axis=1)
    y = data["target"]

    scaler = joblib.load("model/scaler.pkl")

    X_scaled = scaler.transform(X)

    if model_choice == "Logistic Regression":
        model = joblib.load("model/logistic_regression.pkl")

    elif model_choice == "Decision Tree":
        model = joblib.load("model/decision_tree.pkl")

    elif model_choice == "KNN":
        model = joblib.load("model/knn.pkl")

    elif model_choice == "Naive Bayes":
        model = joblib.load("model/naive_bayes.pkl")

    elif model_choice == "Random Forest":
        model = joblib.load("model/random_forest.pkl")

    elif model_choice == "XGBoost":
        model = joblib.load("model/xgboost.pkl")

    y_pred = model.predict(X_scaled)

    # Evaluation Metrics
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)

    try:
        auc = roc_auc_score(y, model.predict_proba(X_scaled), multi_class="ovr")
    except:
        auc = "Not available"

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"AUC: {auc}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
