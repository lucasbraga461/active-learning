import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_model(log_reg, X_test, y_test, threshold=0.5):
    """
    Evaluates a logistic regression model's performance with a given threshold.

    Parameters:
    - log_reg: Trained LogisticRegression model.
    - X_test: Test feature set.
    - y_test: True labels.
    - threshold: Decision threshold for classifying predictions (default = 0.5).
    - plot_confusion_matrix: Boolean flag to plot the confusion matrix (default = False).

    Returns:
    - A dictionary containing accuracy, precision, recall, F1 score, ROC-AUC, FP proportion, FN proportion.
    """

    # Ensure X_test and y_test have the same number of samples
    if len(X_test) != len(y_test):
        raise ValueError("Mismatch between X_test and y_test sizes.")

    # Get probability scores
    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

    # Apply threshold to get final binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Print results
    print(f"Model Performance (Threshold = {threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
