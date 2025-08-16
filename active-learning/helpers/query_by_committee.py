import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


def evaluate_grid_search(X_train, y_train, model, param_grid, cv_folds=5):
    """
    Performs hyperparameter tuning using GridSearchCV and prints performance metrics per CV iteration,
    including F1-score, Precision, and Recall.

    Parameters:
    - X_train: Training feature set.
    - y_train: Training labels.
    - model: Machine learning model to optimize.
    - param_grid: Dictionary of hyperparameters to search.
    - cv_folds: Number of folds for cross-validation.

    Returns:
    - Best trained model with optimal hyperparameters.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="f1", n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"\nBest Parameters for {model.__class__.__name__}: {best_params}")

    # Extract results for all CV iterations
    results = grid_search.cv_results_
    mean_test_f1_scores = results["mean_test_score"]

    print(f"\n{model.__class__.__name__} - Grid Search CV Performance Per Fold:")

    f1_scores, precision_scores, recall_scores = [], [], []

    # Ensure the number of scores matches the number of CV folds
    num_folds = min(cv_folds, len(mean_test_f1_scores))

    for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        if i >= num_folds:  
            break  # Avoid out-of-bounds error if GridSearch returned fewer scores than expected
        best_model = model.set_params(**best_params)  # Apply best params
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        best_model.fit(X_train_fold, y_train_fold)
        y_val_pred = best_model.predict(X_val_fold)

        f1 = f1_score(y_val_fold, y_val_pred)
        precision = precision_score(y_val_fold, y_val_pred)
        recall = recall_score(y_val_fold, y_val_pred)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Fold {i+1}: F1 Score = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

    print(f"\nMean CV F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Mean CV Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean CV Recall: {np.mean(recall_scores):.4f}")

    # Train final model with best hyperparameters
    best_model_final = model.set_params(**best_params)
    best_model_final.fit(X_train, y_train)

    return best_model_final


def train_qbc_committee(X_train, y_train, cv_folds=5):
    """
    Trains a committee of models for Query-by-Committee (QBC) with hyperparameter tuning.

    Returns:
    - Dictionary of best trained models.
    """
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42), 
                                {"C": [0.001, 0.01, 0.1, 1] #, 10]
                                 }),

        "Random Forest": (RandomForestClassifier(random_state=42),
                          {"n_estimators": [30, 50, 100], 
                           "max_depth": [3, 5, 7], 
                           "min_samples_leaf": [4,5,6] #[1, 5, 10]
                           }),

        "Extra Trees": (ExtraTreesClassifier(random_state=42),
                        {"n_estimators": [30, 50, 100], 
                         "max_depth": [3, 5, 7], 
                         "min_samples_leaf": [4,5,6] #[1, 5, 10]
                         }),

        "SVM": (SVC(kernel="rbf", probability=True, random_state=42),
                {"C": [0.01, 0.1, 1] #, 10, 100]
                 })
    }

    trained_models = {}

    for name, (model, param_grid) in models.items():
        print(f"\n--- Hyperparameter Tuning for {name} ---")
        trained_models[name] = evaluate_grid_search(X_train, y_train, model, param_grid, cv_folds=5)

    return trained_models


def select_qbc_samples(X_unlabeled, models):
    """
    Uses Query-by-Committee (QBC) to identify the most uncertain samples and returns
    X_unlabeled with added model prediction scores + the most uncertain CUSTOM_IDs.

    Parameters:
    - X_unlabeled: Unlabeled feature set to score.
    - models: Dictionary of trained models.
    - n_samples: Number of uncertain samples to return.

    Returns:
    - X_unlabeled with additional model prediction score columns.
    - CUSTOM_IDs of the most uncertain samples.
    """
    # Make a copy of X_unlabeled to store model scores without modifying the original
    X_scores = X_unlabeled.copy()

    # Extract only the original training features (avoid modifying X_unlabeled directly)
    X_original = X_unlabeled.iloc[:, :models["Logistic Regression"].n_features_in_]

    model_predictions = np.zeros((X_original.shape[0], len(models)))

    # Store predictions in separate columns
    for i, (name, model) in enumerate(models.items()):
        X_scores[name + "_score"] = model.predict_proba(X_original)[:, 1]  # Use only original features
        model_predictions[:, i] = X_scores[name + "_score"]

    # Compute disagreement (variance across model predictions)
    disagreement_scores = np.var(model_predictions, axis=1)
    X_scores["disagreement_score"] = disagreement_scores

    return X_scores


def stacking_models(X_train, models):
    # Make a copy of X_unlabeled to store model scores without modifying the original
    X_scores = X_train.copy()

    # Extract only the original training features (avoid modifying X_unlabeled directly)
    X_original = X_train.iloc[:, :models["Logistic Regression"].n_features_in_]

    # Store predictions in separate columns
    for i, (name, model) in enumerate(models.items()):
        X_scores[name + "_score"] = model.predict_proba(X_original)[:, 1]  # Use only original features

    return X_scores
