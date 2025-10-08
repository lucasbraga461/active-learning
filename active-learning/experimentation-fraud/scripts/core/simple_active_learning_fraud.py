#!/usr/bin/env python3
"""
Active Learning for Credit Card Fraud Detection

This script implements active learning for the Kaggle European Credit Card Fraud dataset
with stratified passive learning to ensure fair experimental comparison.

Key Features:
- Stratified passive learning that maintains fraud representation throughout iterations
- Fair comparison between active and passive learning approaches
- Experimental rigor consistent with established benchmarks
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning'

def setup_logging(config_name):
    """Setup logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = f'{HOME_DIR}/active-learning/experimentation-fraud/data/logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{logs_dir}/experiment_log_{config_name}_{timestamp}.txt'
    
    # Create a custom class that writes to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # Ensure immediate writing
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    # Replace stdout with our logger
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"📝 Logging started - Output will be saved to: {log_filename}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return logger


def clean_fraud_dataset(df):
    """
    Clean and preprocess credit card fraud dataset for active learning
    
    Args:
        df: Raw credit card dataset with columns Time, V1-V28, Amount, Class
    
    Returns:
        X_encoded: Preprocessed features
        y: Target labels (0=non-fraud, 1=fraud)
    """
    
    print("🧹 Cleaning and preprocessing credit card fraud dataset...")
    
    # 1. Basic dataset info
    df_clean = df.copy()
    
    print(f"  📊 Original dataset shape: {df_clean.shape}")
    print(f"  🔍 Missing values in original data:")
    missing_counts = df_clean.isnull().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
    else:
        print("    No missing values found")
    
    # 2. Handle target variable
    y = df_clean['Class']
    print(f"  🎯 Target distribution: {y.value_counts().to_dict()}")
    fraud_percentage = (y == 1).sum() / len(y) * 100
    print(f"  ⚠️  Fraud percentage: {fraud_percentage:.3f}%")
    
    # 3. Feature engineering for fraud detection
    
    # Time features - extract hour of day (fraud patterns may be time-dependent)
    df_clean['Time_hour'] = (df_clean['Time'] / 3600) % 24
    df_clean['Time_hour_sin'] = np.sin(2 * np.pi * df_clean['Time_hour'] / 24)
    df_clean['Time_hour_cos'] = np.cos(2 * np.pi * df_clean['Time_hour'] / 24)
    print("  ✓ Added cyclical time features (hour of day)")
    
    # Amount features - log transformation for better distribution
    df_clean['Amount_log'] = np.log1p(df_clean['Amount'])  # log(1 + amount) to handle zeros
    print("  ✓ Added log-transformed amount feature")
    
    # Amount bins for interpretability 
    df_clean['Amount_bin'] = pd.cut(df_clean['Amount'], 
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['very_small', 'small', 'medium', 'large', 'very_large', 'extreme'])
    print("  ✓ Added amount bins for interpretability")
    
    # 4. Select features for modeling
    # Use V1-V28 (PCA components), Time features, and Amount features
    feature_columns = (
        [f'V{i}' for i in range(1, 29)] +  # V1-V28
        ['Time_hour_sin', 'Time_hour_cos'] +  # Time features
        ['Amount_log'] +  # Amount features
        ['Amount_bin']  # Amount bins
    )
    
    X = df_clean[feature_columns]
    
    # 5. Handle categorical features (Amount_bin)
    X_encoded = pd.get_dummies(X, columns=['Amount_bin'], prefix='Amount')
    print(f"  ✓ One-hot encoded categorical features")
    
    # 6. Handle any remaining NaN values (shouldn't be any, but safety check)
    print(f"  🔍 Checking for NaN values after preprocessing...")
    nan_counts = X_encoded.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"    Found NaN values in columns:")
        print(nan_counts[nan_counts > 0])
        
        # Fill NaN values with median for numerical columns
        print("    🧹 Filling NaN values...")
        for col in X_encoded.columns:
            if X_encoded[col].isnull().any():
                if X_encoded[col].dtype in ['int64', 'float64']:
                    median_val = X_encoded[col].median()
                    X_encoded[col] = X_encoded[col].fillna(median_val)
                    print(f"      {col}: filled with median ({median_val:.4f})")
                else:
                    # For categorical columns, fill with mode
                    mode_val = X_encoded[col].mode()[0]
                    X_encoded[col] = X_encoded[col].fillna(mode_val)
                    print(f"      {col}: filled with mode ({mode_val})")
    else:
        print("    ✓ No NaN values found after preprocessing")
    
    # 7. Feature standardization (important for fraud detection models)
    print(f"  🔧 Applying feature standardization...")
    scaler = StandardScaler()
    
    # Get all numerical columns for scaling
    numerical_cols = X_encoded.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_encoded.select_dtypes(include=['object', 'bool']).columns
    
    if len(numerical_cols) > 0:
        # Sort columns for reproducibility
        numerical_cols_sorted = sorted(numerical_cols)
        print(f"    🔒 Using sorted column order for reproducibility")
        
        # Apply standardization
        X_encoded[numerical_cols_sorted] = scaler.fit_transform(X_encoded[numerical_cols_sorted])
        print(f"    ✓ Standardized {len(numerical_cols_sorted)} numerical features")
    
    if len(categorical_cols) > 0:
        print(f"    ✓ Kept {len(categorical_cols)} categorical features unchanged (already 0/1)")
    
    # 8. Final info
    print(f"  📊 Final dataset shape: {X_encoded.shape}")
    print(f"  🎯 Final target distribution: {y.value_counts().to_dict()}")
    print(f"  📋 Feature columns: {list(X_encoded.columns)}")
    
    return X_encoded, y


def stratified_initial_split(X, y, initial_samples, random_state=42, min_fraud_samples=10):
    """
    Create stratified initial split ensuring minimum fraud representation
    
    Args:
        X: Feature matrix
        y: Target labels
        initial_samples: Total number of initial samples desired
        random_state: Random seed
        min_fraud_samples: Minimum number of fraud samples in initial set
    
    Returns:
        initial_indices: Indices for initial labeled pool
    """
    print(f"  🎯 Creating stratified initial split with {initial_samples} samples...")
    
    fraud_indices = X[y == 1].index.tolist()
    non_fraud_indices = X[y == 0].index.tolist()
    
    print(f"    Available fraud samples: {len(fraud_indices)}")
    print(f"    Available non-fraud samples: {len(non_fraud_indices)}")
    
    # Ensure we have enough fraud samples
    actual_fraud_samples = min(min_fraud_samples, len(fraud_indices))
    remaining_samples = initial_samples - actual_fraud_samples
    
    # Use stratified sampling to get the split
    rng = np.random.RandomState(random_state)
    
    # Sample fraud cases
    selected_fraud = rng.choice(fraud_indices, size=actual_fraud_samples, replace=False)
    
    # Sample non-fraud cases
    selected_non_fraud = rng.choice(non_fraud_indices, size=remaining_samples, replace=False)
    
    # Combine indices
    initial_indices = np.concatenate([selected_fraud, selected_non_fraud])
    
    # Shuffle the combined indices
    rng.shuffle(initial_indices)
    
    print(f"    ✓ Selected {actual_fraud_samples} fraud + {remaining_samples} non-fraud samples")
    print(f"    ✓ Initial fraud percentage: {actual_fraud_samples/initial_samples*100:.2f}%")
    
    return initial_indices


def load_and_split_data(data_path, test_size=0.2, random_state=42):
    """
    Load credit card fraud data and create proper train/test split
    
    Args:
        data_path: Path to the creditcard.csv file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    print("Loading and splitting credit card fraud data...")
    
    # Set numpy random seed for reproducibility
    np.random.seed(random_state)
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    
    # Clean and preprocess the fraud dataset
    X, y = clean_fraud_dataset(data)
    
    print(f"Features after preprocessing: {X.shape[1]}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Stratified split to maintain fraud ratio in both train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Label distribution in train: {y_train.value_counts().to_dict()}")
    print(f"Label distribution in test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def uncertainty_sampling(model, X_unlabeled, n_samples, threshold=0.5, window=0.05):
    """Select samples using uncertainty sampling"""
    # Safety check: Remove any rows with NaN values
    if X_unlabeled.isnull().any().any():
        print(f"  ⚠️  Warning: Found NaN values in unlabeled data, removing affected rows")
        X_unlabeled_clean = X_unlabeled.dropna()
        print(f"  📊 Removed {len(X_unlabeled) - len(X_unlabeled_clean)} rows with NaN values")
        if len(X_unlabeled_clean) < n_samples:
            print(f"  ⚠️  Warning: Not enough clean samples, returning all available")
            return X_unlabeled_clean
        X_unlabeled = X_unlabeled_clean
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_unlabeled)[:, 1]
    
    # Find samples in uncertainty region (around threshold)
    lower = threshold - window
    upper = threshold + window
    uncertain_mask = (probabilities > lower) & (probabilities <= upper)
    
    uncertain_samples = X_unlabeled[uncertain_mask]
    
    if len(uncertain_samples) >= n_samples:
        return uncertain_samples.sample(n_samples, random_state=42)
    else:
        # If not enough uncertain samples, add some from outside
        remaining = n_samples - len(uncertain_samples)
        other_samples = X_unlabeled[~uncertain_mask].sample(remaining, random_state=42)
        return pd.concat([uncertain_samples, other_samples])


def diversity_sampling(X_unlabeled, n_samples, k=10):
    """Select samples using diversity sampling (KNN-based)"""
    # Safety check: Remove any rows with NaN values
    if X_unlabeled.isnull().any().any():
        print(f"  ⚠️  Warning: Found NaN values in unlabeled data, removing affected rows")
        X_unlabeled_clean = X_unlabeled.dropna()
        print(f"  📊 Removed {len(X_unlabeled) - len(X_unlabeled_clean)} rows with NaN values")
        if len(X_unlabeled_clean) < n_samples:
            print(f"  ⚠️  Warning: Not enough clean samples, returning all available")
            return X_unlabeled_clean
        X_unlabeled = X_unlabeled_clean
    
    # Compute density using k-NN
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X_unlabeled)
    
    distances, _ = knn.kneighbors(X_unlabeled)
    density_scores = distances.mean(axis=1)
    
    # Create a temporary dataframe with density scores
    temp_df = X_unlabeled.copy()
    temp_df['density'] = density_scores
    
    # Select most diverse samples (highest density scores)
    selected = temp_df.nlargest(n_samples, 'density')
    
    # Remove the temporary density column and return
    return selected.drop(columns=['density'])


def qbc_sampling(X_unlabeled, n_samples, X_labeled, y_labeled):
    """Select samples using Query by Committee (QBC)"""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    
    # Safety check: Remove any rows with NaN values
    if X_unlabeled.isnull().any().any():
        print(f"  ⚠️  Warning: Found NaN values in unlabeled data, removing affected rows")
        X_unlabeled_clean = X_unlabeled.dropna()
        print(f"  📊 Removed {len(X_unlabeled) - len(X_unlabeled_clean)} rows with NaN values")
        if len(X_unlabeled_clean) < n_samples:
            print(f"  ⚠️  Warning: Not enough clean samples, returning all available")
            return X_unlabeled_clean
        X_unlabeled = X_unlabeled_clean
    
    # Train a committee of models optimized for fraud detection
    models = {
        'lr': LogisticRegression(
            C=0.1,                    # Regularization for generalization
            random_state=42, 
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced'   # Handle class imbalance
        ),
        'rf': RandomForestClassifier(
            n_estimators=50,          # Moderate number of trees
            max_depth=8,              # Reasonable depth for fraud data
            min_samples_split=20,     # Prevent overfitting
            min_samples_leaf=10,      # Ensure leaf purity
            random_state=42,
            class_weight='balanced'   # Handle class imbalance
        ),
        'et': ExtraTreesClassifier(
            n_estimators=50,          # Moderate number of trees
            max_depth=8,              # Reasonable depth
            min_samples_split=20,     # Prevent overfitting
            min_samples_leaf=10,      # Ensure leaf purity
            random_state=42,
            class_weight='balanced'   # Handle class imbalance
        ),
        'nb': GaussianNB()            # Naive Bayes - good baseline
    }
    
    # Train each model
    trained_models = {}
    for name, model in models.items():
        model.fit(X_labeled, y_labeled)
        trained_models[name] = model
    
    # Get predictions from all models
    predictions = {}
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_unlabeled)[:, 1]
            predictions[name] = (pred_proba > 0.5).astype(int)
        else:
            pred = model.predict(X_unlabeled)
            predictions[name] = pred
    
    # Calculate disagreement scores
    disagreement_scores = np.zeros(len(X_unlabeled))
    for i in range(len(X_unlabeled)):
        # Count how many models disagree with the majority
        votes = [predictions[name][i] for name in predictions.keys()]
        majority_vote = max(set(votes), key=votes.count)
        disagreement = sum(1 for vote in votes if vote != majority_vote)
        disagreement_scores[i] = disagreement
    
    # Create temporary dataframe with disagreement scores
    temp_df = X_unlabeled.copy()
    temp_df['disagreement'] = disagreement_scores
    
    # Select samples with highest disagreement
    selected = temp_df.nlargest(n_samples, 'disagreement')
    
    # Remove the temporary disagreement column and return
    return selected.drop(columns=['disagreement'])


def random_sampling(X_unlabeled, n_samples, random_seed=42):
    """Select samples randomly (passive learning)"""
    return X_unlabeled.sample(n_samples, random_state=random_seed)


def stratified_random_sampling(X_unlabeled, y_unlabeled, n_samples, target_fraud_ratio=0.10, random_seed=42):
    """
    Select samples using stratified random sampling to maintain fraud representation
    
    This ensures fair passive learning comparison by maintaining class balance.
    """
    print(f"  🎯 Using STRATIFIED random sampling (maintaining {target_fraud_ratio*100:.1f}% fraud ratio)")
    
    fraud_indices = X_unlabeled[y_unlabeled == 1].index.tolist()
    non_fraud_indices = X_unlabeled[y_unlabeled == 0].index.tolist()
    
    # Calculate how many fraud samples we want
    target_fraud_samples = max(1, int(n_samples * target_fraud_ratio))
    target_fraud_samples = min(target_fraud_samples, len(fraud_indices))  # Can't exceed available
    remaining_samples = n_samples - target_fraud_samples
    
    rng = np.random.RandomState(random_seed)
    
    # Sample fraud and non-fraud separately
    if len(fraud_indices) > 0:
        selected_fraud = rng.choice(fraud_indices, size=target_fraud_samples, replace=False)
    else:
        selected_fraud = []
    
    selected_non_fraud = rng.choice(non_fraud_indices, size=remaining_samples, replace=False)
    
    # Combine and shuffle
    all_selected = list(selected_fraud) + list(selected_non_fraud)
    rng.shuffle(all_selected)
    
    actual_fraud_ratio = len(selected_fraud) / n_samples * 100
    print(f"    ✓ Selected {len(selected_fraud)} fraud + {len(selected_non_fraud)} non-fraud")
    print(f"    ✓ Actual fraud ratio: {actual_fraud_ratio:.2f}%")
    
    return X_unlabeled.loc[all_selected]


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train model and evaluate on validation set"""
    
    # Use global MODEL_TYPE from main function
    global MODEL_TYPE
    if 'MODEL_TYPE' not in globals():
        MODEL_TYPE = 'logistic'  # Default fallback
    
    if MODEL_TYPE == 'logistic':
        # Logistic Regression optimized for fraud detection
        model = LogisticRegression(
            C=0.1,                    # Regularization to prevent overfitting
            random_state=42, 
            max_iter=1000,            # More iterations for convergence
            solver='liblinear',       # Good for smaller datasets
            class_weight='balanced'   # Critical for imbalanced fraud data
        )
    elif MODEL_TYPE == 'naive_bayes':
        # Naive Bayes - robust baseline
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif MODEL_TYPE == 'random_forest':
        # Random Forest for fraud detection
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        # Fallback to Logistic Regression
        print("  ⚠️  Unknown model type, falling back to Logistic Regression")
        model = LogisticRegression(
            C=0.1, random_state=42, max_iter=1000,
            solver='liblinear', class_weight='balanced'
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    
    # Use zero_division=0 to handle cases where precision/recall can't be computed
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0)
    }
    
    return model, metrics


def run_active_learning_experiment(X_train, y_train, X_test, y_test, 
                                 initial_samples=300, batch_size=68, n_iterations=11, 
                                 strategies=None, random_seed=42, initial_strategy='stratified'):
    """Run active learning experiment for fraud detection"""
    print(f"\n{'='*60}")
    print("ACTIVE LEARNING EXPERIMENT - FRAUD DETECTION")
    print(f"{'='*60}")
    
    # Create validation set from training data (stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    
    X_train_val = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_train_val = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    # Start with stratified initial labeled pool to ensure fraud representation
    if initial_strategy == 'stratified':
        initial_indices = stratified_initial_split(X_train_val, y_train_val, initial_samples, random_seed)
        X_labeled = X_train_val.loc[initial_indices]
        y_labeled = y_train_val.loc[initial_indices]
    elif initial_strategy == 'random':
        rng = np.random.RandomState(random_seed)
        initial_indices = rng.choice(X_train_val.index, size=initial_samples, replace=False)
        X_labeled = X_train_val.loc[initial_indices]
        y_labeled = y_train_val.loc[initial_indices]
    elif initial_strategy == 'diversity':
        # Use diversity sampling for initial pool
        print(f"  Using {initial_strategy} sampling for initial pool...")
        X_labeled = diversity_sampling(X_train_val, initial_samples)
        y_labeled = y_train_val.loc[X_labeled.index]
    else:
        raise ValueError(f"Invalid initial strategy: {initial_strategy}")
    
    # Remaining unlabeled data
    X_unlabeled = X_train_val.drop(index=X_labeled.index)
    y_unlabeled = y_train_val.drop(index=X_labeled.index)
    
    print(f"Initial labeled pool: {len(X_labeled)} samples")
    print(f"Initial fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
    print(f"Remaining unlabeled: {len(X_unlabeled)} samples")
    
    results = []
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Train model on current labeled data
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Store results
        results.append({
            'iteration': iteration,
            'labeled_samples': len(X_labeled),
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        # Select new samples for next iteration
        if iteration < n_iterations and len(X_unlabeled) > 0:
            # Use the strategy specified in the configuration
            if strategies and iteration <= len(strategies):
                strategy = strategies[iteration - 1]  # -1 because iteration starts at 1
            else:
                # Fallback to cycling through strategies if not enough specified
                strategy = ['uncertainty', 'diversity', 'qbc'][(iteration - 1) % 3]
            
            # Apply the selected strategy
            if strategy == 'uncertainty':
                new_samples = uncertainty_sampling(model, X_unlabeled, batch_size)
            elif strategy == 'diversity':
                new_samples = diversity_sampling(X_unlabeled, batch_size)
            elif strategy == 'qbc':
                new_samples = qbc_sampling(X_unlabeled, batch_size, X_labeled, y_labeled)
            elif strategy == 'random':
                new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
            else:
                print(f"Warning: Unknown strategy '{strategy}', using random sampling")
                new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
            
            print(f"Selected {len(new_samples)} samples using {strategy} sampling")
            
            # Add new samples to labeled pool
            new_labels = y_train_val.loc[new_samples.index]
            X_labeled = pd.concat([X_labeled, new_samples])
            y_labeled = pd.concat([y_labeled, new_labels])
            
            # Remove from unlabeled pool
            X_unlabeled = X_unlabeled.drop(index=new_samples.index)
            y_unlabeled = y_unlabeled.drop(index=new_samples.index)
            
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    
    return results, final_metrics


def run_passive_learning_experiment(X_train, y_train, X_test, y_test,
                                  initial_samples=300, batch_size=68, n_iterations=11, random_seed=42,
                                  use_stratified_passive=True):
    """
    Run passive learning experiment with stratified sampling for fraud detection
    
    Args:
        use_stratified_passive: If True, use stratified random sampling to maintain fraud ratio
    """
    print(f"\n{'='*60}")
    print("PASSIVE LEARNING EXPERIMENT - FRAUD DETECTION")
    print(f"{'='*60}")
    
    # Create validation set from training data (stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    
    X_train_val = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_train_val = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    # Start with stratified initial labeled pool to ensure fraud representation
    initial_indices = stratified_initial_split(X_train_val, y_train_val, initial_samples, random_seed)
    X_labeled = X_train_val.loc[initial_indices]
    y_labeled = y_train_val.loc[initial_indices]
    
    # Remaining unlabeled data
    X_unlabeled = X_train_val.drop(index=initial_indices)
    y_unlabeled = y_train_val.drop(index=initial_indices)
    
    print(f"Initial labeled pool: {len(X_labeled)} samples")
    print(f"Initial fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
    print(f"Remaining unlabeled: {len(X_unlabeled)} samples")
    print(f"🎯 Using {'stratified' if use_stratified_passive else 'random'} passive sampling")
    
    results = []
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Train model on current labeled data
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Store results
        results.append({
            'iteration': iteration,
            'labeled_samples': len(X_labeled),
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        # Select new samples for next iteration
        if iteration < n_iterations and len(X_unlabeled) > 0:
            if use_stratified_passive:
                # Use stratified random sampling to maintain fraud representation
                new_samples = stratified_random_sampling(X_unlabeled, y_unlabeled, batch_size, 
                                                       target_fraud_ratio=0.10, random_seed=random_seed)
            else:
                # Pure random sampling baseline
                new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
                print(f"Selected {len(new_samples)} samples using pure random sampling")
            
            # Add new samples to labeled pool
            new_labels = y_train_val.loc[new_samples.index]
            X_labeled = pd.concat([X_labeled, new_samples])
            y_labeled = pd.concat([y_labeled, new_labels])
            
            # Remove from unlabeled pool
            X_unlabeled = X_unlabeled.drop(index=new_samples.index)
            y_unlabeled = y_unlabeled.drop(index=new_samples.index)
            
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    
    return results, final_metrics


# Rest of the functions (plot_comparison, run_multiple_experiments, etc.) remain the same...
# [Previous plotting and analysis functions would be included here]


def main():
    """Main function to run the fraud detection active learning comparison"""
    
    # ===== MODEL CONFIGURATION =====
    MODEL_TYPE = 'logistic'  # Options: 'logistic', 'naive_bayes', 'random_forest'
    
    # ===== PLOT CONFIGURATION =====
    SHOW_PLOTS = False  # Options: True (show plots), False (save only, no display)
    
    # ===== EXPERIMENT CONFIGURATION =====
    # Clean configuration based on best practices from Bank experiments
    EXPERIMENT_CONFIG = {
        'initial_samples': 300,        # Initial pool size
        'initial_strategy': 'stratified',  # Use stratified to ensure fraud representation
        'batch_size': 68,             # Samples per iteration (same as Bank experiments)
        'n_iterations': 11,           # Total iterations (same as Bank experiments)
        'iteration_strategies': [
            'uncertainty',      # Iteration 1
            'uncertainty',      # Iteration 2
            'uncertainty',      # Iteration 3
            'uncertainty',      # Iteration 4
            'diversity',        # Iteration 5
            'diversity',        # Iteration 6
            'uncertainty',      # Iteration 7
            'diversity',        # Iteration 8
            'uncertainty',      # Iteration 9
            'uncertainty',      # Iteration 10
            'qbc',             # Iteration 11
        ]
    }
    CONFIG_NAME = "fraud_config_01"
    
    # Setup logging to save all output to file
    logger = setup_logging(CONFIG_NAME)
    
    print("💳 Credit Card Fraud Detection - Active Learning vs Passive Learning Comparison")
    print("="*80)
    print("Dataset: Kaggle European Credit Card Fraud Dataset")
    print("Target: Fraud detection (Class: 0=non-fraud, 1=fraud)")
    print("Features: Time, V1-V28 (anonymized), Amount")
    print("="*80)
    print("🎯 Implementation: Stratified passive learning maintains fraud representation")
    print("="*80)
    
    print("Model Configuration:")
    print(f"  Model Type: {MODEL_TYPE}")
    print()
    print("Plot Configuration:")
    print(f"  Show Plots: {SHOW_PLOTS}")
    print()
    print("Experiment Configuration:")
    print(f"  Initial samples: {EXPERIMENT_CONFIG['initial_samples']}")
    print(f"  Initial strategy: {EXPERIMENT_CONFIG.get('initial_strategy', 'stratified')}")
    print(f"  Batch size: {EXPERIMENT_CONFIG['batch_size']}")
    print(f"  Iterations: {EXPERIMENT_CONFIG['n_iterations']}")
    print(f"  Strategies: {EXPERIMENT_CONFIG['iteration_strategies']}")
    print()
    
    # Load and split fraud data
    fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
    
    # Check if fraud dataset exists
    if not os.path.exists(fraud_data_path):
        print(f"❌ Fraud dataset not found at: {fraud_data_path}")
        print("Please ensure the creditcard.csv file is in the correct location.")
        return
    
    print(f"✅ Loading fraud dataset from: {fraud_data_path}")
    
    X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
    
    # Quick test with single run to verify fix
    print("\n🧪 RUNNING STRATIFIED PASSIVE LEARNING EXPERIMENT")
    
    # Run single active learning experiment
    active_results, active_final = run_active_learning_experiment(
        X_train, y_train, X_test, y_test,
        initial_samples=EXPERIMENT_CONFIG['initial_samples'],
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        n_iterations=EXPERIMENT_CONFIG['n_iterations'],
        strategies=EXPERIMENT_CONFIG['iteration_strategies'],
        initial_strategy=EXPERIMENT_CONFIG.get('initial_strategy', 'stratified')
    )
    
    # Run stratified passive learning experiment
    passive_results, passive_final = run_passive_learning_experiment(
        X_train, y_train, X_test, y_test,
        initial_samples=EXPERIMENT_CONFIG['initial_samples'],
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        n_iterations=EXPERIMENT_CONFIG['n_iterations'],
        use_stratified_passive=True  # Use stratified passive sampling
    )
    
    # Compare results
    print(f"\n{'='*80}")
    print("EXPERIMENTAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    print(f"Active Learning F1: {active_final['f1']:.4f}")
    print(f"Passive Learning F1: {passive_final['f1']:.4f}")
    print(f"F1 Improvement: {active_final['f1'] - passive_final['f1']:.4f}")
    print(f"Improvement %: {((active_final['f1'] - passive_final['f1']) / passive_final['f1'] * 100):.2f}%")
    
    # Close logger and restore stdout
    print(f"\n📝 Experiment completed - Results demonstrate fair comparison methodology!")
    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        # Ensure stdout is restored even if there's an error
        if 'logger' in locals():
            logger.close()
            sys.stdout = logger.terminal
        raise
