#!/usr/bin/env python3
"""
Simple Active Learning Script with LightGBM for Credit Card Fraud Detection

This script implements active learning for the Kaggle European Credit Card Fraud dataset
using LightGBM as the primary model, with fallback to Logistic Regression.

Key Features:
- Uses LightGBM optimized for fraud detection (better for structured data)
- Optimized for highly imbalanced fraud detection (Class: 0=non-fraud, 1=fraud)
- Uses stratified sampling to maintain fraud representation in initial seed set
- Implements the same AL strategies (uncertainty, diversity, QBC) 
- Same evaluation metrics and statistical testing as Bank experiments
- Enhanced hyperparameters for fraud detection on small labeled sets
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
    
    # 7. Feature standardization (important for some models, LightGBM is robust to scaling)
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
    """Select samples using Query by Committee (QBC) with LightGBM in committee"""
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
    models = {}
    
    # Try to use LightGBM as the primary model in the committee
    try:
        import lightgbm as lgb
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=50,           # Fewer trees for committee diversity
            max_depth=6,               # Moderate depth for fraud detection
            learning_rate=0.15,        # Faster learning for committee
            subsample=0.7,             # More aggressive subsampling for diversity
            colsample_bytree=0.7,      # Feature sampling for diversity
            min_child_samples=15,      # Small minimum samples per leaf
            scale_pos_weight=10,       # Handle class imbalance (approximate ratio)
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=-1
        )
        print("    🚀 LightGBM added to QBC committee")
    except ImportError:
        print("    ⚠️  LightGBM not available for QBC committee")
    
    # Add other models for diversity
    models['lr'] = LogisticRegression(
        C=0.1,                    # Regularization for generalization
        random_state=42, 
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'   # Handle class imbalance
    )
    
    models['rf'] = RandomForestClassifier(
        n_estimators=50,          # Moderate number of trees
        max_depth=8,              # Reasonable depth for fraud data
        min_samples_split=20,     # Prevent overfitting
        min_samples_leaf=10,      # Ensure leaf purity
        random_state=42,
        class_weight='balanced'   # Handle class imbalance
    )
    
    models['et'] = ExtraTreesClassifier(
        n_estimators=50,          # Moderate number of trees
        max_depth=8,              # Reasonable depth
        min_samples_split=20,     # Prevent overfitting
        min_samples_leaf=10,      # Ensure leaf purity
        random_state=42,
        class_weight='balanced'   # Handle class imbalance
    )
    
    models['nb'] = GaussianNB()   # Naive Bayes - good baseline
    
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


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train model and evaluate on validation set"""
    
    # Use global MODEL_TYPE from main function
    global MODEL_TYPE
    if 'MODEL_TYPE' not in globals():
        MODEL_TYPE = 'lightgbm'  # Default to LightGBM
    
    if MODEL_TYPE == 'lightgbm':
        try:
            import lightgbm as lgb
            # LightGBM optimized for fraud detection with small labeled datasets
            model = lgb.LGBMClassifier(
                n_estimators=100,           # Moderate number of trees
                max_depth=8,                # Good depth for fraud patterns
                learning_rate=0.1,          # Moderate learning rate
                subsample=0.8,              # Slight subsampling to prevent overfitting
                colsample_bytree=0.8,       # Feature subsampling
                min_child_samples=20,       # Require more samples per leaf
                min_split_gain=0.01,        # Minimum gain for split
                reg_alpha=0.01,             # L1 regularization
                reg_lambda=0.01,            # L2 regularization
                scale_pos_weight=10,        # Handle class imbalance (approximate fraud ratio)
                random_state=42,
                class_weight='balanced',    # Additional class balancing
                verbose=-1,                 # Suppress verbose output
                n_jobs=-1                   # Use all CPU cores
            )
            print("  🚀 Using LightGBM model")
        except ImportError:
            print("  ⚠️  LightGBM not available, falling back to Logistic Regression")
            MODEL_TYPE = 'logistic'
            model = LogisticRegression(
                C=0.1, random_state=42, max_iter=1000,
                solver='liblinear', class_weight='balanced'
            )
    
    elif MODEL_TYPE == 'logistic':
        # Logistic Regression optimized for fraud detection
        model = LogisticRegression(
            C=0.1,                    # Regularization to prevent overfitting
            random_state=42, 
            max_iter=1000,            # More iterations for convergence
            solver='liblinear',       # Good for smaller datasets
            class_weight='balanced'   # Critical for imbalanced fraud data
        )
        print("  📊 Using Logistic Regression model")
    
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
        print("  🌲 Using Random Forest model")
    
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
            
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    
    return results, final_metrics


def run_passive_learning_experiment(X_train, y_train, X_test, y_test,
                                  initial_samples=300, batch_size=68, n_iterations=11, random_seed=42):
    """Run passive learning experiment (random sampling) for fraud detection"""
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
        
        # Select new samples randomly for next iteration
        if iteration < n_iterations and len(X_unlabeled) > 0:
            new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
            print(f"Selected {len(new_samples)} samples using random sampling")
            
            # Add new samples to labeled pool
            new_labels = y_train_val.loc[new_samples.index]
            X_labeled = pd.concat([X_labeled, new_samples])
            y_labeled = pd.concat([y_labeled, new_labels])
            
            # Remove from unlabeled pool
            X_unlabeled = X_unlabeled.drop(index=new_samples.index)
            
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Fraud in labeled: {(y_labeled == 1).sum()} samples ({(y_labeled == 1).sum()/len(y_labeled)*100:.2f}%)")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    
    return results, final_metrics


def plot_comparison(active_results, passive_results, config_name="", show_plots=True):
    """Plot comparison between active and passive learning"""
    active_df = pd.DataFrame(active_results)
    passive_df = pd.DataFrame(passive_results)
    
    plt.figure(figsize=(15, 5))
    
    # Plot F1 score progression
    plt.subplot(1, 3, 1)
    plt.plot(active_df['iteration'], active_df['f1'], 'o-', label='Active Learning', linewidth=2, markersize=8)
    plt.plot(passive_df['iteration'], passive_df['f1'], 's-', label='Passive Learning', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy progression
    plt.subplot(1, 3, 2)
    plt.plot(active_df['iteration'], active_df['accuracy'], 'o-', label='Active Learning', linewidth=2, markersize=8)
    plt.plot(passive_df['iteration'], passive_df['accuracy'], 's-', label='Passive Learning', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot labeled samples progression
    plt.subplot(1, 3, 3)
    plt.plot(active_df['iteration'], active_df['labeled_samples'], 'o-', label='Active Learning', linewidth=2, markersize=8)
    plt.plot(passive_df['iteration'], passive_df['labeled_samples'], 's-', label='Passive Learning', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Labeled Samples')
    plt.title('Labeled Samples Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with config name if provided
    if config_name:
        plot_filename = f'active_vs_passive_comparison_{config_name}.png'
    else:
        plot_filename = 'active_vs_passive_comparison.png'
    
    plt.savefig(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{plot_filename}', dpi=300, bbox_inches='tight')
    
    # Display plot based on show_plots parameter
    if show_plots:
        # Use non-blocking display - plots will appear but won't pause execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to allow plot to render
    else:
        # Close the plot to free memory
        plt.close()


def run_multiple_experiments(X_train, X_test, y_train, y_test, config, n_runs=10):
    """Run multiple experiments with different random seeds for statistical significance"""
    print(f"\n{'='*80}")
    print(f"RUNNING {n_runs} EXPERIMENTS FOR STATISTICAL SIGNIFICANCE")
    print(f"{'='*80}")
    
    all_active_results = []
    all_passive_results = []
    all_active_finals = []
    all_passive_finals = []
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Use different random seed for each run
        random_seed = 42 + run
        
        # Run active learning experiment
        active_results, active_final = run_active_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=config['initial_samples'],
            batch_size=config['batch_size'],
            n_iterations=config['n_iterations'],
            strategies=config['iteration_strategies'],
            random_seed=random_seed,
            initial_strategy=config.get('initial_strategy', 'stratified')
        )
        
        # Run passive learning experiment
        passive_results, passive_final = run_passive_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=config['initial_samples'],
            batch_size=config['batch_size'],
            n_iterations=config['n_iterations'],
            random_seed=random_seed
        )
        
        # Store results
        all_active_results.append(active_results)
        all_passive_results.append(passive_results)
        all_active_finals.append(active_final)
        all_passive_finals.append(passive_final)
        
        print(f"Run {run + 1} completed - Active F1: {active_final['f1']:.4f}, Passive F1: {passive_final['f1']:.4f}")
    
    return all_active_results, all_passive_results, all_active_finals, all_passive_finals


def perform_statistical_tests(active_finals, passive_finals):
    """Perform statistical significance tests"""
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    
    # Extract F1 scores for testing
    active_f1_scores = [result['f1'] for result in active_finals]
    passive_f1_scores = [result['f1'] for result in passive_finals]
    
    # Calculate basic statistics
    active_mean = np.mean(active_f1_scores)
    active_std = np.std(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    passive_std = np.std(passive_f1_scores)
    
    print(f"Active Learning F1: {active_mean:.4f} ± {active_std:.4f}")
    print(f"Passive Learning F1: {passive_mean:.4f} ± {passive_std:.4f}")
    print(f"Difference: {active_mean - passive_mean:.4f}")
    
    # Paired t-test (since we're comparing the same data splits)
    t_stat, p_value = stats.ttest_rel(active_f1_scores, passive_f1_scores)
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_p_value = stats.wilcoxon(active_f1_scores, passive_f1_scores)
    print(f"\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat:.4f}")
    print(f"  p-value: {w_p_value:.6f}")
    print(f"  Significant (α=0.05): {'Yes' if w_p_value < 0.05 else 'No'}")
    
    # Effect size (Cohen's d)
    cohens_d = (active_mean - passive_mean) / np.sqrt((active_std**2 + passive_std**2) / 2)
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"Effect size interpretation: {effect_interpretation}")
    
    # Confidence intervals (95%)
    active_ci = stats.t.interval(0.95, len(active_f1_scores)-1, active_mean, active_std/np.sqrt(len(active_f1_scores)))
    passive_ci = stats.t.interval(0.95, len(passive_f1_scores)-1, passive_mean, passive_std/np.sqrt(len(passive_f1_scores)))
    
    print(f"\n95% Confidence Intervals:")
    print(f"  Active Learning: [{active_ci[0]:.4f}, {active_ci[1]:.4f}]")
    print(f"  Passive Learning: [{passive_ci[0]:.4f}, {passive_ci[1]:.4f}]")
    
    return {
        'active_mean': active_mean,
        'active_std': active_std,
        'passive_mean': passive_mean,
        'passive_std': passive_std,
        'difference': active_mean - passive_mean,
        't_stat': t_stat,
        'p_value': p_value,
        'w_stat': w_stat,
        'w_p_value': w_p_value,
        'cohens_d': cohens_d,
        'effect_interpretation': effect_interpretation,
        'active_ci': active_ci,
        'passive_ci': passive_ci
    }


def plot_statistical_comparison(active_finals, passive_finals, config_name="", show_plots=True):
    """Plot statistical comparison with distributions and confidence intervals"""
    active_f1_scores = [result['f1'] for result in active_finals]
    passive_f1_scores = [result['f1'] for result in passive_finals]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution comparison
    plt.subplot(2, 3, 1)
    plt.hist(active_f1_scores, alpha=0.7, label='Active Learning', bins=10, color='blue')
    plt.hist(passive_f1_scores, alpha=0.7, label='Passive Learning', bins=10, color='red')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of F1 Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    plt.subplot(2, 3, 2)
    plt.boxplot([active_f1_scores, passive_f1_scores], labels=['Active', 'Passive'])
    plt.ylabel('F1 Score')
    plt.title('Box Plot Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Individual run comparison
    plt.subplot(2, 3, 3)
    runs = range(1, len(active_f1_scores) + 1)
    plt.plot(runs, active_f1_scores, 'o-', label='Active Learning', linewidth=2, markersize=6)
    plt.plot(runs, passive_f1_scores, 's-', label='Passive Learning', linewidth=2, markersize=6)
    plt.xlabel('Run')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Run')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Difference distribution
    plt.subplot(2, 3, 4)
    differences = [a - p for a, p in zip(active_f1_scores, passive_f1_scores)]
    plt.hist(differences, alpha=0.7, bins=10, color='green')
    plt.axvline(0, color='red', linestyle='--', label='No difference')
    plt.xlabel('F1 Score Difference (Active - Passive)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Confidence intervals
    plt.subplot(2, 3, 5)
    active_mean = np.mean(active_f1_scores)
    active_std = np.std(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    passive_std = np.std(passive_f1_scores)
    
    plt.errorbar(['Active', 'Passive'], [active_mean, passive_mean], 
                yerr=[active_std, passive_std], fmt='o', capsize=5, markersize=8)
    plt.ylabel('F1 Score')
    plt.title('Means with Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Effect size visualization
    plt.subplot(2, 3, 6)
    cohens_d = (active_mean - passive_mean) / np.sqrt((active_std**2 + passive_std**2) / 2)
    plt.bar(['Effect Size'], [cohens_d], color='orange', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel("Cohen's d")
    plt.title(f'Effect Size: {cohens_d:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with config name if provided
    if config_name:
        plot_filename = f'statistical_comparison_{config_name}.png'
    else:
        plot_filename = 'statistical_comparison.png'
    
    plt.savefig(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{plot_filename}', dpi=300, bbox_inches='tight')
    
    # Display plot based on show_plots parameter
    if show_plots:
        # Use non-blocking display - plots will appear but won't pause execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to allow plot to render
    else:
        # Close the plot to free memory
        plt.close()


def main():
    """Main function to run the fraud detection active learning comparison with LightGBM"""
    
    # ===== MODEL CONFIGURATION =====
    MODEL_TYPE = 'lightgbm'  # Options: 'lightgbm', 'logistic', 'random_forest'
    
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
    CONFIG_NAME = "fraud_lgbm_config_01"
    
    # Setup logging to save all output to file
    logger = setup_logging(CONFIG_NAME)
    
    print("💳 Credit Card Fraud Detection - Active Learning vs Passive Learning Comparison")
    print("="*80)
    print("Dataset: Kaggle European Credit Card Fraud Dataset")
    print("Target: Fraud detection (Class: 0=non-fraud, 1=fraud)")
    print("Features: Time, V1-V28 (anonymized), Amount")
    print("="*80)
    print("🚀 ENHANCEMENT: Using LightGBM as primary model (optimized for fraud detection)")
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
    
    # ===== STATISTICAL TESTING =====
    STATISTICAL_TESTING = True
    N_RUNS = 10  # Number of runs for statistical testing
    
    if STATISTICAL_TESTING:
        # Run multiple experiments for statistical significance
        all_active_results, all_passive_results, all_active_finals, all_passive_finals = run_multiple_experiments(
            X_train, X_test, y_train, y_test, EXPERIMENT_CONFIG, n_runs=N_RUNS
        )
        
        # Perform statistical tests
        stats_results = perform_statistical_tests(all_active_finals, all_passive_finals)
        
        # Plot statistical comparison
        plot_statistical_comparison(all_active_finals, all_passive_finals, CONFIG_NAME, show_plots=SHOW_PLOTS)
        
        # Use the first run for detailed iteration analysis
        active_results = all_active_results[0]
        passive_results = all_passive_results[0]
        
        # Save statistical results
        stats_filename = f'statistical_results_{CONFIG_NAME}.csv'
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{stats_filename}', index=False)
        print(f"\nStatistical results saved to: {stats_filename}")
        
    else:
        # Single run
        print("\nRunning single experiment...")
        
        # Run active learning experiment
        active_results, active_final = run_active_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=EXPERIMENT_CONFIG['initial_samples'],
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            n_iterations=EXPERIMENT_CONFIG['n_iterations'],
            strategies=EXPERIMENT_CONFIG['iteration_strategies'],
            initial_strategy=EXPERIMENT_CONFIG.get('initial_strategy', 'stratified')
        )
        
        # Run passive learning experiment
        passive_results, passive_final = run_passive_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=EXPERIMENT_CONFIG['initial_samples'],
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            n_iterations=EXPERIMENT_CONFIG['n_iterations']
        )
        
        all_active_finals = [active_final]
        all_passive_finals = [passive_final]
    
    # Compare results
    print(f"\n{'='*80}")
    print("STATISTICAL TEST DATA COMPARISON (Final Test Set Performance)")
    print(f"{'='*80}")
    print("Note: This table shows the final test set F1 scores used in statistical significance testing")
    print("Each row represents one complete experiment run with different random seeds")
    print()
    
    # Create comparison table
    comparison_data = []
    for run in range(len(all_active_finals)):
        active_final = all_active_finals[run]
        passive_final = all_passive_finals[run]
        
        comparison_data.append({
            'Run': run + 1,
            'Random_Seed': 42 + run,
            'Active_F1': active_final['f1'],
            'Passive_F1': passive_final['f1'],
            'Active_Accuracy': active_final['accuracy'],
            'Passive_Accuracy': passive_final['accuracy'],
            'Active_Precision': active_final['precision'],
            'Passive_Precision': passive_final['precision'],
            'Active_Recall': active_final['recall'],
            'Passive_Recall': passive_final['recall'],
            'F1_Improvement': active_final['f1'] - passive_final['f1'],
            'Improvement_%': ((active_final['f1'] - passive_final['f1']) / passive_final['f1'] * 100) if passive_final['f1'] > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Add summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (All 10 Runs)")
    print(f"{'='*80}")
    
    active_f1_scores = [row['Active_F1'] for row in comparison_data]
    passive_f1_scores = [row['Passive_F1'] for row in comparison_data]
    
    active_mean = np.mean(active_f1_scores)
    active_std = np.std(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    passive_std = np.std(passive_f1_scores)
    
    print(f"Active Learning F1: {active_mean:.4f} ± {active_std:.4f}")
    print(f"Passive Learning F1: {passive_mean:.4f} ± {passive_std:.4f}")
    print(f"Mean Improvement: {active_mean - passive_mean:.4f}")
    print(f"Mean Improvement %: {((active_mean - passive_mean) / passive_mean * 100):.2f}%")
    
    # Save results
    results_filename = f'statistical_test_data_{CONFIG_NAME}.csv'
    comparison_df.to_csv(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{results_filename}', index=False)
    
    # Plot comparison
    plot_comparison(active_results, passive_results, CONFIG_NAME, show_plots=SHOW_PLOTS)
    
    print(f"\nStatistical test data saved to: {HOME_DIR}/active-learning/experimentation-fraud/data/{results_filename}")
    print(f"Iteration progression plot saved to: {HOME_DIR}/active-learning/experimentation-fraud/data/active_vs_passive_comparison_{CONFIG_NAME}.png")
    
    # Close logger and restore stdout
    print(f"\n📝 Logging completed - Check the logs folder for detailed output")
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
