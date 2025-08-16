#!/usr/bin/env python3
"""
Simple Active Learning Script

This is a simplified version that's easy to understand and use.
Just run this file and it will compare active vs passive learning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning/active-learning'

def setup_logging(config_name):
    """Setup logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = f'{HOME_DIR}/experimentation/data/logs'
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

def clean_bank_dataset(df, use_numerical_features=False):
    """
    Clean and preprocess bank dataset for active learning
    
    Args:
        df: Raw bank dataset
        use_numerical_features: If True, use numerical age and balance instead of binned versions
    """
    
    print("🧹 Cleaning and preprocessing bank dataset...")
    
    # 1. Handle categorical variables
    df_clean = df.copy()
    
    # Check for missing values first
    print(f"  📊 Original dataset shape: {df_clean.shape}")
    print(f"  🔍 Missing values in original data:")
    missing_counts = df_clean.isnull().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
    else:
        print("    No missing values found")
    
    # Job aggregation
    job_mapping = {
        'management': 'white_collar', 'technician': 'white_collar',
        'entrepreneur': 'white_collar', 'blue-collar': 'blue_collar',
        'services': 'services', 'unemployed': 'unemployed',
        'retired': 'inactive', 'student': 'student',
        'housemaid': 'services', 'self-employed': 'white_collar',
        'unknown': 'unknown'
    }
    df_clean['job_group'] = df_clean['job'].map(job_mapping)
    
    # Age handling - choose between binned and numerical
    if use_numerical_features:
        # Numerical age with standardization
        df_clean['age_norm'] = (df_clean['age'] - df_clean['age'].mean()) / df_clean['age'].std()
        df_clean['age_squared'] = df_clean['age_norm'] ** 2  # For quadratic effects
        print("  ✓ Using numerical age features (normalized + squared)")
    else:
        # Binned age
        df_clean['age_group'] = pd.cut(df_clean['age'], 
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        print("  ✓ Using binned age features (6 age groups)")
    
    # Balance handling - choose between binned and numerical
    if use_numerical_features:
        # Numerical balance with log transformation and standardization
        df_clean['balance_log'] = np.log1p(df_clean['balance'] + 8000)  # Handle negative values
        df_clean['balance_norm'] = (df_clean['balance_log'] - df_clean['balance_log'].mean()) / df_clean['balance_log'].std()
        print("  ✓ Using numerical balance features (log-transformed + normalized)")
    else:
        # Binned balance
        df_clean['balance_log'] = np.log1p(df_clean['balance'] + 8000)
        df_clean['balance_group'] = pd.cut(df_clean['balance_log'],
                                           bins=[0, 3, 4, 5, 6, 7, 12],
                                           labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
        print("  ✓ Using binned balance features (6 balance groups)")
    
    # Cyclical month encoding
    month_to_num = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df_clean['month_num'] = df_clean['month'].map(month_to_num)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month_num'] / 12)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month_num'] / 12)
    
    # 2. Convert to boolean
    df_clean['default_bool'] = (df_clean['default'] == 'yes').astype(int)
    df_clean['housing_bool'] = (df_clean['housing'] == 'yes').astype(int)
    df_clean['loan_bool'] = (df_clean['loan'] == 'yes').astype(int)
    df_clean['y_bool'] = (df_clean['y'] == 'yes').astype(int)
    
    # 3. Select final features based on feature type choice
    if use_numerical_features:
        final_features = [
            'age_norm', 'age_squared', 'job_group', 'marital', 'education', 'default_bool',
            'balance_norm', 'housing_bool', 'loan_bool', 'contact',
            'month_sin', 'month_cos', 'duration', 'campaign', 'poutcome'
        ]
    else:
        final_features = [
            'age_group', 'job_group', 'marital', 'education', 'default_bool',
            'balance_group', 'housing_bool', 'loan_bool', 'contact',
            'month_sin', 'month_cos', 'duration', 'campaign', 'poutcome'
        ]
    
    X = df_clean[final_features]
    y = df_clean['y_bool']
    
    # 4. One-hot encode remaining categorical
    if use_numerical_features:
        # Only encode non-numerical categorical features
        X_encoded = pd.get_dummies(X, columns=['job_group', 'marital', 
                                              'education', 'contact', 'poutcome'])
    else:
        # Encode all categorical features including age and balance groups
        X_encoded = pd.get_dummies(X, columns=['age_group', 'job_group', 'marital', 
                                              'education', 'contact', 'balance_group', 'poutcome'])
    
    # 5. Handle any remaining NaN values
    print(f"  🔍 Checking for NaN values after preprocessing...")
    nan_counts = X_encoded.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"    Found NaN values in columns:")
        print(nan_counts[nan_counts > 0])
        
        # Fill NaN values with appropriate defaults
        print("    🧹 Filling NaN values...")
        
        # Sort columns to ensure consistent order for reproducibility
        sorted_cols = sorted(X_encoded.columns)
        
        for col in sorted_cols:
            if X_encoded[col].isnull().any():
                if X_encoded[col].dtype in ['int64', 'float64']:
                    # For numerical columns, fill with median
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
    
    # 6. Global feature standardization (important for Logistic Regression)
    print(f"  🔧 Applying global feature standardization...")
    from sklearn.preprocessing import StandardScaler
    
    # Separate numerical and categorical features for proper scaling
    numerical_cols = X_encoded.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    
    # Scale numerical features only (categorical are already 0/1)
    if len(numerical_cols) > 0:
        # Ensure reproducible standardization
        scaler = StandardScaler()
        
        # Sort columns to ensure consistent order
        numerical_cols_sorted = sorted(numerical_cols)
        print(f"    🔒 Using sorted column order for reproducibility: {list(numerical_cols_sorted)}")
        
        # Apply standardization to sorted columns
        X_encoded[numerical_cols_sorted] = scaler.fit_transform(X_encoded[numerical_cols_sorted])
        
        print(f"    ✓ Standardized {len(numerical_cols_sorted)} numerical features")
        print(f"    🔒 Standardization parameters:")
        print(f"      Mean values: {dict(zip(numerical_cols_sorted, scaler.mean_))}")
        print(f"      Scale values: {dict(zip(numerical_cols_sorted, scaler.scale_))}")
    
    if len(categorical_cols) > 0:
        print(f"    ✓ Kept {len(categorical_cols)} categorical features unchanged (already scaled 0/1)")
    
    # 7. Final check and info
    print(f"  📊 Final dataset shape: {X_encoded.shape}")
    print(f"  🎯 Target distribution: {y.value_counts().to_dict()}")
    
    return X_encoded, y


def load_and_split_data(data_path, test_size=0.2, random_state=42, use_numerical_features=False):
    """
    Load data and create proper train/test split
    
    Args:
        data_path: Path to the data file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        use_numerical_features: If True, use numerical age and balance instead of binned versions
    """
    print("Loading and splitting data...")
    
    # Set numpy random seed for reproducibility
    np.random.seed(random_state)
    
    # Load data
    data = pd.read_csv(data_path, sep=';')  # Bank dataset uses semicolon separator
    print(f"Dataset shape: {data.shape}")
    
    # Clean and preprocess the bank dataset
    print("Cleaning and preprocessing bank dataset...")
    X, y = clean_bank_dataset(data, use_numerical_features=use_numerical_features)
    
    print(f"Features after preprocessing: {X.shape[1]}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Split into train and test (keep test set untouched)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Label distribution in train: {y_train.value_counts().to_dict()}")
    
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
    
    # Train a committee of models with regularization for small datasets
    models = {
        'lr': LogisticRegression(
            C=0.1,                    # Strong regularization
            random_state=42, 
            max_iter=500,
            solver='liblinear',
            class_weight='balanced'
        ),
        'rf': RandomForestClassifier(
            n_estimators=25,          # Fewer trees to prevent overfitting
            max_depth=5,              # Limit tree depth
            min_samples_split=10,     # Require more samples to split
            min_samples_leaf=5,       # Require more samples in leaves
            random_state=42
        ),
        'et': ExtraTreesClassifier(
            n_estimators=25,          # Fewer trees
            max_depth=5,              # Limit tree depth
            min_samples_split=10,     # Require more samples to split
            min_samples_leaf=5,       # Require more samples in leaves
            random_state=42
        ),
        'nb': GaussianNB()            # Naive Bayes - very robust for small datasets
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
    temp_df['disagreement'] = np.zeros(len(X_unlabeled))
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
    
    # MODEL SELECTION: Choose between different approaches
    # Option 1: Regularized Logistic Regression (current)
    # Option 2: Naive Bayes (very robust for small datasets)
    # Option 3: Ridge Regression (L2 regularization)
    
    # Use global MODEL_TYPE from main function
    global MODEL_TYPE
    if 'MODEL_TYPE' not in globals():
        MODEL_TYPE = 'logistic'  # Default fallback
    
    if MODEL_TYPE == 'logistic':
        # Use regularized Logistic Regression for small datasets
        # C=0.1 means stronger regularization (smaller C = stronger regularization)
        model = LogisticRegression(
            C=0.1,                    # Strong regularization to prevent overfitting
            random_state=42, 
            max_iter=500,             # Reduced iterations
            solver='liblinear',       # Better for small datasets
            class_weight='balanced'   # Handle class imbalance
        )
    elif MODEL_TYPE == 'naive_bayes':
        # Naive Bayes - very robust for small datasets, no overfitting
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif MODEL_TYPE == 'ridge':
        # Ridge Regression with L2 regularization
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(
            alpha=1.0,                # L2 regularization strength
            random_state=42,
            class_weight='balanced'
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }
    
    return model, metrics


def run_active_learning_experiment(X_train, y_train, X_test, y_test, 
                                 initial_samples=300, batch_size=68, n_iterations=5, 
                                 strategies=None, random_seed=42, initial_strategy='random'):
    """Run active learning experiment"""
    print(f"\n{'='*60}")
    print("ACTIVE LEARNING EXPERIMENT")
    print(f"{'='*60}")
    
    # Create validation set from training data
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed
    )
    
    # Start with initial labeled pool using specified strategy
    if initial_strategy == 'random':
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
    print(f"Remaining unlabeled: {len(X_unlabeled)} samples")
    
    results = []
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Train model on current labeled data
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
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
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
    
    return results, final_metrics


def run_passive_learning_experiment(X_train, y_train, X_test, y_test,
                                  initial_samples=300, batch_size=68, n_iterations=5, random_seed=42):
    """Run passive learning experiment (random sampling)"""
    print(f"\n{'='*60}")
    print("PASSIVE LEARNING EXPERIMENT")
    print(f"{'='*60}")
    
    # Create validation set from training data
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed
    )
    
    # Start with initial labeled pool
    rng = np.random.RandomState(random_seed)
    initial_indices = rng.choice(X_train_val.index, size=initial_samples, replace=False)
    X_labeled = X_train_val.loc[initial_indices]
    y_labeled = y_train_val.loc[initial_indices]
    
    # Remaining unlabeled data
    X_unlabeled = X_train_val.drop(index=initial_indices)
    
    print(f"Initial labeled pool: {len(X_labeled)} samples")
    print(f"Remaining unlabeled: {len(X_unlabeled)} samples")
    
    results = []
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Train model on current labeled data
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
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
            print(f"Remaining unlabeled: {len(X_unlabeled)}")
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    final_model, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
    
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
    
    plt.savefig(f'{HOME_DIR}/experimentation/data/{plot_filename}', dpi=300, bbox_inches='tight')
    
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
            initial_strategy=config.get('initial_strategy', 'random')
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
    
    plt.savefig(f'{HOME_DIR}/experimentation/data/{plot_filename}', dpi=300, bbox_inches='tight')
    
    # Display plot based on show_plots parameter
    if show_plots:
        # Use non-blocking display - plots will appear but won't pause execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to allow plot to render
    else:
        # Close the plot to free memory
        plt.close()


def main():
    """Main function to run the comparison"""
    
    # ===== MODEL CONFIGURATION =====
    # Choose your model type for training
    MODEL_TYPE = 'logistic'  # Options: 'logistic', 'naive_bayes', 'ridge'
    
    # ===== PLOT CONFIGURATION =====
    # Set to False to disable plot display (useful for headless runs or batch processing)
    SHOW_PLOTS = False  # Options: True (show plots), False (save only, no display)
    
    # ===== EXPERIMENT CONFIGURATION =====
    # Define your experiment configuration here
    
    # IMPORTANT: Always set CONFIG_NAME when you change configurations!
    # This ensures your results are saved with unique filenames.
    
    # Note: Uncertainty and QBC cannot be used for initial sampling because they require trained models.
    
    # Configuration 2: Start with diversity, then cycle through strategies
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,      # How many samples to start with
    #     'initial_strategy': 'diversity',  # Initial sampling strategy
    #     'batch_size': 68,            # How many samples to add per iteration
    #     'n_iterations': 5,           # Total number of iterations
    #     
    #     # Define sampling strategy for each iteration
    #     # Options: 'random', 'uncertainty', 'diversity', 'qbc'
    #     'iteration_strategies': [
    #         'diversity',      # Iteration 1: Start with diversity sampling
    #         'uncertainty',    # Iteration 2: Use uncertainty sampling
    #         'qbc',           # Iteration 3: Use QBC
    #         'uncertainty',    # Iteration 4: Back to uncertainty
    #         'diversity',      # Iteration 5: Back to diversity
    #     ]
    # }
    
    # Configuration 3: Conservative approach with smaller batches
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 200,
    #     'initial_strategy': 'random',    # Start with random sampling
    #     'batch_size': 50,
    #     'n_iterations': 6,
    #     'iteration_strategies': [
    #         'uncertainty',   # Iteration 1: Uncertainty
    #         'diversity',     # Iteration 2: Diversity
    #         'qbc',           # Iteration 3: QBC
    #         'uncertainty',   # Iteration 4: Uncertainty
    #         'diversity',     # Iteration 5: Diversity
    #         'qbc',           # Iteration 6: QBC
    #     ]
    # }
    # CONFIG_NAME = "config3_conservative_random_start"
    
    # Configuration 4: Aggressive approach with larger batches
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 500,
    #     'initial_strategy': 'diversity', # Start with diversity sampling
    #     'batch_size': 100,
    #     'n_iterations': 4,
    #     'iteration_strategies': [
    #         'qbc',           # Iteration 1: QBC
    #         'uncertainty',   # Iteration 2: Uncertainty
    #         'qbc',           # Iteration 3: QBC
    #         'uncertainty',   # Iteration 4: Uncertainty
    #     ]
    # }
    # CONFIG_NAME = "config4_aggressive_diversity_start"
    
    # Configuration 5: Start with diversity, then cycle through strategies
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,
    #     'initial_strategy': 'random',  # How to select initial samples (random/diversity only)
    #     'batch_size': 68,
    #     'n_iterations': 11,
    #     'iteration_strategies': [
    #         'uncertainty',    # Keep good iterations
    #         'diversity',      # Keep good iterations
    #         'uncertainty',    # Keep good iterations
    #         'diversity',      # ← Changed: Problem iteration 4
    #         'uncertainty',    # Keep iteration 5
    #         'diversity',      # ← Changed: Problem iteration 6
    #         'uncertainty',    # Keep iteration 7
    #         'diversity',      # ← Changed: Problem iteration 8
    #         'diversity',      # Keep good iterations
    #         'diversity',      # Keep good iterations
    #         'qbc',           # Iteration 11
    #     ]
    # }
    
    # Configuration 6: Start with diversity, then cycle through strategies
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,
    #     'initial_strategy': 'random',  # How to select initial samples (random/diversity only)
    #     'batch_size': 68,
    #     'n_iterations': 11,
    #     # Mix strategies within iterations
    #     'iteration_strategies': [
    #         'uncertainty',           # Iteration 1
    #         'diversity',             # Iteration 2
    #         'uncertainty',           # Iteration 3
    #         'uncertainty+diversity', # Iteration 4: Hybrid
    #         'uncertainty',           # Iteration 5
    #         'uncertainty+diversity', # Iteration 6: Hybrid
    #         'uncertainty',           # Iteration 7
    #         'uncertainty+diversity', # Iteration 8: Hybrid
    #         'diversity',             # Iteration 9
    #         'diversity',             # Iteration 10
    #         'qbc',                   # Iteration 11
    #     ]
    # }
    
    # Configuration 7: Start with diversity, then cycle through strategies
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,
    #     'initial_strategy': 'random',  # How to select initial samples (random/diversity only)
    #     'batch_size': 68,
    #     'n_iterations': 11,
    #     'iteration_strategies': [
    #         'uncertainty',    # Keep good iterations
    #         'diversity',      # Keep good iterations
    #         'uncertainty',    # Keep good iterations
    #         'diversity',      # ← Changed: Problem iteration 4
    #         'uncertainty',    # Keep iteration 5
    #         'diversity',      # ← Changed: Problem iteration 6
    #         'uncertainty',    # Keep iteration 7
    #         'diversity',      # ← Changed: Problem iteration 8
    #         'diversity',      # Keep good iterations
    #         'diversity',      # Keep good iterations
    #         'qbc',           # Iteration 11
    #     ]
    # }
    
    # Configuration 8: Bank Dataset - Binned Features (Current)
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # Start with random sampling
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': False,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'diversity',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'qbc',             # Iteration 9
    #         'qbc',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # Start with random sampling
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'uncertainty',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config8_binned_features"  # Change this when switching feature types
    
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # Start with random sampling
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'uncertainty',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config9"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # Start with random sampling
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'uncertainty',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config10"  # Change this when switching feature types
    
    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # Start with random sampling
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config11"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'diversity',  # random or diversity
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config12"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # random or diversity
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'uncertainty',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config13"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # random or diversity
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'diversity',             # Iteration 9
    #         'qbc',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config14"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # random or diversity
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'diversity',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'diversity',             # Iteration 9
    #         'qbc',             # Iteration 10
    #         'diversity',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config15"  # Change this when switching feature types

    # EXPERIMENT_CONFIG = {
    #     'initial_samples': 300,        # Initial pool size
    #     'initial_strategy': 'random',  # random or diversity
    #     'batch_size': 68,             # Samples per iteration
    #     'n_iterations': 11,           # Total iterations
    #     'use_numerical_features': True,  # False = binned age/balance, True = numerical
    #     'iteration_strategies': [
    #         'uncertainty',      # Iteration 1
    #         'uncertainty',      # Iteration 2
    #         'uncertainty',      # Iteration 3
    #         'uncertainty',      # Iteration 4
    #         'diversity',        # Iteration 5
    #         'diversity',        # Iteration 6
    #         'uncertainty',        # Iteration 7
    #         'diversity',        # Iteration 8
    #         'diversity',             # Iteration 9
    #         'uncertainty',             # Iteration 10
    #         'qbc',             # Iteration 11
    #     ]
    # }
    # CONFIG_NAME = "config16"  # Change this when switching feature types

    EXPERIMENT_CONFIG = {
        'initial_samples': 300,
        'initial_strategy': 'random',
        'batch_size': 68,
        'n_iterations': 11,
        'use_numerical_features': True,
        'iteration_strategies': ['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty', 'diversity', 'diversity', 'uncertainty', 'diversity', 'uncertainty', 'uncertainty', 'qbc']
    }
    CONFIG_NAME = "config70"






















    # ===== FEATURE TYPE COMPARISON =====
    # 
    # BINNED FEATURES (use_numerical_features=False):
    # - Age: 6 categorical groups (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
    # - Balance: 6 categorical groups (very_low, low, medium, high, very_high, extreme)
    # - Total features: ~25-30 (more sparse, categorical)
    # - Pros: Better for small datasets, clear uncertainty regions, business interpretable
    # - Cons: Loses fine-grained information, more sparse
    # 
    # NUMERICAL FEATURES (use_numerical_features=True):
    # - Age: Normalized + squared (2 continuous features)
    # - Balance: Log-transformed + normalized (1 continuous feature)
    # - Total features: ~20-25 (less sparse, continuous)
    # - Pros: Preserves fine-grained information, less sparse, captures non-linear effects
    # - Cons: May be unstable with small datasets, harder to interpret
    # ===== END CONFIGURATION =====
    
    # Setup logging to save all output to file
    logger = setup_logging(CONFIG_NAME)
    
    # Choose between bank.csv (4,523 rows) or bank-full.csv (45,211 rows)
    USE_FULL_DATASET = True  # Set to False to use smaller dataset
    
    print("🏦 Bank Marketing Dataset - Active Learning vs Passive Learning Comparison")
    print("="*80)
    print("Dataset: UCI Bank Marketing (Portuguese Bank)")
    print("Target: Whether customer subscribed to term deposit (yes/no)")
    print("Features: Age, job, marital status, education, balance, housing, loan, etc.")
    print("="*80)
    
    # Show dataset size information
    if USE_FULL_DATASET:
        print("📊 Dataset Size: 45,211 rows (Full dataset)")
        print("📊 Expected splits: ~36,169 train, ~9,042 test")
    else:
        print("📊 Dataset Size: 4,523 rows (Smaller dataset)")
        print("📊 Expected splits: ~3,617 train, ~904 test")
    print("="*80)
    
    
    print("Model Configuration:")
    print(f"  Model Type: {MODEL_TYPE}")
    print()
    print("Plot Configuration:")
    print(f"  Show Plots: {SHOW_PLOTS}")
    print()
    print("Experiment Configuration:")
    print(f"  Initial samples: {EXPERIMENT_CONFIG['initial_samples']}")
    print(f"  Initial strategy: {EXPERIMENT_CONFIG.get('initial_strategy', 'random')}")
    print(f"  Batch size: {EXPERIMENT_CONFIG['batch_size']}")
    print(f"  Iterations: {EXPERIMENT_CONFIG['n_iterations']}")
    print(f"  Strategies: {EXPERIMENT_CONFIG['iteration_strategies']}")
    
    # Validate initial strategy
    valid_initial_strategies = ['random', 'diversity']
    initial_strategy = EXPERIMENT_CONFIG.get('initial_strategy', 'random')
    if initial_strategy not in valid_initial_strategies:
        raise ValueError(f"Initial strategy must be one of {valid_initial_strategies}. "
                       f"Uncertainty and QBC require trained models and cannot be used for initial sampling.")
    
    print(f"  ✓ Initial strategy '{initial_strategy}' is valid")
    print()
    
    # Load and split data
    if USE_FULL_DATASET:
        bank_data_path = f'{HOME_DIR}/data/uci_dataset_00222_bank/bank-full.csv'
        print("📊 Using FULL bank dataset (45,211 rows)")
    else:
        bank_data_path = f'{HOME_DIR}/data/uci_dataset_00222_bank/bank.csv'
        print("📊 Using smaller bank dataset (4,523 rows)")
    
    # Check if bank dataset exists
    import os
    if not os.path.exists(bank_data_path):
        print(f"❌ Bank dataset not found at: {bank_data_path}")
        print("Please download the bank dataset to the specified location.")
        print("You can download it from: https://archive.ics.uci.edu/dataset/222/bank+marketing")
        return
    
    print(f"✅ Loading bank dataset from: {bank_data_path}")
    
    # Get feature type choice from configuration
    use_numerical = EXPERIMENT_CONFIG.get('use_numerical_features', False)
    print(f"Feature type: {'Numerical' if use_numerical else 'Binned'} age and balance")
    
    X_train, X_test, y_train, y_test = load_and_split_data(
        bank_data_path, 
        use_numerical_features=use_numerical
    )
    
    # ===== CHOOSE YOUR EVALUATION MODE =====
    # Set to True for statistical significance testing (multiple runs)
    # Set to False for single run (faster, less robust)
    STATISTICAL_TESTING = True
    N_RUNS = 10  # Number of runs for statistical testing
    
    # Initialize variables for both modes
    active_results = None
    passive_results = None
    active_final = None
    passive_final = None
    all_active_results = None
    all_passive_results = None
    all_active_finals = None
    all_passive_finals = None
    
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
        active_final = all_active_finals[0]
        passive_final = all_passive_finals[0]
        
        # Save statistical results
        stats_filename = f'statistical_results_{CONFIG_NAME}.csv'
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv(f'{HOME_DIR}/experimentation/data/{stats_filename}', index=False)
        print(f"\nStatistical results saved to: {stats_filename}")
        
        # Debug: Show what we're actually comparing
        print(f"\n🔍 DEBUG: Data Sources for Tables")
        print(f"Statistical Test Data Table: All 10 runs (Final Test Set Performance)")
        print(f"Iteration Progression Plot: Run 1 (Validation Set Performance)")
        print(f"Summary Section: Run 1 (Final Test Set Performance)")
        print(f"Note: Statistical test data shows all runs, iteration plot shows training progress")
        
    else:
        # Single run (original behavior)
        print("\nRunning single experiment...")
        
        # Run active learning experiment with custom configuration
        active_results, active_final = run_active_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=EXPERIMENT_CONFIG['initial_samples'],
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            n_iterations=EXPERIMENT_CONFIG['n_iterations'],
            strategies=EXPERIMENT_CONFIG['iteration_strategies'],
            initial_strategy=EXPERIMENT_CONFIG.get('initial_strategy', 'random')
        )
        
        # Run passive learning experiment
        passive_results, passive_final = run_passive_learning_experiment(
            X_train, y_train, X_test, y_test,
            initial_samples=EXPERIMENT_CONFIG['initial_samples'],
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            n_iterations=EXPERIMENT_CONFIG['n_iterations']
        )
    
    # Compare results - Show statistical test data instead of validation performance
    print(f"\n{'='*80}")
    print("STATISTICAL TEST DATA COMPARISON (Final Test Set Performance)")
    print(f"{'='*80}")
    print("Note: This table shows the final test set F1 scores used in statistical significance testing")
    print("Each row represents one complete experiment run with different random seeds")
    print()
    
    # Create comparison table using the statistical test data
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
            'F1_Improvement': active_final['f1'] - passive_final['f1'],
            'Improvement_%': ((active_final['f1'] - passive_final['f1']) / passive_final['f1'] * 100) if passive_final['f1'] > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Add summary statistics row
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
    
    # Save results with config name
    results_filename = f'statistical_test_data_{CONFIG_NAME}.csv'
    comparison_df.to_csv(f'{HOME_DIR}/experimentation/data/{results_filename}', index=False)
    
    # Plot comparison with config name (still use Run 1 for iteration progression)
    plot_comparison(active_results, passive_results, CONFIG_NAME, show_plots=SHOW_PLOTS)
    
    print(f"\nStatistical test data saved to: {HOME_DIR}/experimentation/data/{results_filename}")
    print(f"Iteration progression plot saved to: {HOME_DIR}/experimentation/data/active_vs_passive_comparison_{CONFIG_NAME}.png")
    
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