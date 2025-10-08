#!/usr/bin/env python3
"""
Comprehensive Iteration Analysis: Full 10 Runs × 11 Iterations

This script runs the COMPLETE experiment structure:
- 10 runs with different random seeds
- 11 iterations per run 
- Shows both iteration-by-iteration volatility AND final aggregated results
- Reveals whether massive improvements are real or just unstable foundations
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
import copy
warnings.filterwarnings('ignore')

HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning'

def clean_fraud_dataset(df):
    """Clean and preprocess credit card fraud dataset for active learning"""
    
    print("🧹 Cleaning and preprocessing credit card fraud dataset...")
    
    df_clean = df.copy()
    
    # Handle target variable
    y = df_clean['Class']
    print(f"  🎯 Target distribution: {y.value_counts().to_dict()}")
    fraud_percentage = (y == 1).sum() / len(y) * 100
    print(f"  ⚠️  Fraud percentage: {fraud_percentage:.3f}%")
    
    # Feature engineering for fraud detection
    df_clean['Time_hour'] = (df_clean['Time'] / 3600) % 24
    df_clean['Time_hour_sin'] = np.sin(2 * np.pi * df_clean['Time_hour'] / 24)
    df_clean['Time_hour_cos'] = np.cos(2 * np.pi * df_clean['Time_hour'] / 24)
    
    df_clean['Amount_log'] = np.log1p(df_clean['Amount'])
    
    df_clean['Amount_bin'] = pd.cut(df_clean['Amount'], 
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['very_small', 'small', 'medium', 'large', 'very_large', 'extreme'])
    
    # Select features for modeling
    feature_columns = (
        [f'V{i}' for i in range(1, 29)] +
        ['Time_hour_sin', 'Time_hour_cos'] +
        ['Amount_log'] +
        ['Amount_bin']
    )
    
    X = df_clean[feature_columns]
    
    # Handle categorical features
    X_encoded = pd.get_dummies(X, columns=['Amount_bin'], prefix='Amount')
    
    # Feature standardization
    scaler = StandardScaler()
    numerical_cols = X_encoded.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numerical_cols) > 0:
        numerical_cols_sorted = sorted(numerical_cols)
        X_encoded[numerical_cols_sorted] = scaler.fit_transform(X_encoded[numerical_cols_sorted])
    
    print(f"  📊 Final dataset shape: {X_encoded.shape}")
    
    return X_encoded, y


def load_and_split_data(data_path, test_size=0.2, random_state=42):
    """Load credit card fraud data and create proper train/test split"""
    print("Loading and splitting credit card fraud data...")
    
    np.random.seed(random_state)
    
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    
    X, y = clean_fraud_dataset(data)
    
    # Stratified split to maintain fraud ratio in both train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def stratified_initial_split(X, y, initial_samples, random_state=42, min_fraud_samples=10):
    """Create stratified initial split ensuring minimum fraud representation"""
    
    fraud_indices = X[y == 1].index.tolist()
    non_fraud_indices = X[y == 0].index.tolist()
    
    actual_fraud_samples = min(min_fraud_samples, len(fraud_indices))
    remaining_samples = initial_samples - actual_fraud_samples
    
    rng = np.random.RandomState(random_state)
    
    selected_fraud = rng.choice(fraud_indices, size=actual_fraud_samples, replace=False)
    selected_non_fraud = rng.choice(non_fraud_indices, size=remaining_samples, replace=False)
    
    initial_indices = np.concatenate([selected_fraud, selected_non_fraud])
    rng.shuffle(initial_indices)
    
    return initial_indices


def uncertainty_sampling(model, X_unlabeled, n_samples, threshold=0.5, window=0.05):
    """Select samples using uncertainty sampling"""
    probabilities = model.predict_proba(X_unlabeled)[:, 1]
    
    lower = threshold - window
    upper = threshold + window
    uncertain_mask = (probabilities > lower) & (probabilities <= upper)
    
    uncertain_samples = X_unlabeled[uncertain_mask]
    
    if len(uncertain_samples) >= n_samples:
        return uncertain_samples.sample(n_samples, random_state=42)
    else:
        remaining = n_samples - len(uncertain_samples)
        other_samples = X_unlabeled[~uncertain_mask].sample(remaining, random_state=42)
        return pd.concat([uncertain_samples, other_samples])


def diversity_sampling(X_unlabeled, n_samples, k=10):
    """Select samples using diversity sampling (KNN-based)"""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X_unlabeled)
    
    distances, _ = knn.kneighbors(X_unlabeled)
    density_scores = distances.mean(axis=1)
    
    temp_df = X_unlabeled.copy()
    temp_df['density'] = density_scores
    
    selected = temp_df.nlargest(n_samples, 'density')
    
    return selected.drop(columns=['density'])


def qbc_sampling(X_unlabeled, n_samples, X_labeled, y_labeled):
    """Select samples using Query by Committee (QBC)"""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    
    models = {
        'lr': LogisticRegression(C=0.1, random_state=42, max_iter=1000, solver='liblinear', class_weight='balanced'),
        'rf': RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42, class_weight='balanced'),
        'et': ExtraTreesClassifier(n_estimators=50, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42, class_weight='balanced'),
        'nb': GaussianNB()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_labeled, y_labeled)
        trained_models[name] = model
    
    predictions = {}
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_unlabeled)[:, 1]
            predictions[name] = (pred_proba > 0.5).astype(int)
        else:
            pred = model.predict(X_unlabeled)
            predictions[name] = pred
    
    disagreement_scores = np.zeros(len(X_unlabeled))
    for i in range(len(X_unlabeled)):
        votes = [predictions[name][i] for name in predictions.keys()]
        majority_vote = max(set(votes), key=votes.count)
        disagreement = sum(1 for vote in votes if vote != majority_vote)
        disagreement_scores[i] = disagreement
    
    temp_df = X_unlabeled.copy()
    temp_df['disagreement'] = disagreement_scores
    
    selected = temp_df.nlargest(n_samples, 'disagreement')
    
    return selected.drop(columns=['disagreement'])


def matched_quantity_random_sampling(X_unlabeled, y_unlabeled, target_fraud_count, target_non_fraud_count, random_seed):
    """
    Sample exactly target_fraud_count fraud + target_non_fraud_count non-fraud samples
    Uses random selection within each class - ensures fair comparison
    """
    fraud_indices = X_unlabeled[y_unlabeled == 1].index.tolist()
    non_fraud_indices = X_unlabeled[y_unlabeled == 0].index.tolist()
    
    rng = np.random.RandomState(random_seed)
    
    actual_fraud = min(target_fraud_count, len(fraud_indices))
    actual_non_fraud = min(target_non_fraud_count, len(non_fraud_indices))
    
    selected_fraud = rng.choice(fraud_indices, size=actual_fraud, replace=False) if actual_fraud > 0 else []
    selected_non_fraud = rng.choice(non_fraud_indices, size=actual_non_fraud, replace=False) if actual_non_fraud > 0 else []
    
    all_selected = list(selected_fraud) + list(selected_non_fraud)
    rng.shuffle(all_selected)
    
    return X_unlabeled.loc[all_selected]


def train_and_evaluate(X_train, y_train, X_val, y_val, model_type='logistic'):
    """Train model and evaluate on validation set"""
    
    if model_type == 'logistic':
        model = LogisticRegression(
            C=0.1, random_state=42, max_iter=1000,
            solver='liblinear', class_weight='balanced'
        )
    elif model_type == 'logistic_unregularized':
        model = LogisticRegression(
            C=1.0, random_state=42, max_iter=1000,
            solver='liblinear', class_weight='balanced'
        )
    elif model_type == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                objective='binary', boosting_type='gbdt', num_leaves=31,
                learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.8,
                bagging_freq=5, verbose=0, random_state=42,
                class_weight='balanced', n_estimators=100
            )
        except ImportError:
            print("  ⚠️  LightGBM not available, falling back to Logistic Regression")
            model = LogisticRegression(
                C=0.1, random_state=42, max_iter=1000,
                solver='liblinear', class_weight='balanced'
            )
    else:
        print(f"  ⚠️  Unknown model type '{model_type}', using Logistic Regression")
        model = LogisticRegression(
            C=0.1, random_state=42, max_iter=1000,
            solver='liblinear', class_weight='balanced'
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0)
    }
    
    return model, metrics


def run_single_complete_experiment(X_train, y_train, X_test, y_test, config, random_seed=42):
    """
    Run a single complete experiment (11 iterations) with detailed tracking
    Returns both iteration-by-iteration results AND final test performance
    """
    
    # Create IDENTICAL validation splits for both approaches
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    
    # Create IDENTICAL base data for both approaches
    X_train_val = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_train_val = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    # Create IDENTICAL initial labeled pools for both approaches
    initial_indices = stratified_initial_split(X_train_val, y_train_val, config['initial_samples'], random_seed)
    
    # ACTIVE LEARNING SETUP - Gets its own independent copy
    X_labeled_active = X_train_val.loc[initial_indices].copy()
    y_labeled_active = y_train_val.loc[initial_indices].copy()
    X_unlabeled_active = X_train_val.drop(index=initial_indices).copy()
    y_unlabeled_active = y_train_val.drop(index=initial_indices).copy()
    
    # PASSIVE LEARNING SETUP - Gets its own independent copy
    X_labeled_passive = X_train_val.loc[initial_indices].copy()  # IDENTICAL initial pool
    y_labeled_passive = y_train_val.loc[initial_indices].copy()
    X_unlabeled_passive = X_train_val.drop(index=initial_indices).copy()  # IDENTICAL unlabeled pool
    y_unlabeled_passive = y_train_val.drop(index=initial_indices).copy()
    
    # Storage for iteration progression
    iteration_results = []
    
    # PARALLEL ITERATION LOOP (ALL 11 ITERATIONS)
    for iteration in range(1, config['n_iterations'] + 1):
        
        # ===================
        # ACTIVE LEARNING
        # ===================
        active_model, active_metrics = train_and_evaluate(
            X_labeled_active, y_labeled_active, X_val, y_val, config['model_type']
        )
        
        # Select new samples using active learning strategy
        active_new_samples = None
        active_fraud_count = 0
        active_non_fraud_count = 0
        
        if iteration < config['n_iterations'] and len(X_unlabeled_active) > 0:
            # Get the strategy for this specific iteration
            current_strategy = config['strategy_sequence'][iteration - 1] if iteration <= len(config['strategy_sequence']) else 'uncertainty'
            
            if current_strategy == 'uncertainty':
                active_new_samples = uncertainty_sampling(active_model, X_unlabeled_active, config['batch_size'])
            elif current_strategy == 'diversity':
                active_new_samples = diversity_sampling(X_unlabeled_active, config['batch_size'])
            elif current_strategy == 'qbc':
                active_new_samples = qbc_sampling(X_unlabeled_active, config['batch_size'], X_labeled_active, y_labeled_active)
            else:
                active_new_samples = X_unlabeled_active.sample(config['batch_size'], random_state=random_seed + iteration)
            
            # Count fraud/non-fraud in active learning's selection
            active_new_labels = y_unlabeled_active.loc[active_new_samples.index]
            active_fraud_count = (active_new_labels == 1).sum()
            active_non_fraud_count = (active_new_labels == 0).sum()
            
            # Add to active learning labeled pool
            X_labeled_active = pd.concat([X_labeled_active, active_new_samples])
            y_labeled_active = pd.concat([y_labeled_active, active_new_labels])
            
            # Remove from active learning unlabeled pool
            X_unlabeled_active = X_unlabeled_active.drop(index=active_new_samples.index)
            y_unlabeled_active = y_unlabeled_active.drop(index=active_new_samples.index)
        
        # ===================
        # PASSIVE LEARNING - MATCHED QUANTITIES
        # ===================
        passive_model, passive_metrics = train_and_evaluate(
            X_labeled_passive, y_labeled_passive, X_val, y_val, config['model_type']
        )
        
        # Select new samples using MATCHED QUANTITIES from active learning
        if iteration < config['n_iterations'] and len(X_unlabeled_passive) > 0 and active_new_samples is not None:
            # Match the exact composition that active learning found
            passive_new_samples = matched_quantity_random_sampling(
                X_unlabeled_passive, y_unlabeled_passive,
                active_fraud_count, active_non_fraud_count,
                random_seed + iteration + 1000  # Different seed than active learning
            )
            
            # Add to passive learning labeled pool
            passive_new_labels = y_unlabeled_passive.loc[passive_new_samples.index]
            X_labeled_passive = pd.concat([X_labeled_passive, passive_new_samples])
            y_labeled_passive = pd.concat([y_labeled_passive, passive_new_labels])
            
            # Remove from passive learning unlabeled pool
            X_unlabeled_passive = X_unlabeled_passive.drop(index=passive_new_samples.index)
            y_unlabeled_passive = y_unlabeled_passive.drop(index=passive_new_samples.index)
        
        # Calculate differences
        f1_difference = active_metrics['f1'] - passive_metrics['f1']
        f1_improvement_pct = (f1_difference / passive_metrics['f1'] * 100) if passive_metrics['f1'] > 0 else 0
        
        # Store iteration results
        current_strategy = config['strategy_sequence'][iteration - 1] if iteration <= len(config['strategy_sequence']) else 'uncertainty'
        iteration_results.append({
            'run_seed': random_seed,
            'iteration': iteration,
            'strategy_used': current_strategy,
            'active_f1': active_metrics['f1'],
            'passive_f1': passive_metrics['f1'],
            'active_accuracy': active_metrics['accuracy'],
            'passive_accuracy': passive_metrics['accuracy'],
            'active_precision': active_metrics['precision'],
            'passive_precision': passive_metrics['precision'],
            'active_recall': active_metrics['recall'],
            'passive_recall': passive_metrics['recall'],
            'f1_difference': f1_difference,
            'f1_improvement_pct': f1_improvement_pct,
            'active_labeled_count': len(X_labeled_active),
            'passive_labeled_count': len(X_labeled_passive),
            'active_fraud_count': (y_labeled_active == 1).sum(),
            'passive_fraud_count': (y_labeled_passive == 1).sum(),
            'active_fraud_rate': (y_labeled_active == 1).sum()/len(y_labeled_active)*100,
            'passive_fraud_rate': (y_labeled_passive == 1).sum()/len(y_labeled_passive)*100,
            'batch_fraud_count': active_fraud_count if active_new_samples is not None else 0,
            'batch_non_fraud_count': active_non_fraud_count if active_new_samples is not None else 0
        })
    
    # FINAL TEST SET EVALUATION (what gets aggregated across runs)
    active_final_model, active_final_metrics = train_and_evaluate(
        X_labeled_active, y_labeled_active, X_test, y_test, config['model_type']
    )
    
    passive_final_model, passive_final_metrics = train_and_evaluate(
        X_labeled_passive, y_labeled_passive, X_test, y_test, config['model_type']
    )
    
    return iteration_results, active_final_metrics, passive_final_metrics


def comprehensive_analysis(X_train, y_train, X_test, y_test, config=None, n_runs=10):
    """
    Run comprehensive analysis: 10 runs × 11 iterations
    Shows both volatility AND final aggregated results
    """
    
    # Use provided config or default to Champion Config 101
    if config is None:
        config = {
            'initial_samples': 300,
            'batch_size': 68,
            'n_iterations': 11,
            'model_type': 'logistic',
            'strategy_sequence': ['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty', 'diversity',
                                 'uncertainty', 'uncertainty', 'diversity', 'uncertainty', 'uncertainty', 'qbc']
        }
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE ANALYSIS: {n_runs} RUNS × 11 ITERATIONS")
    print(f"{'='*100}")
    print(f"🔬 METHODOLOGY: Fair Parallel Comparison with complete tracking")
    print(f"🎯 GOAL: Understand BOTH volatility AND final aggregated performance")
    print(f"📊 CONFIGURATION: {config.get('name', 'Custom')} ({config['model_type']})")
    print(f"🎯 STRATEGY: {config['strategy_sequence']}")
    print(f"{'='*100}")
    
    # Storage for all results
    all_iteration_results = []
    all_active_finals = []
    all_passive_finals = []
    
    # Run all experiments
    for run in range(n_runs):
        random_seed = 42 + run
        print(f"\n🏃 Running experiment {run+1}/{n_runs} (seed={random_seed})...")
        
        iteration_results, active_final, passive_final = run_single_complete_experiment(
            X_train, y_train, X_test, y_test, config, random_seed
        )
        
        all_iteration_results.extend(iteration_results)
        all_active_finals.append(active_final)
        all_passive_finals.append(passive_final)
        
        print(f"   Final: Active F1={active_final['f1']:.4f}, Passive F1={passive_final['f1']:.4f}, Diff={active_final['f1']-passive_final['f1']:+.4f}")
    
    # Convert to DataFrame for analysis
    iteration_df = pd.DataFrame(all_iteration_results)
    
    print(f"\n{'='*100}")
    print(f"VOLATILITY ANALYSIS: Iteration-by-Iteration Performance")
    print(f"{'='*100}")
    
    # Show volatility patterns
    print(f"\n📊 PERFORMANCE VOLATILITY BY ITERATION:")
    print(f"{'Iter':>4} {'Strategy':>11} {'Active_Avg':>10} {'Active_Std':>10} {'Passive_Avg':>11} {'Passive_Std':>11} {'Avg_Diff':>10} {'Avg_Imp%':>10}")
    print(f"{'-'*4} {'-'*11} {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    
    for iteration in range(1, 12):
        iter_data = iteration_df[iteration_df['iteration'] == iteration]
        if len(iter_data) > 0:
            strategy = iter_data['strategy_used'].iloc[0]
            active_avg = iter_data['active_f1'].mean()
            active_std = iter_data['active_f1'].std()
            passive_avg = iter_data['passive_f1'].mean()
            passive_std = iter_data['passive_f1'].std()
            avg_diff = iter_data['f1_difference'].mean()
            avg_imp = iter_data['f1_improvement_pct'].mean()
            
            print(f"{iteration:4d} {strategy:>11} {active_avg:10.4f} {active_std:10.4f} {passive_avg:11.4f} {passive_std:11.4f} {avg_diff:10.4f} {avg_imp:9.1f}%")
    
    # Show volatility statistics
    print(f"\n💡 VOLATILITY INSIGHTS:")
    active_volatility = iteration_df.groupby('iteration')['active_f1'].std().mean()
    passive_volatility = iteration_df.groupby('iteration')['passive_f1'].std().mean()
    print(f"  📈 Average Active Learning volatility (std): {active_volatility:.4f}")
    print(f"  📈 Average Passive Learning volatility (std): {passive_volatility:.4f}")
    print(f"  🎯 Active Learning is {active_volatility/passive_volatility:.2f}x more volatile than Passive Learning")
    
    # Show when divergence starts
    iter_improvements = iteration_df.groupby('iteration')['f1_improvement_pct'].mean()
    significant_iters = iter_improvements[abs(iter_improvements) > 10]
    if len(significant_iters) > 0:
        first_significant = significant_iters.index[0]
        print(f"  🚨 Significant divergence (>10%) starts at iteration {first_significant}")
    
    print(f"\n{'='*100}")
    print(f"FINAL AGGREGATED RESULTS: Test Set Performance")
    print(f"{'='*100}")
    
    # Extract final F1 scores
    active_f1_scores = [metrics['f1'] for metrics in all_active_finals]
    passive_f1_scores = [metrics['f1'] for metrics in all_passive_finals]
    
    # Calculate statistics
    active_mean = np.mean(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    active_std = np.std(active_f1_scores, ddof=1)
    passive_std = np.std(passive_f1_scores, ddof=1)
    
    improvement = active_mean - passive_mean
    improvement_pct = (improvement / passive_mean * 100) if passive_mean > 0 else 0
    
    # Statistical tests
    t_stat, p_value = stats.ttest_rel(active_f1_scores, passive_f1_scores)
    w_stat, w_p_value = stats.wilcoxon(active_f1_scores, passive_f1_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_runs - 1) * active_std**2 + (n_runs - 1) * passive_std**2) / (2 * n_runs - 2))
    cohens_d = improvement / pooled_std if pooled_std > 0 else 0
    
    print(f"\n📊 FINAL PERFORMANCE STATISTICS:")
    print(f"  🔴 Active Learning F1:  {active_mean:.4f} ± {active_std:.4f}")
    print(f"  🔵 Passive Learning F1: {passive_mean:.4f} ± {passive_std:.4f}")
    print(f"  📈 Mean Improvement: {improvement:.4f} ({improvement_pct:+.1f}%)")
    print(f"  🧪 Statistical Significance: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.6f})")
    print(f"  📏 Effect Size: Cohen's d = {cohens_d:.3f} ({'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'})")
    
    # Show run-by-run results
    print(f"\n📋 RUN-BY-RUN FINAL RESULTS:")
    print(f"{'Run':>3} {'Seed':>4} {'Active_F1':>10} {'Passive_F1':>11} {'Difference':>11} {'Improvement%':>12}")
    print(f"{'-'*3} {'-'*4} {'-'*10} {'-'*11} {'-'*11} {'-'*12}")
    
    for i in range(n_runs):
        seed = 42 + i
        active_f1 = active_f1_scores[i]
        passive_f1 = passive_f1_scores[i]
        diff = active_f1 - passive_f1
        imp_pct = (diff / passive_f1 * 100) if passive_f1 > 0 else 0
        
        print(f"{i+1:3d} {seed:4d} {active_f1:10.4f} {passive_f1:11.4f} {diff:11.4f} {imp_pct:11.1f}%")
    
    # Note: Detailed results are saved by the calling script in the appropriate location
    # No need to save redundant copies here
    
    final_results_data = []
    for i in range(n_runs):
        final_results_data.append({
            'run': i + 1,
            'seed': 42 + i,
            'active_f1': active_f1_scores[i],
            'passive_f1': passive_f1_scores[i],
            'active_accuracy': all_active_finals[i]['accuracy'],
            'passive_accuracy': all_passive_finals[i]['accuracy'],
            'active_precision': all_active_finals[i]['precision'],
            'passive_precision': all_passive_finals[i]['precision'],
            'active_recall': all_active_finals[i]['recall'],
            'passive_recall': all_passive_finals[i]['recall'],
            'f1_difference': active_f1_scores[i] - passive_f1_scores[i],
            'improvement_pct': ((active_f1_scores[i] - passive_f1_scores[i]) / passive_f1_scores[i] * 100) if passive_f1_scores[i] > 0 else 0
        })
    
    final_df = pd.DataFrame(final_results_data)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"  📊 Results are saved by the main experiment script")
    print(f"  📋 Check the experiment-specific directories for detailed outputs")
    
    return iteration_df, final_df, {
        'active_mean': active_mean,
        'passive_mean': passive_mean,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'active_volatility': active_volatility,
        'passive_volatility': passive_volatility
    }


def main():
    """Main function to run comprehensive analysis"""
    
    print("🔍 COMPREHENSIVE ITERATION ANALYSIS: 10 Runs × 11 Iterations")
    print("="*100)
    print("🎯 Purpose: Understand BOTH volatility AND final aggregated results")
    print("📊 Scope: Complete experimental pipeline with detailed tracking")
    print("🔬 Methodology: Fair parallel comparison across all runs and iterations")
    print("="*100)
    
    # Load fraud data
    fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
    
    if not os.path.exists(fraud_data_path):
        print(f"❌ Fraud dataset not found at: {fraud_data_path}")
        return
    
    print(f"✅ Loading fraud dataset from: {fraud_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
    
    # Run comprehensive analysis
    iteration_df, final_df, summary_stats = comprehensive_analysis(X_train, y_train, X_test, y_test, n_runs=10)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"📈 Final Result: Active Learning shows {summary_stats['improvement_pct']:+.1f}% improvement")
    print(f"🎭 Volatility: Active Learning is {summary_stats['active_volatility']/summary_stats['passive_volatility']:.2f}x more volatile")
    print(f"🧪 Statistical Significance: {'Yes' if summary_stats['p_value'] < 0.05 else 'No'}")
    print(f"📁 Check the CSV files for complete detailed data")


if __name__ == "__main__":
    main()
