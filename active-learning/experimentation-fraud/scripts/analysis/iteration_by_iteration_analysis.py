#!/usr/bin/env python3
"""
Iteration-by-Iteration Analysis: Active vs Passive Learning Divergence

This script shows EXACTLY how Active Learning and Passive Learning diverge
iteration by iteration up to iteration 6, helping understand the massive differences.

Focus: Detailed analysis of one configuration with rich logging
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
    print(f"  🎯 Creating stratified initial split with {initial_samples} samples...")
    
    fraud_indices = X[y == 1].index.tolist()
    non_fraud_indices = X[y == 0].index.tolist()
    
    print(f"    Available fraud samples: {len(fraud_indices)}")
    print(f"    Available non-fraud samples: {len(non_fraud_indices)}")
    
    actual_fraud_samples = min(min_fraud_samples, len(fraud_indices))
    remaining_samples = initial_samples - actual_fraud_samples
    
    rng = np.random.RandomState(random_state)
    
    selected_fraud = rng.choice(fraud_indices, size=actual_fraud_samples, replace=False)
    selected_non_fraud = rng.choice(non_fraud_indices, size=remaining_samples, replace=False)
    
    initial_indices = np.concatenate([selected_fraud, selected_non_fraud])
    rng.shuffle(initial_indices)
    
    print(f"    ✓ Selected {actual_fraud_samples} fraud + {remaining_samples} non-fraud samples")
    print(f"    ✓ Initial fraud percentage: {actual_fraud_samples/initial_samples*100:.2f}%")
    
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


def detailed_iteration_analysis(X_train, y_train, X_test, y_test, random_seed=42):
    """
    Run detailed iteration-by-iteration analysis showing how AL and PL diverge
    """
    print(f"\n{'='*80}")
    print(f"DETAILED ITERATION-BY-ITERATION ANALYSIS")
    print(f"{'='*80}")
    print(f"🔬 METHODOLOGY: Fair Parallel Comparison (NO experimental bias)")
    print(f"🎯 GOAL: Understand how Active vs Passive Learning diverge")
    print(f"📊 ANALYSIS: Iterations 1-6 with detailed logging")
    print(f"⚙️  CONFIGURATION: Config 101 Champion (LR + Mixed Strategy Sequence)")
    print(f"{'='*80}")
    
    # Configuration - Using ACTUAL Config 101 (Champion) strategy sequence
    config = {
        'initial_samples': 300,
        'batch_size': 68,
        'n_iterations': 6,  # Stop at iteration 6 as requested
        'model_type': 'logistic',
        'strategy_sequence': ['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty', 'diversity', 'uncertainty']  # First 6 from champion config
    }
    
    # Create IDENTICAL validation splits for both approaches
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    
    # Create IDENTICAL base data for both approaches
    X_train_val = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_train_val = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    print(f"🔍 VALIDATION SPLIT:")
    print(f"  Training pool: {len(X_train_val)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Fraud in training pool: {(y_train_val == 1).sum()}")
    print(f"  Fraud in validation: {(y_val == 1).sum()}")
    print(f"🎯 STRATEGY SEQUENCE: {config['strategy_sequence']}")
    
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
    
    print(f"\n🚀 INITIAL STATE (IDENTICAL FOR BOTH):")
    print(f"  Labeled pool: {len(X_labeled_active)} samples")
    print(f"  Fraud in labeled: {(y_labeled_active == 1).sum()} ({(y_labeled_active == 1).sum()/len(y_labeled_active)*100:.2f}%)")
    print(f"  Unlabeled pool: {len(X_unlabeled_active)} samples")
    print(f"  Fraud in unlabeled: {(y_unlabeled_active == 1).sum()} ({(y_unlabeled_active == 1).sum()/len(y_unlabeled_active)*100:.3f}%)")
    
    # Storage for iteration progression
    iteration_results = []
    
    # PARALLEL ITERATION LOOP
    for iteration in range(1, config['n_iterations'] + 1):
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # ===================
        # ACTIVE LEARNING
        # ===================
        print(f"\n🔴 ACTIVE LEARNING - Iteration {iteration}")
        print(f"  📊 Current labeled pool: {len(X_labeled_active)} samples")
        print(f"  🎯 Fraud in labeled: {(y_labeled_active == 1).sum()} samples ({(y_labeled_active == 1).sum()/len(y_labeled_active)*100:.2f}%)")
        
        active_model, active_metrics = train_and_evaluate(
            X_labeled_active, y_labeled_active, X_val, y_val, config['model_type']
        )
        
        print(f"  📈 Validation Performance: F1={active_metrics['f1']:.4f}, Acc={active_metrics['accuracy']:.4f}, Prec={active_metrics['precision']:.4f}, Rec={active_metrics['recall']:.4f}")
        
        # Select new samples using active learning strategy
        active_new_samples = None
        active_fraud_count = 0
        active_non_fraud_count = 0
        
        if iteration < config['n_iterations'] and len(X_unlabeled_active) > 0:
            # Get the strategy for this specific iteration
            current_strategy = config['strategy_sequence'][iteration - 1] if iteration <= len(config['strategy_sequence']) else 'uncertainty'
            print(f"  🎯 Selecting {config['batch_size']} new samples using {current_strategy} sampling...")
            
            if current_strategy == 'uncertainty':
                active_new_samples = uncertainty_sampling(active_model, X_unlabeled_active, config['batch_size'])
            elif current_strategy == 'diversity':
                active_new_samples = diversity_sampling(X_unlabeled_active, config['batch_size'])
            else:
                active_new_samples = X_unlabeled_active.sample(config['batch_size'], random_state=random_seed + iteration)
            
            # Count fraud/non-fraud in active learning's selection
            active_new_labels = y_unlabeled_active.loc[active_new_samples.index]
            active_fraud_count = (active_new_labels == 1).sum()
            active_non_fraud_count = (active_new_labels == 0).sum()
            
            print(f"  ✅ Active Learning found: {active_fraud_count} fraud + {active_non_fraud_count} non-fraud = {len(active_new_samples)} total")
            print(f"  📊 Active Learning fraud rate in batch: {active_fraud_count/len(active_new_samples)*100:.2f}%")
            
            # Add to active learning labeled pool
            X_labeled_active = pd.concat([X_labeled_active, active_new_samples])
            y_labeled_active = pd.concat([y_labeled_active, active_new_labels])
            
            # Remove from active learning unlabeled pool
            X_unlabeled_active = X_unlabeled_active.drop(index=active_new_samples.index)
            y_unlabeled_active = y_unlabeled_active.drop(index=active_new_samples.index)
        
        # ===================
        # PASSIVE LEARNING - MATCHED QUANTITIES
        # ===================
        print(f"\n🔵 PASSIVE LEARNING - Iteration {iteration}")
        print(f"  📊 Current labeled pool: {len(X_labeled_passive)} samples")
        print(f"  🎯 Fraud in labeled: {(y_labeled_passive == 1).sum()} samples ({(y_labeled_passive == 1).sum()/len(y_labeled_passive)*100:.2f}%)")
        
        passive_model, passive_metrics = train_and_evaluate(
            X_labeled_passive, y_labeled_passive, X_val, y_val, config['model_type']
        )
        
        print(f"  📈 Validation Performance: F1={passive_metrics['f1']:.4f}, Acc={passive_metrics['accuracy']:.4f}, Prec={passive_metrics['precision']:.4f}, Rec={passive_metrics['recall']:.4f}")
        
        # Select new samples using MATCHED QUANTITIES from active learning
        if iteration < config['n_iterations'] and len(X_unlabeled_passive) > 0 and active_new_samples is not None:
            print(f"  🎯 Matching Active Learning's selection: {active_fraud_count} fraud + {active_non_fraud_count} non-fraud...")
            
            # Match the exact composition that active learning found
            passive_new_samples = matched_quantity_random_sampling(
                X_unlabeled_passive, y_unlabeled_passive,
                active_fraud_count, active_non_fraud_count,
                random_seed + iteration + 1000  # Different seed than active learning
            )
            
            print(f"  ✅ Passive Learning selected: {(y_unlabeled_passive.loc[passive_new_samples.index] == 1).sum()} fraud + {(y_unlabeled_passive.loc[passive_new_samples.index] == 0).sum()} non-fraud = {len(passive_new_samples)} total")
            print(f"  📊 Passive Learning fraud rate in batch: {(y_unlabeled_passive.loc[passive_new_samples.index] == 1).sum()/len(passive_new_samples)*100:.2f}%")
            
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
        
        print(f"\n📊 ITERATION {iteration} SUMMARY:")
        print(f"  🔴 Active Learning F1:  {active_metrics['f1']:.4f}")
        print(f"  🔵 Passive Learning F1: {passive_metrics['f1']:.4f}")
        print(f"  📈 Difference: {f1_difference:+.4f} ({f1_improvement_pct:+.1f}%)")
        print(f"  🎯 Both labeled pools now have: {len(X_labeled_active)} samples")
        print(f"  🔴 Active fraud in pool: {(y_labeled_active == 1).sum()} ({(y_labeled_active == 1).sum()/len(y_labeled_active)*100:.2f}%)")
        print(f"  🔵 Passive fraud in pool: {(y_labeled_passive == 1).sum()} ({(y_labeled_passive == 1).sum()/len(y_labeled_passive)*100:.2f}%)")
        
        # Store results
        current_strategy = config['strategy_sequence'][iteration - 1] if iteration <= len(config['strategy_sequence']) else 'uncertainty'
        iteration_results.append({
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
    
    # Create summary table
    print(f"\n{'='*100}")
    print(f"ITERATION-BY-ITERATION DIVERGENCE ANALYSIS")
    print(f"{'='*100}")
    
    print(f"{'Iter':>4} {'Strategy':>11} {'Active_F1':>10} {'Passive_F1':>11} {'Difference':>11} {'Improvement%':>12} {'AL_Fraud%':>10} {'PL_Fraud%':>10} {'Batch_Fraud':>11}")
    print(f"{'-'*4} {'-'*11} {'-'*10} {'-'*11} {'-'*11} {'-'*12} {'-'*10} {'-'*10} {'-'*11}")
    
    for result in iteration_results:
        print(f"{result['iteration']:4d} {result['strategy_used']:>11} {result['active_f1']:10.4f} {result['passive_f1']:11.4f} {result['f1_difference']:11.4f} {result['f1_improvement_pct']:11.1f}% {result['active_fraud_rate']:9.2f}% {result['passive_fraud_rate']:9.2f}% {result['batch_fraud_count']:11d}")
    
    # Analysis insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Find when divergence starts
    significant_divergence = next((r for r in iteration_results if abs(r['f1_improvement_pct']) > 10), None)
    if significant_divergence:
        print(f"  🎯 Significant divergence (>10%) starts at iteration {significant_divergence['iteration']}")
    
    # Fraud accumulation analysis
    final_result = iteration_results[-1]
    print(f"  📊 By iteration {config['n_iterations']}:")
    print(f"    - Active Learning fraud rate: {final_result['active_fraud_rate']:.2f}%")
    print(f"    - Passive Learning fraud rate: {final_result['passive_fraud_rate']:.2f}%")
    print(f"    - Final F1 difference: {final_result['f1_difference']:.4f} ({final_result['f1_improvement_pct']:+.1f}%)")
    
    # Batch analysis
    total_batch_fraud = sum(r['batch_fraud_count'] for r in iteration_results)
    total_batch_samples = sum(r['batch_fraud_count'] + r['batch_non_fraud_count'] for r in iteration_results)
    if total_batch_samples > 0:
        avg_batch_fraud_rate = total_batch_fraud / total_batch_samples * 100
        print(f"  🎯 Active Learning's average batch fraud rate: {avg_batch_fraud_rate:.2f}%")
        natural_fraud_rate = (y_train_val == 1).sum() / len(y_train_val) * 100
        print(f"  📊 Natural fraud rate in dataset: {natural_fraud_rate:.3f}%")
        print(f"  🚀 Active Learning finds fraud at {avg_batch_fraud_rate/natural_fraud_rate:.1f}x the natural rate!")
    
    # Save detailed results
    results_df = pd.DataFrame(iteration_results)
    results_filename = f'{HOME_DIR}/active-learning/experimentation-fraud/data/iteration_by_iteration_analysis.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\n💾 Detailed results saved to: {results_filename}")
    
    return iteration_results


def main():
    """Main function to run iteration-by-iteration analysis"""
    
    print("🔍 ITERATION-BY-ITERATION ANALYSIS: Understanding AL vs PL Divergence")
    print("="*80)
    print("🎯 Purpose: Understand how Active Learning diverges from Passive Learning")
    print("📊 Focus: Detailed analysis through iterations 1-6")
    print("🔬 Methodology: Fair parallel comparison with matched quantities")
    print("="*80)
    
    # Load fraud data
    fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
    
    if not os.path.exists(fraud_data_path):
        print(f"❌ Fraud dataset not found at: {fraud_data_path}")
        return
    
    print(f"✅ Loading fraud dataset from: {fraud_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
    
    # Run detailed analysis
    iteration_results = detailed_iteration_analysis(X_train, y_train, X_test, y_test, random_seed=42)
    
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"📁 Check the detailed CSV file for complete iteration-by-iteration data")
    print(f"💡 This shows EXACTLY how Active Learning finds better samples than random sampling")


if __name__ == "__main__":
    main()
