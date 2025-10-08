#!/usr/bin/env python3
"""
Performance Evaluation Script for LLM Business Validation System

This script analyzes the performance of the LLM-based business validation system
by comparing predictions against ground truth labels and computing key metrics.

Usage:
    python evaluate_performance.py --input nyc_validation_68_improved-withactuals.csv
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from pathlib import Path


def load_validation_data(csv_path: str) -> pd.DataFrame:
    """Load validation results with ground truth labels."""
    df = pd.read_csv(csv_path)
    
    # Ensure boolean columns are properly formatted
    if df['is_valid'].dtype == 'object':
        df['is_valid'] = df['is_valid'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    if df['true_label_is_valid'].dtype == 'object':
        df['true_label_is_valid'] = df['true_label_is_valid'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    
    return df


def compute_metrics(df: pd.DataFrame, low_confidence_threshold: float = 70.0) -> dict:
    """Compute performance metrics for the validation system."""
    y_true = df['true_label_is_valid']
    y_pred = df['is_valid']
    
    # Basic classification metrics
    agreement_rate = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Failure rate (cases needing manual review)
    failure_rate = len(df[df['confidence'] < low_confidence_threshold]) / len(df)
    
    # Confidence analysis
    avg_confidence = df['confidence'].mean()
    low_confidence_count = len(df[df['confidence'] < 70])
    high_confidence_count = len(df[df['confidence'] >= 90])
    
    return {
        'agreement_rate': agreement_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'failure_rate': failure_rate,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'total_cases': len(df),
        'actually_valid': sum(y_true),
        'actually_invalid': len(df) - sum(y_true),
        'predicted_valid': sum(y_pred),
        'predicted_invalid': len(df) - sum(y_pred),
        'avg_confidence': avg_confidence,
        'low_confidence_count': low_confidence_count,
        'high_confidence_count': high_confidence_count,
        'low_confidence_threshold': low_confidence_threshold
    }


def analyze_false_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze false negative cases (valid restaurants marked as invalid)."""
    false_negatives = df[(df['true_label_is_valid'] == True) & (df['is_valid'] == False)]
    return false_negatives[['business_name', 'address', 'city', 'confidence', 'reasoning']]


def analyze_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze false positive cases (invalid businesses marked as valid)."""
    false_positives = df[(df['true_label_is_valid'] == False) & (df['is_valid'] == True)]
    return false_positives[['business_name', 'address', 'city', 'confidence', 'reasoning']]


def print_performance_report(metrics: dict, false_negatives: pd.DataFrame, false_positives: pd.DataFrame):
    """Print comprehensive performance report."""
    print('=' * 60)
    print('LLM BUSINESS VALIDATION PERFORMANCE REPORT')
    print('=' * 60)
    print()
    
    # Main metrics table
    print('PERFORMANCE METRICS')
    print('-' * 55)
    print(f"{'Metric':<35} {'Value':<20}")
    print('-' * 55)
    print(f"{'Agreement Rate':<35} {metrics['agreement_rate']:.1%}")
    print(f"{'Precision':<35} {metrics['precision']:.1%}")
    print(f"{'Recall':<35} {metrics['recall']:.1%}")
    print(f"{'F1-score':<35} {metrics['f1_score']:.1%}")
    print(f"{'Failure Rate (manual review needed)':<35} {metrics['failure_rate']:.1%}")
    print()
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    print('CONFUSION MATRIX')
    print('-' * 55)
    print(f"True Positives (Correctly identified valid):     {cm['tp']}")
    print(f"True Negatives (Correctly identified invalid):   {cm['tn']}")
    print(f"False Positives (Incorrectly marked valid):      {cm['fp']}")
    print(f"False Negatives (Missed valid restaurants):      {cm['fn']}")
    print()
    
    # Detailed breakdown
    print('DETAILED BREAKDOWN')
    print('-' * 55)
    print(f"Total cases:                    {metrics['total_cases']}")
    print(f"Actually valid restaurants:     {metrics['actually_valid']} ({metrics['actually_valid']/metrics['total_cases']:.1%})")
    print(f"Actually invalid businesses:    {metrics['actually_invalid']} ({metrics['actually_invalid']/metrics['total_cases']:.1%})")
    print(f"Predicted as valid:             {metrics['predicted_valid']} ({metrics['predicted_valid']/metrics['total_cases']:.1%})")
    print(f"Predicted as invalid:           {metrics['predicted_invalid']} ({metrics['predicted_invalid']/metrics['total_cases']:.1%})")
    print()
    
    # Confidence analysis
    print('CONFIDENCE ANALYSIS')
    print('-' * 55)
    print(f"Average confidence:             {metrics['avg_confidence']:.1f}%")
    print(f"Cases with confidence < 70%:    {metrics['low_confidence_count']} ({metrics['low_confidence_count']/metrics['total_cases']:.1%})")
    print(f"Cases with confidence >= 90%:   {metrics['high_confidence_count']} ({metrics['high_confidence_count']/metrics['total_cases']:.1%})")
    print()
    
    # Error analysis
    if len(false_negatives) > 0:
        print('FALSE NEGATIVES (Valid restaurants marked as invalid)')
        print('-' * 55)
        print(f"Total: {len(false_negatives)} cases")
        print()
        for i, (_, row) in enumerate(false_negatives.head(5).iterrows(), 1):
            print(f"{i}. {row['business_name']} - Confidence: {row['confidence']}%")
            print(f"   Address: {row['address']}, {row['city']}")
            print(f"   Reasoning: {row['reasoning'][:100]}...")
            print()
        if len(false_negatives) > 5:
            print(f"   ... and {len(false_negatives) - 5} more cases")
            print()
    
    if len(false_positives) > 0:
        print('FALSE POSITIVES (Invalid businesses marked as valid)')
        print('-' * 55)
        print(f"Total: {len(false_positives)} cases")
        print()
        for i, (_, row) in enumerate(false_positives.iterrows(), 1):
            print(f"{i}. {row['business_name']} - Confidence: {row['confidence']}%")
            print(f"   Address: {row['address']}, {row['city']}")
            print(f"   Reasoning: {row['reasoning'][:100]}...")
            print()


def save_metrics_to_file(metrics: dict, output_path: str):
    """Save metrics to a structured file."""
    with open(output_path, 'w') as f:
        f.write("# LLM Business Validation Performance Metrics\n\n")
        f.write("## Summary Metrics\n")
        f.write(f"- Agreement Rate: {metrics['agreement_rate']:.1%}\n")
        f.write(f"- Precision: {metrics['precision']:.1%}\n")
        f.write(f"- Recall: {metrics['recall']:.1%}\n")
        f.write(f"- F1-score: {metrics['f1_score']:.1%}\n")
        f.write(f"- Failure Rate: {metrics['failure_rate']:.1%}\n\n")
        
        f.write("## Confusion Matrix\n")
        cm = metrics['confusion_matrix']
        f.write(f"- True Positives: {cm['tp']}\n")
        f.write(f"- True Negatives: {cm['tn']}\n")
        f.write(f"- False Positives: {cm['fp']}\n")
        f.write(f"- False Negatives: {cm['fn']}\n\n")
        
        f.write("## Dataset Statistics\n")
        f.write(f"- Total cases: {metrics['total_cases']}\n")
        f.write(f"- Actually valid: {metrics['actually_valid']} ({metrics['actually_valid']/metrics['total_cases']:.1%})\n")
        f.write(f"- Actually invalid: {metrics['actually_invalid']} ({metrics['actually_invalid']/metrics['total_cases']:.1%})\n")
        f.write(f"- Average confidence: {metrics['avg_confidence']:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM business validation performance")
    parser.add_argument("--input", required=True, help="Path to CSV file with predictions and ground truth")
    parser.add_argument("--output", help="Path to save metrics summary (optional)")
    parser.add_argument("--confidence-threshold", type=float, default=70.0, 
                       help="Confidence threshold for manual review (default: 70.0)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading validation data from: {args.input}")
    df = load_validation_data(args.input)
    
    # Compute metrics
    metrics = compute_metrics(df, args.confidence_threshold)
    
    # Analyze errors
    false_negatives = analyze_false_negatives(df)
    false_positives = analyze_false_positives(df)
    
    # Print report
    print_performance_report(metrics, false_negatives, false_positives)
    
    # Save metrics if requested
    if args.output:
        save_metrics_to_file(metrics, args.output)
        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
