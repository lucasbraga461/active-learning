#!/usr/bin/env python3
"""
Final Comprehensive LR Regularized Experiments

This script runs the FINAL comprehensive experiments for Logistic Regression with:
- Multiple strategy configurations from Bank experiments
- 10 runs × 11 iterations each
- Fair parallel comparison methodology
- Complete statistical analysis

Configurations tested:
1. Champion Config (uncertainty + diversity + qbc mix)
2. Uncertainty-focused config
3. Diversity-focused config
4. QBC-focused config
5. Balanced mixed strategy
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directories to path
sys.path.append('../core')
sys.path.append('../analysis')

# Import the comprehensive analysis framework
from comprehensive_iteration_analysis import comprehensive_analysis, load_and_split_data

HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning'

def create_lr_config(config_id, name, strategy_sequence, description):
    """Create a Logistic Regression configuration"""
    return {
        'config_id': config_id,
        'name': name,
        'model_type': 'logistic',  # LR regularized (C=0.1)
        'regularized': True,
        'initial_samples': 300,
        'batch_size': 68,
        'n_iterations': 11,
        'strategy_sequence': strategy_sequence,
        'description': description
    }

def setup_logging(experiment_name):
    """Setup logging for the experiment"""
    logs_dir = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_comprehensive_logs/lr_regularized'
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{logs_dir}/LR_REGULARIZED_5configs_10runs_11iters_{timestamp}.txt'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"📝 Logging started - {experiment_name}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Log file: {log_filename}")
    print("="*100)
    
    return logger

def main():
    """Run final comprehensive LR experiments"""
    
    logger = setup_logging("Final LR Regularized Comprehensive Experiments")
    
    print("🎯 FINAL COMPREHENSIVE LOGISTIC REGRESSION EXPERIMENTS")
    print("="*100)
    print("🔬 OBJECTIVE: Compare different AL strategies with LR regularized")
    print("📊 SCOPE: 5 configurations × 10 runs × 11 iterations = 550 experiments")
    print("⚙️  MODEL: Logistic Regression (C=0.1, regularized, class_weight='balanced')")
    print("🎯 DATASET: Credit Card Fraud Detection (284K samples, 0.173% fraud)")
    print("="*100)
    
    # Load fraud data ONCE for all experiments
    fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
    
    if not os.path.exists(fraud_data_path):
        print(f"❌ Fraud dataset not found at: {fraud_data_path}")
        return
    
    print(f"✅ Loading fraud dataset from: {fraud_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
    
    # Define LR strategy configurations
    configs = [
        
        create_lr_config(
            config_id=1001,
            name="Champion_Config62",
            strategy_sequence=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty', 'diversity',
                              'uncertainty', 'uncertainty', 'diversity', 'uncertainty', 'uncertainty', 'qbc'],
            description="CHAMPION from Bank experiments - balanced uncertainty/diversity/qbc strategy"
        ),
        
        create_lr_config(
            config_id=1002,
            name="Uncertainty_Focused",
            strategy_sequence=['uncertainty'] * 11,
            description="Pure uncertainty sampling - focuses on model uncertainty"
        ),
        
        create_lr_config(
            config_id=1003,
            name="Diversity_Focused",
            strategy_sequence=['uncertainty', 'uncertainty', 'diversity', 'diversity', 'diversity',
                              'diversity', 'diversity', 'diversity', 'uncertainty', 'uncertainty', 'qbc'],
            description="Diversity-heavy strategy - explores feature space broadly"
        ),
        
        create_lr_config(
            config_id=1004,
            name="QBC_Focused", 
            strategy_sequence=['uncertainty', 'uncertainty', 'uncertainty', 'qbc', 'qbc',
                              'qbc', 'qbc', 'qbc', 'qbc', 'qbc', 'qbc'],
            description="Query-by-Committee heavy - leverages ensemble disagreement"
        ),
        
        create_lr_config(
            config_id=1005,
            name="Balanced_Mixed",
            strategy_sequence=['uncertainty', 'diversity', 'qbc', 'uncertainty', 'diversity',
                              'qbc', 'uncertainty', 'diversity', 'qbc', 'uncertainty', 'diversity'],
            description="Perfectly balanced rotation of all three strategies"
        ),
    ]
    
    # Run all configurations
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n🚀 STARTING CONFIG {config['config_id']} ({i+1}/{len(configs)})")
        print(f"📋 Name: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"🎯 Strategy: {config['strategy_sequence']}")
        print("-" * 80)
        
        try:
            # Run comprehensive analysis for this configuration
            iteration_df, final_df, summary_stats = comprehensive_analysis(
                X_train, y_train, X_test, y_test, 
                config=config,  # Pass the config
                n_runs=10
            )
            
            # Store results
            config_result = {
                'config_id': config['config_id'],
                'name': config['name'],
                'description': config['description'],
                'strategy_sequence': str(config['strategy_sequence']),
                'active_f1_mean': summary_stats['active_mean'],
                'passive_f1_mean': summary_stats['passive_mean'],
                'improvement': summary_stats['improvement'],
                'improvement_pct': summary_stats['improvement_pct'],
                'p_value': summary_stats['p_value'],
                'cohens_d': summary_stats['cohens_d'],
                'active_volatility': summary_stats['active_volatility'],
                'passive_volatility': summary_stats['passive_volatility'],
                'volatility_ratio': summary_stats['active_volatility'] / summary_stats['passive_volatility'],
                'significant': summary_stats['p_value'] < 0.05,
                'success': True
            }
            
            all_results.append(config_result)
            
            # Save individual config results
            results_dir = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_lr_comprehensive'
            os.makedirs(results_dir, exist_ok=True)
            
            config_name = f"lr_config_{config['config_id']}_{config['name'].lower()}"
            iteration_df.to_csv(f'{results_dir}/{config_name}_iterations.csv', index=False)
            final_df.to_csv(f'{results_dir}/{config_name}_final.csv', index=False)
            
            print(f"✅ CONFIG {config['config_id']} COMPLETED")
            print(f"📊 Result: Active F1={summary_stats['active_mean']:.4f}, Passive F1={summary_stats['passive_mean']:.4f}")
            print(f"📈 Improvement: {summary_stats['improvement_pct']:+.1f}% (p={summary_stats['p_value']:.6f})")
            
        except Exception as e:
            print(f"❌ CONFIG {config['config_id']} FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            config_result = {
                'config_id': config['config_id'],
                'name': config['name'],
                'error': str(e),
                'success': False
            }
            all_results.append(config_result)
    
    # Final summary analysis
    print(f"\n{'='*100}")
    print("FINAL LR COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    print(f"✅ Successful: {len(successful_results)}/{len(all_results)} configurations")
    
    if successful_results:
        print(f"\n🏆 PERFORMANCE RANKING:")
        sorted_results = sorted(successful_results, key=lambda x: x['improvement_pct'], reverse=True)
        
        for i, result in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"
            significance = "✅" if result['significant'] else "❌"
            print(f"{rank} {result['name']:20s}: {result['improvement_pct']:+6.1f}% {significance} (volatility: {result['volatility_ratio']:.2f}x)")
        
        print(f"\n📊 STRATEGY ANALYSIS:")
        print(f"{'Strategy':20s} {'Improvement%':>12s} {'Volatility':>10s} {'P-Value':>10s}")
        print("-" * 54)
        
        for result in sorted_results:
            print(f"{result['name']:20s} {result['improvement_pct']:11.1f}% {result['volatility_ratio']:9.2f}x {result['p_value']:9.6f}")
        
        # Best strategy insights
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"  🏆 Best Strategy: {best['name']} (+{best['improvement_pct']:.1f}%)")
        print(f"  📉 Worst Strategy: {worst['name']} (+{worst['improvement_pct']:.1f}%)")
        print(f"  📊 Performance Range: {worst['improvement_pct']:.1f}% to {best['improvement_pct']:.1f}%")
        
        avg_improvement = np.mean([r['improvement_pct'] for r in successful_results])
        avg_volatility = np.mean([r['volatility_ratio'] for r in successful_results])
        
        print(f"  📈 Average Improvement: {avg_improvement:.1f}%")
        print(f"  🎭 Average Volatility Ratio: {avg_volatility:.2f}x")
    
    if failed_results:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        for result in failed_results:
            print(f"   {result['name']}: {result['error']}")
    
    # Save comprehensive summary
    if successful_results:
        summary_df = pd.DataFrame(successful_results)
        summary_filename = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_lr_comprehensive_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"\n💾 Summary saved to: {summary_filename}")
    
    print(f"\n🎉 FINAL LR COMPREHENSIVE EXPERIMENTS COMPLETED!")
    print(f"📁 Results saved to: active-learning/experimentation-fraud/results/current/final_lr_comprehensive/")
    
    logger.close()
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
