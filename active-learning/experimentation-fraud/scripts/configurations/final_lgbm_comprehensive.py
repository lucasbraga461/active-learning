#!/usr/bin/env python3
"""
Final Comprehensive LightGBM Experiments

This script runs the FINAL comprehensive experiments for LightGBM with:
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

def create_lgbm_config(config_id, name, strategy_sequence, description):
    """Create a LightGBM configuration"""
    return {
        'config_id': config_id,
        'name': name,
        'model_type': 'lightgbm',  # LightGBM with built-in regularization
        'regularized': False,  # LightGBM handles regularization internally
        'initial_samples': 300,
        'batch_size': 68,
        'n_iterations': 11,
        'strategy_sequence': strategy_sequence,
        'description': description
    }

def setup_logging(experiment_name):
    """Setup logging for the experiment"""
    logs_dir = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_comprehensive_logs/lightgbm'
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{logs_dir}/LIGHTGBM_5configs_10runs_11iters_{timestamp}.txt'
    
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
    """Run final comprehensive LightGBM experiments"""
    
    logger = setup_logging("Final LightGBM Comprehensive Experiments")
    
    print("🚀 FINAL COMPREHENSIVE LIGHTGBM EXPERIMENTS")
    print("="*100)
    print("🔬 OBJECTIVE: Compare different AL strategies with LightGBM")
    print("📊 SCOPE: 5 configurations × 10 runs × 11 iterations = 550 experiments")
    print("⚙️  MODEL: LightGBM (n_estimators=100, learning_rate=0.05, class_weight='balanced')")
    print("🎯 DATASET: Credit Card Fraud Detection (284K samples, 0.173% fraud)")
    print("="*100)
    
    # Load fraud data ONCE for all experiments
    fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
    
    if not os.path.exists(fraud_data_path):
        print(f"❌ Fraud dataset not found at: {fraud_data_path}")
        return
    
    print(f"✅ Loading fraud dataset from: {fraud_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
    
    # Define LightGBM strategy configurations
    configs = [
        
        create_lgbm_config(
            config_id=2001,
            name="Champion_Config95",
            strategy_sequence=['uncertainty', 'uncertainty', 'diversity', 'uncertainty', 'uncertainty',
                              'diversity', 'uncertainty', 'uncertainty', 'uncertainty', 'qbc', 'qbc'],
            description="CHAMPION from Bank experiments - LightGBM optimized strategy"
        ),
        
        create_lgbm_config(
            config_id=2002,
            name="Uncertainty_Focused",
            strategy_sequence=['uncertainty'] * 11,
            description="Pure uncertainty sampling - leverages LightGBM probability estimates"
        ),
        
        create_lgbm_config(
            config_id=2003,
            name="Diversity_Focused",
            strategy_sequence=['uncertainty', 'uncertainty', 'diversity', 'diversity', 'diversity',
                              'diversity', 'diversity', 'diversity', 'uncertainty', 'uncertainty', 'qbc'],
            description="Diversity-heavy strategy - explores feature space with tree-based insights"
        ),
        
        create_lgbm_config(
            config_id=2004,
            name="QBC_Focused", 
            strategy_sequence=['uncertainty', 'uncertainty', 'uncertainty', 'qbc', 'qbc',
                              'qbc', 'qbc', 'qbc', 'qbc', 'qbc', 'qbc'],
            description="Query-by-Committee heavy - benefits from LightGBM ensemble nature"
        ),
        
        create_lgbm_config(
            config_id=2005,
            name="Tree_Optimized",
            strategy_sequence=['uncertainty', 'diversity', 'uncertainty', 'diversity', 'qbc',
                              'uncertainty', 'diversity', 'qbc', 'uncertainty', 'qbc', 'qbc'],
            description="Tree-optimized strategy - balanced approach tailored for gradient boosting"
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
            results_dir = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_lgbm_comprehensive'
            os.makedirs(results_dir, exist_ok=True)
            
            config_name = f"lgbm_config_{config['config_id']}_{config['name'].lower()}"
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
    print("FINAL LIGHTGBM COMPREHENSIVE EXPERIMENT SUMMARY")
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
        
        # LightGBM specific insights
        print(f"\n🚀 LIGHTGBM SPECIFIC INSIGHTS:")
        uncertainty_results = [r for r in successful_results if 'uncertainty' in r['name'].lower()]
        diversity_results = [r for r in successful_results if 'diversity' in r['name'].lower()]
        qbc_results = [r for r in successful_results if 'qbc' in r['name'].lower()]
        
        if uncertainty_results:
            unc_avg = np.mean([r['improvement_pct'] for r in uncertainty_results])
            print(f"  🎯 Uncertainty-focused strategies: {unc_avg:.1f}% avg improvement")
        
        if diversity_results:
            div_avg = np.mean([r['improvement_pct'] for r in diversity_results])
            print(f"  🌐 Diversity-focused strategies: {div_avg:.1f}% avg improvement")
            
        if qbc_results:
            qbc_avg = np.mean([r['improvement_pct'] for r in qbc_results])
            print(f"  🤝 QBC-focused strategies: {qbc_avg:.1f}% avg improvement")
    
    if failed_results:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        for result in failed_results:
            print(f"   {result['name']}: {result['error']}")
    
    # Save comprehensive summary
    if successful_results:
        summary_df = pd.DataFrame(successful_results)
        summary_filename = f'{HOME_DIR}/active-learning/experimentation-fraud/results/current/final_lgbm_comprehensive_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"\n💾 Summary saved to: {summary_filename}")
    
    print(f"\n🎉 FINAL LIGHTGBM COMPREHENSIVE EXPERIMENTS COMPLETED!")
    print(f"📁 Results saved to: active-learning/experimentation-fraud/results/current/final_lgbm_comprehensive/")
    
    logger.close()
    sys.stdout = logger.terminal

if __name__ == "__main__":
    main()
