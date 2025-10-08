#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Best Configurations from Bank Experiments

This script runs the TOP 5 configurations from each category that performed best 
on the Bank Marketing dataset, adapted for fraud detection with stratified passive learning.

Categories:
1. LR Regularized + Standardized (Top 5) - CHAMPIONS from Bank experiments
2. LightGBM + Standardized (Top 5) 
3. LR Unregularized + Standardized (Top 5) for comparison
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append('.')

def create_config(config_id, name, model_type, regularized, strategies, description):
    """Create a standardized configuration"""
    return {
        'config_id': config_id,
        'name': name,
        'model_type': model_type,
        'regularized': regularized,
        'initial_samples': 300,
        'initial_strategy': 'stratified',  # Essential for fraud detection
        'batch_size': 68,
        'n_iterations': 11,
        'iteration_strategies': strategies,
        'description': description
    }

def run_experiment_config(config):
    """Run a single experiment configuration"""
    print(f"\n{'='*80}")
    print(f"RUNNING CONFIG {config['config_id']}: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Model: {config['model_type']} ({'Regularized' if config['regularized'] else 'Unregularized'})")
    print(f"Strategy: {config['iteration_strategies']}")
    print(f"{'='*80}")
    
    try:
        # Import the appropriate experiment module
        if config['model_type'] == 'lightgbm':
            from simple_active_learning_fraud_lgbm import run_multiple_experiments, load_and_split_data, perform_statistical_tests
            print("🚀 Using LightGBM implementation")
        else:
            from simple_active_learning_fraud import run_multiple_experiments, load_and_split_data, perform_statistical_tests
            print("📊 Using Logistic Regression implementation")
        
        # Set up the global model configuration
        import simple_active_learning_fraud as fraud_module
        fraud_module.MODEL_TYPE = config['model_type']
        
        # Load fraud data
        HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning'
        fraud_data_path = f'{HOME_DIR}/active-learning/data/european-credit-card-dataset/creditcard.csv'
        X_train, X_test, y_train, y_test = load_and_split_data(fraud_data_path)
        
        # Run the experiment
        all_active_results, all_passive_results, all_active_finals, all_passive_finals = run_multiple_experiments(
            X_train, X_test, y_train, y_test, config, n_runs=10
        )
        
        # Perform statistical tests
        stats_results = perform_statistical_tests(all_active_finals, all_passive_finals)
        
        # Save results
        config_name = f"fraud_config_{config['config_id']:03d}_{config['name'].lower().replace(' ', '_')}"
        
        import pandas as pd
        stats_filename = f'statistical_results_{config_name}.csv'
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{stats_filename}', index=False)
        
        # Create comparison table
        comparison_data = []
        for run in range(len(all_active_finals)):
            active_final = all_active_finals[run]
            passive_final = all_passive_finals[run]
            
            comparison_data.append({
                'Config_ID': config['config_id'],
                'Config_Name': config['name'],
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
        results_filename = f'statistical_test_data_{config_name}.csv'
        comparison_df.to_csv(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{results_filename}', index=False)
        
        # Summary statistics
        active_f1_scores = [row['Active_F1'] for row in comparison_data]
        passive_f1_scores = [row['Passive_F1'] for row in comparison_data]
        
        active_mean = np.mean(active_f1_scores)
        passive_mean = np.mean(passive_f1_scores)
        improvement = active_mean - passive_mean
        improvement_pct = (improvement / passive_mean * 100) if passive_mean > 0 else 0
        
        print(f"\n✅ CONFIG {config['config_id']} COMPLETED")
        print(f"Active Learning F1: {active_mean:.4f}")
        print(f"Passive Learning F1: {passive_mean:.4f}")
        print(f"Mean Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        print(f"Statistical Significance: {'Yes' if stats_results['p_value'] < 0.05 else 'No'} (p={stats_results['p_value']:.6f})")
        print(f"Effect Size: {stats_results['effect_interpretation']} (Cohen's d={stats_results['cohens_d']:.3f})")
        
        return {
            'config_id': config['config_id'],
            'name': config['name'],
            'active_f1': active_mean,
            'passive_f1': passive_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'p_value': stats_results['p_value'],
            'cohens_d': stats_results['cohens_d'],
            'significant': stats_results['p_value'] < 0.05,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ ERROR in Config {config['config_id']}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_id': config['config_id'],
            'name': config['name'],
            'error': str(e),
            'success': False
        }

def main():
    """Run all best configurations from Bank experiments on Fraud data"""
    
    print("🎯 Credit Card Fraud Detection - Best Bank Experiment Configurations")
    print("="*80)
    print("🎯 Implementation: All experiments use stratified passive learning for fair comparison")
    print("🏆 Running TOP 5 configurations from each Bank experiment category")
    print("="*80)
    
    # TOP 5 CONFIGURATIONS FROM BANK EXPERIMENTS
    
    configs = [
        
        # ===== CATEGORY 1: LR REGULARIZED + STANDARDIZED (CHAMPIONS) =====
        
        create_config(
            config_id=101,
            name="Config62_Champion",
            model_type="logistic",
            regularized=True,
            strategies=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty',  # 1-4: Build confidence
                       'diversity',                                                    # 5: Prevent overfitting
                       'uncertainty', 'uncertainty',                                  # 6-7: Maintain momentum  
                       'diversity',                                                    # 8: Strategic exploration
                       'uncertainty', 'uncertainty',                                  # 9-10: Final refinement
                       'qbc'],                                                        # 11: Ensemble disagreement
            description="OVERALL CHAMPION from Bank experiments (6.57% improvement). LR + Regularized + Standardized."
        ),
        
        create_config(
            config_id=102,
            name="Config58_Runner_Up",
            model_type="logistic", 
            regularized=True,
            strategies=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty',  # 1-4: Build confidence
                       'diversity',                                                    # 5: Prevent overfitting
                       'uncertainty', 'uncertainty',                                  # 6-7: Maintain momentum
                       'diversity',                                                    # 8: Strategic exploration  
                       'uncertainty', 'uncertainty',                                  # 9-10: Final refinement
                       'qbc'],                                                        # 11: Ensemble disagreement
            description="Runner-up from regularized standardized LR (6.57% improvement). Same as Config 62."
        ),
        
        create_config(
            config_id=103,
            name="Config59_High_Performer", 
            model_type="logistic",
            regularized=True,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Build confidence
                       'diversity',                                                    # 4: Early diversification
                       'uncertainty', 'uncertainty',                                  # 5-6: Maintain momentum
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 8-10: Extended refinement
                       'qbc'],                                                        # 11: Ensemble disagreement
            description="High-performing regularized standardized LR (6.39% improvement)."
        ),
        
        create_config(
            config_id=104,
            name="Config50_Baseline_Plus",
            model_type="logistic",
            regularized=True, 
            strategies=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty',  # 1-4: Extended confidence building
                       'uncertainty',                                                  # 5: Continue uncertainty
                       'diversity',                                                    # 6: Mid-point diversification
                       'uncertainty', 'uncertainty',                                  # 7-8: Return to uncertainty
                       'diversity',                                                    # 9: Late diversification
                       'uncertainty',                                                  # 10: Final uncertainty
                       'qbc'],                                                        # 11: Ensemble finale
            description="Strong regularized standardized LR baseline (5.8% improvement)."
        ),
        
        create_config(
            config_id=105, 
            name="Config61_Alternative",
            model_type="logistic",
            regularized=True,
            strategies=['uncertainty', 'uncertainty',                                # 1-2: Quick confidence building
                       'diversity',                                                    # 3: Early diversification
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 4-6: Extended uncertainty
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty', 'uncertainty',                                  # 8-9: Refinement
                       'qbc', 'qbc'],                                                 # 10-11: Extended QBC
            description="Alternative regularized standardized LR strategy (5.7% improvement)."
        ),
        
        # ===== CATEGORY 2: LIGHTGBM + STANDARDIZED =====
        
        create_config(
            config_id=201,
            name="Config95_LightGBM_Champion",
            model_type="lightgbm",
            regularized=False,  # LightGBM has built-in regularization
            strategies=['uncertainty', 'uncertainty',                                # 1-2: Build confidence  
                       'diversity',                                                    # 3: Prevent overfitting
                       'uncertainty', 'uncertainty',                                  # 4-5: Maintain momentum
                       'diversity',                                                    # 6: Strategic exploration
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 7-9: Final refinement
                       'qbc', 'qbc'],                                                 # 10-11: Extended ensemble disagreement
            description="LightGBM CHAMPION from Bank experiments (4.33% improvement)."
        ),
        
        create_config(
            config_id=202,
            name="Config83_LightGBM_Runner_Up",
            model_type="lightgbm",
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Extended confidence building
                       'diversity',                                                    # 4: Prevent overfitting
                       'uncertainty', 'uncertainty',                                  # 5-6: Maintain momentum
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty', 'uncertainty',                                  # 8-9: Refinement
                       'qbc', 'qbc'],                                                 # 10-11: Extended QBC finale
            description="LightGBM runner-up (3.9% improvement)."
        ),
        
        create_config(
            config_id=203,
            name="Config89_LightGBM_Alternative",
            model_type="lightgbm",
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty',  # 1-4: Extended confidence
                       'diversity',                                                    # 5: Mid-point diversification
                       'uncertainty',                                                  # 6: Return to uncertainty
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty', 'uncertainty',                                  # 8-9: Refinement
                       'qbc', 'qbc'],                                                 # 10-11: QBC finale
            description="Alternative LightGBM strategy (3.7% improvement)."
        ),
        
        create_config(
            config_id=204,
            name="Config96_LightGBM_Variant", 
            model_type="lightgbm",
            regularized=False,
            strategies=['uncertainty', 'uncertainty',                                # 1-2: Quick start
                       'diversity',                                                    # 3: Early diversification
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 4-6: Extended uncertainty
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty',                                                  # 8: Refinement
                       'qbc', 'qbc', 'qbc'],                                         # 9-11: Extended QBC
            description="LightGBM variant with extended QBC (3.5% improvement)."
        ),
        
        create_config(
            config_id=205,
            name="Config94_LightGBM_Baseline",
            model_type="lightgbm", 
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Build confidence
                       'diversity', 'diversity',                                      # 4-5: Extended diversification
                       'uncertainty', 'uncertainty',                                  # 6-7: Return to uncertainty
                       'diversity',                                                    # 8: Late diversification
                       'uncertainty',                                                  # 9: Final uncertainty
                       'qbc', 'qbc'],                                                 # 10-11: QBC finale
            description="LightGBM baseline with extended diversity (3.2% improvement)."
        ),
        
        # ===== CATEGORY 3: LR UNREGULARIZED + STANDARDIZED (for comparison) =====
        
        create_config(
            config_id=301,
            name="Config124_Unregularized_Champion",
            model_type="logistic_unregularized",
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Build confidence
                       'diversity',                                                    # 4: Prevent overfitting
                       'uncertainty',                                                  # 5: Maintain momentum
                       'diversity',                                                    # 6: Strategic exploration
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 7-9: Final refinement
                       'qbc', 'qbc'],                                                 # 10-11: Ensemble disagreement
            description="Unregularized LR CHAMPION from Bank experiments (5.37% improvement)."
        ),
        
        create_config(
            config_id=302,
            name="Config118_Unregularized_Runner_Up",
            model_type="logistic_unregularized",
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Build confidence
                       'diversity',                                                    # 4: Prevent overfitting
                       'uncertainty',                                                  # 5: Maintain momentum
                       'diversity',                                                    # 6: Strategic exploration
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 7-9: Final refinement
                       'qbc', 'qbc'],                                                 # 10-11: Ensemble disagreement
            description="Same as Config 124 - duplicate with 5.37% improvement."
        ),
        
        create_config(
            config_id=303,
            name="Config121_Unregularized_Alternative",
            model_type="logistic_unregularized",
            regularized=False,
            strategies=['uncertainty', 'uncertainty',                                # 1-2: Quick confidence
                       'diversity',                                                    # 3: Early diversification
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 4-6: Extended uncertainty
                       'diversity',                                                    # 7: Strategic exploration
                       'uncertainty', 'uncertainty',                                  # 8-9: Refinement
                       'qbc', 'qbc'],                                                 # 10-11: QBC finale
            description="Unregularized LR alternative (5.23% improvement)."
        ),
        
        create_config(
            config_id=304,
            name="Config128_Unregularized_Variant",
            model_type="logistic_unregularized", 
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty', 'uncertainty',  # 1-4: Extended confidence
                       'diversity',                                                    # 5: Mid-point diversification
                       'uncertainty', 'uncertainty',                                  # 6-7: Maintain momentum
                       'diversity',                                                    # 8: Strategic exploration
                       'uncertainty',                                                  # 9: Refinement
                       'qbc', 'qbc'],                                                 # 10-11: QBC finale
            description="Unregularized LR variant (4.91% improvement)."
        ),
        
        create_config(
            config_id=305,
            name="Config122_Unregularized_Baseline",
            model_type="logistic_unregularized",
            regularized=False,
            strategies=['uncertainty', 'uncertainty', 'uncertainty',                 # 1-3: Build confidence
                       'diversity', 'diversity',                                      # 4-5: Extended diversification
                       'uncertainty', 'uncertainty', 'uncertainty',                   # 6-8: Extended uncertainty
                       'diversity',                                                    # 9: Late diversification
                       'qbc', 'qbc'],                                                 # 10-11: QBC finale
            description="Unregularized LR baseline (3.50% improvement)."
        ),
    ]
    
    # Run all configurations
    results = []
    
    for config in configs:
        print(f"\n⏱️  Starting Config {config['config_id']}: {config['name']}")
        result = run_experiment_config(config)
        results.append(result)
        
        if result['success']:
            print(f"✅ Config {config['config_id']} completed successfully")
        else:
            print(f"❌ Config {config['config_id']} failed")
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY - BEST BANK CONFIGS ON FRAUD DATA")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Completed: {len(successful_results)}/{len(results)} configurations")
    
    if successful_results:
        print(f"\nTop Performers:")
        sorted_results = sorted(successful_results, key=lambda x: x['improvement'], reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):
            status = "🏆" if i == 0 else f"{i+1:2d}"
            significance = "✅" if result['significant'] else "❌"
            print(f"{status} Config {result['config_id']:3d}: {result['improvement']:+.4f} F1 ({result['improvement_pct']:+5.1f}%) {significance} {result['name']}")
        
        # Category analysis
        print(f"\nCategory Performance:")
        
        lr_reg = [r for r in successful_results if r['config_id'] < 200]
        lgbm = [r for r in successful_results if 200 <= r['config_id'] < 300]
        lr_unreg = [r for r in successful_results if r['config_id'] >= 300]
        
        categories = [
            ("LR Regularized + Standardized", lr_reg),
            ("LightGBM + Standardized", lgbm), 
            ("LR Unregularized + Standardized", lr_unreg)
        ]
        
        for name, category_results in categories:
            if category_results:
                avg_improvement = np.mean([r['improvement'] for r in category_results])
                significant_count = sum(1 for r in category_results if r['significant'])
                print(f"  {name}: {avg_improvement:+.4f} avg improvement ({significant_count}/{len(category_results)} significant)")
    
    if failed_results:
        print(f"\nFailed Configurations:")
        for result in failed_results:
            print(f"❌ Config {result['config_id']}: {result['name']} - {result['error']}")
    
    # Save summary
    import pandas as pd
    summary_data = []
    for result in results:
        if result['success']:
            summary_data.append({
                'Config_ID': result['config_id'],
                'Config_Name': result['name'],
                'Active_F1': result['active_f1'],
                'Passive_F1': result['passive_f1'],
                'F1_Improvement': result['improvement'],
                'Improvement_%': result['improvement_pct'],
                'P_Value': result['p_value'],
                'Cohens_D': result['cohens_d'],
                'Significant': result['significant']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_filename = 'fraud_best_configs_summary.csv'
        HOME_DIR = '/Users/lucasbraga/Documents/GitHub/active-learning'
        summary_df.to_csv(f'{HOME_DIR}/active-learning/experimentation-fraud/data/{summary_filename}', index=False)
        print(f"\n📊 Summary saved to: {summary_filename}")
    
    print(f"\n🎉 Experiment campaign completed!")
    print(f"📁 All results saved to: active-learning/experimentation-fraud/data/")

if __name__ == "__main__":
    main()
