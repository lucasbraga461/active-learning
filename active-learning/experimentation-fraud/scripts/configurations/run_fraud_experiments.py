#!/usr/bin/env python3
"""
Run Credit Card Fraud Detection Experiments

This script runs the full Active Learning experiments for the credit card fraud dataset
using the same methodology as the Bank Marketing experiments.
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def run_experiment(script_name, description):
    """Run an experiment script and handle any errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}")
    
    try:
        # Import and run the main function from the script
        if script_name == 'simple_active_learning_fraud.py':
            from simple_active_learning_fraud import main
        elif script_name == 'simple_active_learning_fraud_lgbm.py':
            from simple_active_learning_fraud_lgbm import main
        else:
            print(f"❌ Unknown script: {script_name}")
            return False
        
        # Run the experiment
        main()
        print(f"\n✅ {description} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in {description}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all fraud detection experiments"""
    print("🚀 Starting Credit Card Fraud Detection Active Learning Experiments")
    print("="*80)
    
    # List of experiments to run
    experiments = [
        ('simple_active_learning_fraud.py', 'Fraud Detection with Logistic Regression'),
        ('simple_active_learning_fraud_lgbm.py', 'Fraud Detection with LightGBM'),
    ]
    
    results = []
    
    for script, description in experiments:
        print(f"\n⏱️  Starting: {description}")
        success = run_experiment(script, description)
        results.append((description, success))
        
        if success:
            print(f"✅ Completed: {description}")
        else:
            print(f"❌ Failed: {description}")
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    successful_experiments = sum(1 for _, success in results if success)
    total_experiments = len(results)
    
    print(f"\nCompleted {successful_experiments}/{total_experiments} experiments successfully")
    
    if successful_experiments == total_experiments:
        print("\n🎉 All experiments completed successfully!")
        print("Check the active-learning/experimentation-fraud/data/ folder for results and logs.")
    else:
        print(f"\n⚠️  {total_experiments - successful_experiments} experiments failed. Check the logs for details.")

if __name__ == "__main__":
    main()
