#!/usr/bin/env python3
"""
Test Runner for Matched Quantities Approach

Quick test to verify the matched quantities methodology produces realistic results
before running the full experimental suite.
"""

import sys
import os
import subprocess

def run_single_test(script_name, model_type):
    """Run a single matched quantities test"""
    print(f"\n{'='*60}")
    print(f"TESTING: {script_name} ({model_type.upper()})")
    print(f"{'='*60}")
    
    try:
        # Change to the correct directory and run
        result = subprocess.run([
            'python3', script_name
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            
            # Extract key results from output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-30:]:  # Check last 30 lines for results
                if 'Active Learning  F1:' in line or 'Passive Learning F1:' in line or 'Improvement %:' in line:
                    print(f"  📊 {line.strip()}")
            
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr[:500]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"💥 {script_name} crashed: {e}")
        return False

def main():
    """Run matched quantities tests"""
    print("🧪 MATCHED QUANTITIES METHODOLOGY TEST")
    print("="*80)
    print("🎯 Objective: Verify matched quantities produces realistic improvements (15-30%)")
    print("🚫 Problem: Previous approach showed unrealistic 456% improvement")
    print("✅ Solution: Match sample quantities to isolate selection quality")
    print("="*80)
    
    # Test both implementations
    tests = [
        ('simple_active_learning_fraud_matched_quantities.py', 'Logistic Regression'),
        ('simple_active_learning_fraud_lgbm_matched_quantities.py', 'LightGBM')
    ]
    
    results = []
    
    for script, model_type in tests:
        success = run_single_test(script, model_type)
        results.append((script, model_type, success))
    
    # Summary
    print(f"\n{'='*80}")
    print("MATCHED QUANTITIES TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = 0
    for script, model_type, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_type:20} | {status}")
        if success:
            successful_tests += 1
    
    print(f"\nResults: {successful_tests}/{len(tests)} tests passed")
    
    if successful_tests == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Matched quantities methodology is working correctly")
        print("🚀 Ready to proceed with full experimental suite")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("🔧 Please review and fix issues before proceeding")
    
    print(f"\n📁 Results saved to: data/matched_quantities_results/")
    print(f"📊 Compare with previous results in: data/stratified_results/")

if __name__ == "__main__":
    main()
