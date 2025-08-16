#!/usr/bin/env python3
"""
Run Multiple Active Learning Configurations (Configs 20-40)

This script runs 21 different active learning configurations: 20 new ones (config20-config39) 
plus the current champion config17 as config40. Each configuration is designed based on
insights from our analysis of why config17 succeeded where config16 failed.

Key Insights:
- Late iterations (9-11) are critical for final performance
- Uncertainty sampling in late iterations maintains learning momentum
- Diversity sampling should be used strategically in middle iterations
- QBC should be used in the final iteration for ensemble disagreement
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add the current directory to Python path to import simple_active_learning
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_config_file(config_number, config_dict):
    """Create a temporary config file for the given configuration"""
    
    # Read the original file
    with open('simple_active_learning.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create the new config section
    new_config = f"""    EXPERIMENT_CONFIG = {{
        'initial_samples': {config_dict['initial_samples']},
        'initial_strategy': '{config_dict['initial_strategy']}',
        'batch_size': {config_dict['batch_size']},
        'n_iterations': {config_dict['n_iterations']},
        'use_numerical_features': {config_dict['use_numerical_features']},
        'iteration_strategies': {config_dict['iteration_strategies']}
    }}
    CONFIG_NAME = "config{config_number}"
"""
    
    # Find and replace the existing config
    start_marker = "    EXPERIMENT_CONFIG = {"
    end_marker = "    CONFIG_NAME ="
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"❌ Error: Could not find EXPERIMENT_CONFIG in simple_active_learning.py")
        return False
    
    # Find the end of the config section
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print(f"❌ Error: Could not find CONFIG_NAME in simple_active_learning.py")
        return False
    
    # Find the end of the line containing CONFIG_NAME
    end_line_idx = content.find('\n', end_idx)
    if end_line_idx == -1:
        end_line_idx = len(content)
    
    # Replace the entire config section
    new_content = content[:start_idx] + new_config + content[end_line_idx:]
    
    # Write the modified file
    with open('simple_active_learning.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def run_config(config_number, config_dict):
    """Run a single configuration"""
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING CONFIG {config_number}")
    print(f"{'='*80}")
    print(f"Initial samples: {config_dict['initial_samples']}")
    print(f"Initial strategy: {config_dict['initial_strategy']}")
    print(f"Batch size: {config_dict['batch_size']}")
    print(f"Iterations: {config_dict['n_iterations']}")
    print(f"Feature type: {'Numerical' if config_dict['use_numerical_features'] else 'Binned'}")
    print(f"Strategies: {config_dict['iteration_strategies']}")
    print(f"{'='*80}")
    
    # Create the config file
    if not create_config_file(config_number, config_dict):
        return False
    
    # Run the configuration
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, 'simple_active_learning.py'], 
                              capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Config {config_number} completed successfully in {duration:.1f} seconds")
            
            # Check if the log file was created
            log_dir = "data/logs"
            if os.path.exists(log_dir):
                log_files = [f for f in os.listdir(log_dir) if f.startswith(f'experiment_log_config{config_number}_')]
                if log_files:
                    print(f"📝 Log saved: {log_files[0]}")
                else:
                    print("⚠️  Warning: No log file found")
            
            return True
        else:
            print(f"❌ Config {config_number} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Config {config_number} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Config {config_number} failed with error: {e}")
        return False

def main():
    """Main function to run all configurations"""
    
    print("🎯 ACTIVE LEARNING CONFIGURATION OPTIMIZATION")
    print("="*80)
    print("Running 18 unique configurations: configs 20,21,22,23,24,25,26,27,29,30,31,33,34,35,36,37,39,40")
    print("NOTE: Configs 28,32,38 are duplicates of config23 and have been commented out")
    print("Based on analysis: Late iterations (9-11) are critical for performance")
    print("="*80)
    
    # Define all configurations
    configs = {
        20: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty (like config17)
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        21: {
            'initial_samples': 300,
            'initial_strategy': 'diversity',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'diversity',        # 2: Build diversity
                'uncertainty',      # 3: Add uncertainty
                'diversity',        # 4: Strategic diversity
                'uncertainty',      # 5: Critical uncertainty
                'diversity',        # 6: Strategic diversity
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        22: {
            'initial_samples': 400,
            'initial_strategy': 'random',
            'batch_size': 60,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        23: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'uncertainty',      # 3: Pure uncertainty
                'uncertainty',      # 4: Maintain uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Back to uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        24: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'diversity',        # 2: Early diversity
                'uncertainty',      # 3: Back to uncertainty
                'uncertainty',      # 4: Build momentum
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        25: {
            'initial_samples': 350,
            'initial_strategy': 'random',
            'batch_size': 65,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        26: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'uncertainty',      # 3: Pure uncertainty
                'diversity',        # 4: Strategic diversity
                'uncertainty',      # 5: Back to uncertainty
                'diversity',        # 6: Strategic diversity
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        27: {
            'initial_samples': 300,
            'initial_strategy': 'diversity',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        # 28: DUPLICATE OF CONFIG 23 - COMMENTED OUT TO AVOID CONFUSION
        # {
        #     'initial_samples': 300,
        #     'initial_strategy': 'random',
        #     'batch_size': 68,
        #     'n_iterations': 11,
        #     'use_numerical_features': True,
        #     'iteration_strategies': [
        #         'uncertainty',      # 1: Start with uncertainty
        #         'uncertainty',      # 2: Build momentum
        #         'uncertainty',      # 3: Pure uncertainty
        #         'uncertainty',      # 4: Maintain uncertainty
        #         'diversity',        # 5: Strategic diversity
        #         'uncertainty',      # 6: Back to uncertainty
        #         'uncertainty',      # 7: Critical iteration - uncertainty
        #         'diversity',        # 8: Strategic diversity
        #         'uncertainty',      # 9: CRITICAL - uncertainty
        #         'uncertainty',      # 10: Maintain momentum
        #         'qbc',             # 11: Final ensemble disagreement
        #     ]
        # },
        
        29: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'diversity',        # 2: Early diversity
                'uncertainty',      # 3: Back to uncertainty
                'uncertainty',      # 4: Build momentum
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        30: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'uncertainty',      # 3: Pure uncertainty
                'diversity',        # 4: Strategic diversity
                'uncertainty',      # 5: Back to uncertainty
                'diversity',        # 6: Strategic diversity
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        31: {
            'initial_samples': 400,
            'initial_strategy': 'diversity',
            'batch_size': 60,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        # 32: DUPLICATE OF CONFIG 23 - COMMENTED OUT TO AVOID CONFUSION
        # {
        #     'initial_samples': 300,
        #     'initial_strategy': 'random',
        #     'batch_size': 68,
        #     'n_iterations': 11,
        #     'use_numerical_features': True,
        #     'iteration_strategies': [
        #         'uncertainty',      # 1: Start with uncertainty
        #         'uncertainty',      # 2: Build momentum
        #         'uncertainty',      # 3: Pure uncertainty
        #         'uncertainty',      # 4: Maintain uncertainty
        #         'diversity',        # 5: Strategic diversity
        #         'uncertainty',      # 6: Back to uncertainty
        #         'uncertainty',      # 7: Critical iteration - uncertainty
        #         'diversity',        # 8: Strategic diversity
        #         'uncertainty',      # 9: CRITICAL - uncertainty
        #         'uncertainty',      # 10: Maintain momentum
        #         'qbc',             # 11: Final ensemble disagreement
        #     ]
        # },
        
        33: {
            'initial_samples': 350,
            'initial_strategy': 'diversity',
            'batch_size': 65,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        34: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'diversity',        # 2: Early diversity
                'uncertainty',      # 3: Back to uncertainty
                'uncertainty',      # 4: Build momentum
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        35: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'uncertainty',      # 3: Pure uncertainty
                'diversity',        # 4: Strategic diversity
                'uncertainty',      # 5: Back to uncertainty
                'diversity',        # 6: Strategic diversity
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        36: {
            'initial_samples': 400,
            'initial_strategy': 'random',
            'batch_size': 60,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        37: {
            'initial_samples': 300,
            'initial_strategy': 'diversity',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },
        
        # 38: DUPLICATE OF CONFIG 23 - COMMENTED OUT TO AVOID CONFUSION
        # {
        #     'initial_samples': 300,
        #     'initial_strategy': 'random',
        #     'batch_size': 68,
        #     'n_iterations': 11,
        #     'use_numerical_features': True,
        #     'iteration_strategies': [
        #         'uncertainty',      # 1: Start with uncertainty
        #         'uncertainty',      # 2: Build momentum
        #         'uncertainty',      # 3: Pure uncertainty
        #         'uncertainty',      # 4: Maintain uncertainty
        #         'diversity',        # 5: Strategic diversity
        #         'uncertainty',      # 6: Back to uncertainty
        #         'uncertainty',      # 7: Critical iteration - uncertainty
        #         'diversity',        # 8: Strategic diversity
        #         'uncertainty',      # 9: CRITICAL - uncertainty
        #         'uncertainty',      # 10: Maintain momentum
        #         'qbc',             # 11: Final ensemble disagreement
        #     ]
        # },
        
        39: {
            'initial_samples': 350,
            'initial_strategy': 'diversity',
            'batch_size': 65,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'diversity',        # 3: Add diversity
                'uncertainty',      # 4: Back to uncertainty
                'diversity',        # 5: Strategic diversity
                'uncertainty',      # 6: Maintain uncertainty
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        },

        40: {
            'initial_samples': 300,
            'initial_strategy': 'random',
            'batch_size': 68,
            'n_iterations': 11,
            'use_numerical_features': True,
            'iteration_strategies': [
                'uncertainty',      # 1: Start with uncertainty
                'uncertainty',      # 2: Build momentum
                'uncertainty',      # 3: Maintain uncertainty
                'uncertainty',      # 4: Build momentum
                'diversity',        # 5: Strategic diversity
                'diversity',        # 6: Strategic diversity
                'uncertainty',      # 7: Critical iteration - uncertainty
                'diversity',        # 8: Strategic diversity
                'uncertainty',      # 9: CRITICAL - uncertainty (like config17)
                'uncertainty',      # 10: Maintain momentum
                'qbc',             # 11: Final ensemble disagreement
            ]
        }
    }

    # Track results
    successful_configs = []
    failed_configs = []

    # Run all configurations
    for config_num in range(20, 41):  # Changed to 41 to include config40
        if config_num in configs:
            success = run_config(config_num, configs[config_num])
            if success:
                successful_configs.append(config_num)
            else:
                failed_configs.append(config_num)
            
            # Brief pause between configs
            time.sleep(2)

    # Summary
    print(f"\n{'='*80}")
    print("🎯 CONFIGURATION RUN SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful configurations: {len(successful_configs)}")
    print(f"❌ Failed configurations: {len(failed_configs)}")

    if successful_configs:
        print(f"✅ Successful: {successful_configs}")

    if failed_configs:
        print(f"❌ Failed: {failed_configs}")

    print(f"\n📁 All results saved in data/ folder with respective config numbers")
    print(f"📊 Config40 = config17 (current champion) for easy comparison")
    print(f"🎯 Look for F1 scores higher than config17's 0.5184 ± 0.0086")

    print(f"\n{'='*80}")
    print("🚀 CONFIGURATION OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 