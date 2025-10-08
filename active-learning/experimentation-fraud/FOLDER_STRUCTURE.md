# 📁 Experimentation-Fraud Folder Structure

This document describes the organized structure of the Active Learning fraud detection experiments.

## 🗂️ **Directory Structure**

```
experimentation-fraud/
├── 📂 scripts/                     # All experimental scripts
│   ├── 📂 core/                    # Core experimental implementations
│   │   ├── simple_active_learning_fraud.py                    # Main LR implementation
│   │   ├── simple_active_learning_fraud_lgbm.py              # LightGBM implementation  
│   │   ├── simple_active_learning_fraud_matched_quantities.py # LR with matched quantities
│   │   ├── simple_active_learning_fraud_lgbm_matched_quantities.py # LGBM with matched quantities
│   │   └── simple_active_learning_fraud_parallel_fair.py     # Fair parallel comparison
│   ├── 📂 analysis/                # Analysis and testing scripts
│   │   ├── comprehensive_iteration_analysis.py               # Full 10×11 analysis
│   │   ├── iteration_by_iteration_analysis.py               # Detailed iteration tracking
│   │   └── iteration_by_iteration_analysis.ipynb           # Jupyter analysis notebook
│   └── 📂 configurations/          # Configuration runners
│       ├── run_all_configs_fair_parallel.py                 # All 15 configs runner
│       ├── run_fraud_configs_best.py                        # Best configs from Bank experiments
│       ├── run_fraud_experiments.py                         # Simple experiment runner
│       └── run_matched_quantities_test.py                   # Matched quantities tester
├── 📂 results/                     # All experimental results
│   ├── 📂 current/                 # Latest valid results
│   │   ├── comprehensive_final_results.csv                  # Final aggregated performance
│   │   ├── comprehensive_iteration_analysis.csv            # Complete iteration tracking  
│   │   ├── iteration_by_iteration_analysis.csv            # Detailed progression data
│   │   ├── 📂 logs/                # Execution logs
│   │   ├── 📂 matched_quantities_results/  # Matched quantities experiments
│   │   └── 📂 stratified_results/  # Stratified sampling experiments
│   └── 📂 archive/                 # Historical/deprecated results
│       ├── simple_active_learning_fraud_deprecated.py      # Old problematic version
│       └── 📂 unfair_passive_learning_results_20250917/   # Invalid results (data leakage)
├── 📂 documentation/               # Research documentation
│   ├── RESEARCH_FINDINGS.md       # 📊 Comprehensive research findings
│   └── README.md                   # Project overview and setup
├── 📂 model/                      # Saved models (if any)
└── FOLDER_STRUCTURE.md            # This file
```

## 🎯 **Key Files by Purpose**

### **🔬 Main Experiments**
- **`scripts/core/simple_active_learning_fraud.py`** - Primary LR experiments with fair comparison
- **`scripts/core/simple_active_learning_fraud_lgbm.py`** - LightGBM version of main experiments
- **`scripts/analysis/comprehensive_iteration_analysis.py`** - Complete 10 runs × 11 iterations analysis

### **📊 Critical Results**
- **`results/current/comprehensive_final_results.csv`** - Final performance summary (THE main results)
- **`results/current/comprehensive_iteration_analysis.csv`** - Detailed volatility analysis
- **`documentation/RESEARCH_FINDINGS.md`** - 📋 Complete research findings & conclusions

### **⚙️ Configuration Runners**
- **`scripts/configurations/run_all_configs_fair_parallel.py`** - Runs all 15 Bank experiment configs
- **`scripts/configurations/run_fraud_configs_best.py`** - Runs top 5 configs from each category

### **📈 Analysis Tools**
- **`scripts/analysis/iteration_by_iteration_analysis.py`** - Tracks learning volatility
- **`scripts/analysis/comprehensive_iteration_analysis.py`** - Full experimental pipeline

## 🚫 **Deprecated/Archive**

### **❌ Invalid Results (results/archive/)**
- `unfair_passive_learning_results_20250917/` - Results with experimental bias (DO NOT USE)
- `simple_active_learning_fraud_deprecated.py` - Old version with flawed passive learning

### **⚠️ Why Archived Results Are Invalid**
The archived results used **unfair passive learning strategies** that:
- Used stratified sampling with artificially high fraud rates (10% vs natural 0.173%)
- Created data leakage and temporal bias
- Generated misleading 456% improvement claims

## 🎯 **Usage Guide**

### **For Research/Analysis:**
1. Read: `documentation/RESEARCH_FINDINGS.md`
2. Review: `results/current/comprehensive_final_results.csv`
3. Explore: `scripts/analysis/comprehensive_iteration_analysis.py`

### **For Running New Experiments:**
1. Single config: `scripts/core/simple_active_learning_fraud.py`
2. Multiple configs: `scripts/configurations/run_all_configs_fair_parallel.py`
3. Analysis: `scripts/analysis/iteration_by_iteration_analysis.py`

### **For Understanding Methodology:**
1. Fair comparison: `scripts/core/simple_active_learning_fraud_parallel_fair.py`
2. Matched quantities: `scripts/core/simple_active_learning_fraud_matched_quantities.py`
3. Iteration tracking: `scripts/analysis/comprehensive_iteration_analysis.py`

## ✅ **Quality Assurance**

All scripts in `scripts/core/` and `scripts/analysis/` implement:
- ✅ Fair parallel comparison (no shared data)
- ✅ No temporal bias  
- ✅ No data leakage
- ✅ Matched quantities for fair comparison
- ✅ Research-grade experimental rigor

**Use only files in `scripts/` and `results/current/` for valid research.**
