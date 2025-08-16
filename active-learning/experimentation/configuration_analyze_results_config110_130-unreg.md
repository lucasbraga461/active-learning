# 🔍 Unregularized Logistic Regression Active Learning Configuration Analysis Report

**Generated on:** 2025-08-12 22:38:08
**Total log files found:** 102
**Model:** Logistic Regression WITHOUT Regularization (C=1.0, default parameters)

---

## 📊 Results Summary

## 📈 All Unregularized Logistic Regression Configuration Results

| Config | Active F1 | Passive F1 | Improvement | Improvement % | P-value | Significant |
|--------|------------|-------------|-------------|----------------|---------|-------------|
| 124 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 118 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 126 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 130 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 110 | 0.5275 | 0.5013 | 0.0262 | 5.23% | 0.001953 | Yes |
| 121 | 0.5275 | 0.5013 | 0.0262 | 5.23% | 0.001953 | Yes |
| 120 | 0.5259 | 0.5013 | 0.0246 | 4.91% | 0.001953 | Yes |
| 128 | 0.5259 | 0.5013 | 0.0246 | 4.91% | 0.001953 | Yes |
| 115 | 0.5295 | 0.5077 | 0.0218 | 4.29% | 0.009766 | Yes |
| 122 | 0.5290 | 0.5111 | 0.0179 | 3.50% | 0.005859 | Yes |
| 112 | 0.5290 | 0.5111 | 0.0179 | 3.50% | 0.005859 | Yes |
| 114 | 0.4977 | 0.5013 | -0.0036 | -0.72% | 0.695312 | No |
| 123 | 0.4900 | 0.5013 | -0.0113 | -2.25% | 0.064453 | No |
| 125 | 0.4932 | 0.5077 | -0.0145 | -2.86% | 0.064453 | No |
| 119 | 0.4932 | 0.5077 | -0.0145 | -2.86% | 0.064453 | No |
| 129 | 0.4932 | 0.5077 | -0.0145 | -2.86% | 0.064453 | No |
| 117 | 0.4961 | 0.5111 | -0.0150 | -2.93% | 0.003906 | Yes |
| 127 | 0.4961 | 0.5111 | -0.0150 | -2.93% | 0.003906 | Yes |
| 111 | 0.4916 | 0.5077 | -0.0161 | -3.17% | 0.005859 | Yes |

## 📊 Statistical Analysis

✅ **Statistically significant improvements:** 14
   **Configs:** [110, 111, 112, 115, 117, 118, 120, 121, 122, 124, 126, 127, 128, 130]

### 🏆 Best Unregularized LR Performer

| Metric | Value |
|--------|-------|
| **Config** | 124 |
| **Active F1** | 0.5282 |
| **Passive F1** | 0.5013 |
| **Absolute Improvement** | 0.0269 |
| **Relative Improvement** | 5.37% |
| **P-value** | 0.001953 |
| **Statistically Significant** | Yes |

### 🥇 Top 5 Unregularized LR Performers

| Rank | Config | Active F1 | Passive F1 | Absolute Improvement | Relative Improvement % | P-value | Significant |
|------|--------|------------|-------------|----------------------|------------------------|---------|-------------|
| 1 | 124 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 2 | 118 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 3 | 126 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 4 | 130 | 0.5282 | 0.5013 | 0.0269 | 5.37% | 0.001953 | Yes |
| 5 | 110 | 0.5275 | 0.5013 | 0.0262 | 5.23% | 0.001953 | Yes |

## 💾 Data Export

**CSV Results:** `data/configuration_analysis_results_config110_130_unreg.csv`

## 🔍 Strategy Analysis

### 📊 Strategy Patterns in Top Performers

**⚠️ IMPORTANT NOTE ABOUT DUPLICATES:**
Configs 113 and 116 were found to be **EXACT DUPLICATES** of other configs.
They have been commented out in `run_configs_110_130.py` to avoid confusion.

**Note:** This analysis is based on actual configuration files and log data.

## 💡 Recommendations

### 🎯 Focus on These Statistically Significant Unregularized LR Configurations

- **Config 124:** Analyze strategy patterns
- **Config 118:** Analyze strategy patterns
- **Config 126:** Analyze strategy patterns
- **Config 130:** Analyze strategy patterns
- **Config 110:** Analyze strategy patterns
- **Config 121:** Analyze strategy patterns
- **Config 120:** Analyze strategy patterns
- **Config 128:** Analyze strategy patterns
- **Config 115:** Analyze strategy patterns
- **Config 122:** Analyze strategy patterns
- **Config 112:** Analyze strategy patterns
- **Config 117:** Analyze strategy patterns
- **Config 127:** Analyze strategy patterns
- **Config 111:** Analyze strategy patterns

### 🔍 Next Steps

1. **Examine successful unregularized LR configurations** for common patterns
2. **Create new configs** based on successful strategies
3. **Fine-tune model parameters** for optimal performance

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Unregularized LR Configurations Analyzed** | 19 |
| **Statistically Significant Improvements** | 14 |
| **Best F1 Score** | 0.5282 (Config 124) |
| **Best Absolute Improvement** | 0.0269 |
| **Best Relative Improvement** | 5.37% |

---

## 🚀 Unregularized Logistic Regression Active Learning Experiment Summary

**Experiment Goal:** Test whether removing regularization from Logistic Regression can improve performance on various active learning strategies.

**Key Features:**
- **Model:** Logistic Regression WITHOUT regularization (C=1.0, default parameters)
- **Data:** Bank Marketing dataset with global standardization
- **Strategies:** Uncertainty, Diversity, and QBC sampling
- **Configurations:** 21 different strategy combinations (configs 110-130)

**Key Findings:**
- **Best Configuration:** Config 124 with F1 = 0.5282
- **Best Improvement:** 0.0269 (5.37% over passive learning)
- **Statistical Significance:** 14 out of 19 configs show significant improvements

**Technical Details:**
- **Regularization:** None (C=1.0) - allows full model flexibility
- **Solver:** 'lbfgs' - better for unregularized optimization
- **Global Standardization:** Ensures fair comparison with regularized versions
