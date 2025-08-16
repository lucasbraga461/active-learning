# 🔍 LightGBM Active Learning Configuration Analysis Report

**Generated on:** 2025-08-12 22:38:06
**Total log files found:** 102
**Model:** LightGBM (Gradient Boosting) with global feature standardization

---

## 📊 Results Summary

## 📈 All LightGBM Configuration Results

| Config | Active F1 | Passive F1 | Improvement | Improvement % | P-value | Significant |
|--------|------------|-------------|-------------|----------------|---------|-------------|
| 95 | 0.5395 | 0.5171 | 0.0224 | 4.33% | 0.001953 | Yes |
| 94 | 0.5329 | 0.5140 | 0.0189 | 3.68% | 0.009766 | Yes |
| 89 | 0.5329 | 0.5140 | 0.0189 | 3.68% | 0.009766 | Yes |
| 88 | 0.5327 | 0.5140 | 0.0187 | 3.64% | 0.027344 | Yes |
| 93 | 0.5327 | 0.5140 | 0.0187 | 3.64% | 0.027344 | Yes |
| 80 | 0.5433 | 0.5252 | 0.0181 | 3.45% | 0.027344 | Yes |
| 91 | 0.5292 | 0.5140 | 0.0152 | 2.96% | 0.083984 | No |
| 99 | 0.5292 | 0.5140 | 0.0152 | 2.96% | 0.083984 | No |
| 97 | 0.5292 | 0.5140 | 0.0152 | 2.96% | 0.083984 | No |
| 87 | 0.5292 | 0.5140 | 0.0152 | 2.96% | 0.083984 | No |
| 83 | 0.5367 | 0.5252 | 0.0115 | 2.19% | 0.160156 | No |
| 81 | 0.5082 | 0.5171 | -0.0089 | -1.72% | 0.130859 | No |
| 90 | 0.5069 | 0.5171 | -0.0102 | -1.97% | 0.105469 | No |
| 84 | 0.5069 | 0.5171 | -0.0102 | -1.97% | 0.105469 | No |
| 100 | 0.5069 | 0.5171 | -0.0102 | -1.97% | 0.105469 | No |
| 92 | 0.5114 | 0.5252 | -0.0138 | -2.63% | 0.105469 | No |
| 86 | 0.5114 | 0.5252 | -0.0138 | -2.63% | 0.105469 | No |
| 98 | 0.5114 | 0.5252 | -0.0138 | -2.63% | 0.105469 | No |
| 96 | 0.4977 | 0.5140 | -0.0163 | -3.17% | 0.027344 | Yes |

## 📊 Statistical Analysis

✅ **Statistically significant improvements:** 7
   **Configs:** [80, 88, 89, 93, 94, 95, 96]

### 🏆 Best LightGBM Performer

| Metric | Value |
|--------|-------|
| **Config** | 95 |
| **Active F1** | 0.5395 |
| **Passive F1** | 0.5171 |
| **Absolute Improvement** | 0.0224 |
| **Relative Improvement** | 4.33% |
| **P-value** | 0.001953 |
| **Statistically Significant** | Yes |

### 🥇 Top 5 LightGBM Performers

| Rank | Config | Active F1 | Passive F1 | Absolute Improvement | Relative Improvement % | P-value | Significant |
|------|--------|------------|-------------|----------------------|------------------------|---------|-------------|
| 1 | 95 | 0.5395 | 0.5171 | 0.0224 | 4.33% | 0.001953 | Yes |
| 2 | 94 | 0.5329 | 0.5140 | 0.0189 | 3.68% | 0.009766 | Yes |
| 3 | 89 | 0.5329 | 0.5140 | 0.0189 | 3.68% | 0.009766 | Yes |
| 4 | 88 | 0.5327 | 0.5140 | 0.0187 | 3.64% | 0.027344 | Yes |
| 5 | 93 | 0.5327 | 0.5140 | 0.0187 | 3.64% | 0.027344 | Yes |

## 💾 Data Export

**CSV Results:** `data/configuration_analysis_results_c80_100_lightgbm.csv`

## 🔍 Strategy Analysis

### 📊 Strategy Patterns in Top Performers

**⚠️ IMPORTANT NOTE ABOUT DUPLICATES:**
Configs 82 and 85 were found to be **EXACT DUPLICATES** of other configs.
They have been commented out in `run_configs_80_100.py` to avoid confusion.

**Note:** This analysis is based on actual configuration files and log data.

## 💡 Recommendations

### 🎯 Focus on These Statistically Significant LightGBM Configurations

- **Config 95:** Analyze strategy patterns
- **Config 94:** Analyze strategy patterns
- **Config 89:** Analyze strategy patterns
- **Config 88:** Analyze strategy patterns
- **Config 93:** Analyze strategy patterns
- **Config 80:** Analyze strategy patterns
- **Config 96:** Analyze strategy patterns

### 🔍 Next Steps

1. **Examine successful LightGBM configurations** for common patterns
2. **Create new configs** based on successful strategies
3. **Fine-tune LightGBM hyperparameters** for optimal performance

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Total LightGBM Configurations Analyzed** | 19 |
| **Statistically Significant Improvements** | 7 |
| **Best F1 Score** | 0.5395 (Config 95) |
| **Best Absolute Improvement** | 0.0224 |
| **Best Relative Improvement** | 4.33% |

---

## 🚀 LightGBM Active Learning Experiment Summary

**Experiment Goal:** Test LightGBM (gradient boosting) performance on various active learning strategies with global feature standardization.

**Key Features:**
- **Model:** LightGBM with optimized hyperparameters
- **Data:** Bank Marketing dataset with global standardization
- **Strategies:** Uncertainty, Diversity, and QBC sampling
- **Configurations:** 21 different strategy combinations (configs 80-100)

**Key Findings:**
- **Best Configuration:** Config 95 with F1 = 0.5395
- **Best Improvement:** 0.0224 (4.33% over passive learning)
- **Statistical Significance:** 7 out of 19 configs show significant improvements
