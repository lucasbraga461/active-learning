# 🔍 Active Learning Configuration Analysis Report (Configs 50-70)

**Generated on:** 2025-08-12 22:38:03
**Total log files found:** 102
**Model:** Logistic Regression WITH Regularization (C=0.1) and global feature standardization

---

## 📊 Results Summary

## 📈 All Configuration Results

| Config | Active F1 | Passive F1 | Improvement | Improvement % | P-value | Significant |
|--------|------------|-------------|-------------|----------------|---------|-------------|
| 62 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 58 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 68 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 50 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 70 | 0.5379 | 0.5056 | 0.0323 | 6.39% | 0.001953 | Yes |
| 59 | 0.5379 | 0.5056 | 0.0323 | 6.39% | 0.001953 | Yes |
| 64 | 0.5379 | 0.5056 | 0.0323 | 6.39% | 0.001953 | Yes |
| 60 | 0.5366 | 0.5056 | 0.0310 | 6.13% | 0.001953 | Yes |
| 56 | 0.5366 | 0.5056 | 0.0310 | 6.13% | 0.001953 | Yes |
| 65 | 0.5366 | 0.5056 | 0.0310 | 6.13% | 0.001953 | Yes |
| 66 | 0.5369 | 0.5124 | 0.0245 | 4.78% | 0.001953 | Yes |
| 52 | 0.5369 | 0.5124 | 0.0245 | 4.78% | 0.001953 | Yes |
| 55 | 0.5361 | 0.5128 | 0.0233 | 4.54% | 0.001953 | Yes |
| 57 | 0.5097 | 0.5056 | 0.0041 | 0.81% | 0.431641 | No |
| 67 | 0.5097 | 0.5056 | 0.0041 | 0.81% | 0.431641 | No |
| 51 | 0.5165 | 0.5128 | 0.0037 | 0.72% | 0.625000 | No |
| 69 | 0.5137 | 0.5128 | 0.0009 | 0.18% | 0.845703 | No |
| 63 | 0.5137 | 0.5128 | 0.0009 | 0.18% | 0.845703 | No |
| 61 | 0.5097 | 0.5124 | -0.0027 | -0.53% | 0.275391 | No |

## 📊 Statistical Analysis

✅ **Statistically significant improvements:** 13
   **Configs:** [50, 52, 55, 56, 58, 59, 60, 62, 64, 65, 66, 68, 70]

### 🏆 Best Performer

| Metric | Value |
|--------|-------|
| **Config** | 62 |
| **Active F1** | 0.5388 |
| **Passive F1** | 0.5056 |
| **Absolute Improvement** | 0.0332 |
| **Relative Improvement** | 6.57% |
| **P-value** | 0.001953 |
| **Statistically Significant** | Yes |

### 🥇 Top 5 Performers

| Rank | Config | Active F1 | Passive F1 | Absolute Improvement | Relative Improvement % | P-value | Significant |
|------|--------|------------|-------------|----------------------|------------------------|---------|-------------|
| 1 | 62 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 2 | 58 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 3 | 68 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 4 | 50 | 0.5388 | 0.5056 | 0.0332 | 6.57% | 0.001953 | Yes |
| 5 | 70 | 0.5379 | 0.5056 | 0.0323 | 6.39% | 0.001953 | Yes |

## 💾 Data Export

**CSV Results:** `data/configuration_analysis_results_c50_70_lr.csv`

## 🔍 Strategy Analysis

### 📊 Strategy Patterns in Top Performers

**⚠️ IMPORTANT NOTE ABOUT DUPLICATES:**
Configs 53 and 54 were found to be **EXACT DUPLICATES** of other configs.
They have been commented out in `run_configs_50_70.py` to avoid confusion.

**Note:** This analysis is based on actual configuration files and log data.

## 💡 Recommendations

### 🎯 Focus on These Statistically Significant Configurations

- **Config 62:** Analyze strategy patterns
- **Config 58:** Analyze strategy patterns
- **Config 68:** Analyze strategy patterns
- **Config 50:** Analyze strategy patterns
- **Config 70:** Analyze strategy patterns
- **Config 59:** Analyze strategy patterns
- **Config 64:** Analyze strategy patterns
- **Config 60:** Analyze strategy patterns
- **Config 56:** Analyze strategy patterns
- **Config 65:** Analyze strategy patterns
- **Config 66:** Analyze strategy patterns
- **Config 52:** Analyze strategy patterns
- **Config 55:** Analyze strategy patterns

### 🔍 Next Steps

1. **Examine successful configurations** for common patterns
2. **Create new configs** based on successful strategies
3. **Fine-tune parameters** for optimal performance

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Configurations Analyzed** | 19 |
| **Statistically Significant Improvements** | 13 |
| **Best F1 Score** | 0.5388 (Config 62) |
| **Best Absolute Improvement** | 0.0332 |
| **Best Relative Improvement** | 6.57% |

---

## 🚀 Active Learning Experiment Summary (Configs 50-70)

**Experiment Goal:** Test various active learning strategies with Logistic Regression using global feature standardization.

**Key Features:**
- **Model:** Logistic Regression with regularization (C=0.1)
- **Data:** Bank Marketing dataset with global standardization
- **Strategies:** Uncertainty, Diversity, and QBC sampling
- **Configurations:** 21 different strategy combinations (configs 50-70)

**Key Findings:**
- **Best Configuration:** Config 62 with F1 = 0.5388
- **Best Improvement:** 0.0332 (6.57% over passive learning)
- **Statistical Significance:** 13 out of 19 configs show significant improvements

**Technical Details:**
- **Regularization:** Strong L2 regularization (C=0.1) to prevent overfitting
- **Global Standardization:** Z-score normalization of numerical features
- **Feature Engineering:** Numerical age/balance with log transformation
