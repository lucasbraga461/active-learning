# 🧠 Comprehensive Active Learning & Multi-LLM Research Repository

**Companion Repository for the IEEE Access Publication:**  
*"Active Learning for Imbalanced Classification: Empirical Insights, Iteration Scheduling, and LLM-Augmented Validation"*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Access](https://img.shields.io/badge/Published-IEEE%20Access%202025-blue.svg)](https://ieeexplore.ieee.org/)

## 📚 Access the Research

📖 **IEEE Access Publication**: [IEEE Xplore](https://ieeexplore.ieee.org/document/11215701) *(DOI: 10.1109/ACCESS.2025.3624650)*  
🧪 **Research Repository**: [GitHub - Active Learning](https://github.com/lucasbraga461/active-learning)  
📊 **Datasets & Code**: [IEEE DataPort - DOI: 10.21227/29cz-j345](https://dx.doi.org/10.21227/29cz-j345)  
👤 **Connect with Author**: [LinkedIn - Lucas Braga](https://linkedin.com/in/lucasbraga461) | [ORCID](https://orcid.org/0009-0007-5397-5652)

### Citation
```bibtex
@article{braga2025activelearning,
  title={{Active Learning for Imbalanced Classification: Empirical Insights, Iteration Scheduling, and LLM-Augmented Validation}},
  author={Benevides e Braga, Lucas},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE},
  doi={10.1109/ACCESS.2025.3624650},
  url={https://ieeexplore.ieee.org/document/11215701},
  note={Code and data available at: https://github.com/lucasbraga461/active-learning}
}
```

### 📊 Research Datasets
- **🏦 UCI Bank Marketing**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (45,211 samples, 11% positive class)
- **🚨 European Credit Card**: Fraud detection dataset (284,807 transactions, 0.173% fraud rate)  
- **🤖 NYC Restaurant Dataset**: Business validation dataset (115 samples with ground truth)
- **🎲 Synthetic Datasets**: [IEEE DataPort - DOI: 10.21227/29cz-j345](https://dx.doi.org/10.21227/29cz-j345)

---

## 📖 About This Repository

A comprehensive multi-domain research project demonstrating advanced Active Learning strategies across **banking**, **fraud detection**, and **business validation** domains, plus a complete **multi-LLM validation framework**. This repository provides all practical code examples, experimental configurations, and analysis scripts from the IEEE Access research paper.

## 🏆 Key Research Achievements

### 📊 **Banking Active Learning** (UCI Bank Marketing)
- **🎯 Champion Strategy**: Config 62 achieves **6.57% F1 improvement** over passive learning
- **📈 Comprehensive Analysis**: 75 unique configurations tested across 4 experimental ranges
- **🔬 Statistical Rigor**: 750 total runs (10 runs × 75 configurations)
- **🏭 Production-Validated**: 4-1-2-1-2-1 strategy pattern proven optimal

### 🚨 **Fraud Detection Active Learning** (European Credit Card)  
- **⚡ Extraordinary Performance**: **300-600% F1 improvements** for Logistic Regression
- **🛡️ Robust Validation**: 550 controlled experiments with rigorous anti-bias measures
- **🎯 Model Comparison**: LightGBM stable (84-136% gains), LR extreme but volatile
- **📊 Massive Scale**: 284,807 transactions, 0.173% fraud rate

### 🤖 **Multi-LLM Business Validation** (NYC Restaurants)
- **🥇 Champion Provider**: Perplexity AI achieves **97.6% F1-score** 
- **⚖️ Comprehensive Comparison**: OpenAI GPT-4o, Perplexity Sonar, Google Gemini
- **💰 Cost Analysis**: Complete speed/cost/accuracy trade-off evaluation
- **🏗️ Production Framework**: Extensible multi-provider architecture

---

## 📌 What's Inside

### 🏦 **Banking Active Learning Research** (`active-learning/experimentation/`)
**The original 75-configuration study on UCI Bank Marketing Dataset:**
- **3 Sampling Strategies**: Uncertainty Sampling, Diversity Sampling (k-NN), Query-by-Committee (QBC)
- **3 Model Types**: Logistic Regression (regularized/unregularized), LightGBM  
- **2 Feature Engineering Approaches**: Binned vs. standardized features
- **Champion Strategy**: 4-1-2-1-2-1 pattern (uncertainty-diversity-uncertainty-diversity-uncertainty-qbc)
- **Complete Results**: [`COMPREHENSIVE_EXPERIMENT_REPORT.md`](active-learning/experimentation/COMPREHENSIVE_EXPERIMENT_REPORT.md)

### 🚨 **Fraud Detection Research** (`active-learning/experimentation-fraud/`)
**Breakthrough active learning results on highly imbalanced fraud detection:**
- **Dataset**: European Credit Card (284,807 transactions, 0.173% fraud rate)
- **Extreme Improvements**: 300-600% F1 gains for Logistic Regression, 84-136% for LightGBM
- **Fair Parallel Comparison**: Rigorous methodology preventing data leakage and bias
- **Comprehensive Results**: [`RESEARCH_FINDINGS.md`](active-learning/experimentation-fraud/RESEARCH_FINDINGS.md)

### 🤖 **Multi-LLM Business Validation** (`llm-business-validation/`)
**Production-ready validation system comparing major LLM providers:**
- **Provider-Agnostic Architecture**: OpenAI, Perplexity, Gemini support
- **Robust JSON Parsing**: Handles malformed responses gracefully
- **Batched Processing**: Concurrent validation with progress tracking
- **Performance Analysis**: Complete cost/speed/accuracy evaluation
- **CLI Interface**: Simple command-line operation

---

## 🔍 Research Findings Summary

### **🏦 Banking Active Learning (Config 62 Champion)**
- **Performance**: 6.57% F1 improvement over passive learning
- **Model**: Logistic Regression with regularization (C=0.1)  
- **Features**: Globally standardized numerical features
- **Strategy Pattern**: `uncertainty(4) → diversity(1) → uncertainty(2) → diversity(1) → uncertainty(2) → qbc(1)`

### **🚨 Fraud Detection Breakthrough Results**

| Model Type | Best Config | Active F1 | Passive F1 | Improvement | Stability |
|------------|-------------|-----------|------------|-------------|-----------|
| **LightGBM** | 2001 | 0.8259 | 0.3855 | **+114.2%** | ±0.0155 (stable) |
| **Logistic Regression** | 1005 | 0.7591 | 0.1063 | **+614.1%** | ±0.0105 (volatile) |

### **🤖 Multi-LLM Performance Comparison**

| Provider | F1-Score | Speed | Cost | Best For |
|----------|----------|--------|------|----------|
| **Perplexity** | **97.6%** | 4.2s/req | $0.58 | Production (best accuracy) |
| **OpenAI** | 79.8% | 2.4s/req | $0.35 | Budget-conscious |
| **Gemini** | 65.8% | 2.7s/req | $0.17 | Cost-sensitive |

---

## 📁 Repository Structure

```
active-learning/
├── active-learning/                          # 🏦 Banking & Fraud Detection Research
│   ├── experimentation/                      # 📊 UCI Bank Marketing (75 configs)
│   │   ├── COMPREHENSIVE_EXPERIMENT_REPORT.md      # Complete results analysis
│   │   ├── simple_active_learning*.py              # Core experiment scripts
│   │   ├── run_configs_*.py                        # Configuration runners
│   │   ├── data/logs/                              # Detailed experiment logs
│   │   ├── data/*_comparison_*.png                 # Performance visualizations
│   │   └── configuration_analysis_report_*.md      # Range-specific reports
│   ├── experimentation-fraud/               # 🚨 Credit Card Fraud (550 experiments)
│   │   ├── RESEARCH_FINDINGS.md                   # Breakthrough results analysis  
│   │   ├── scripts/core/                          # AL implementations
│   │   ├── scripts/configurations/                # Experiment runners
│   │   ├── scripts/analysis/                      # Performance analysis
│   │   └── results/                               # Comprehensive results
│   ├── data/                                # 📊 Datasets
│   │   ├── uci_dataset_00222_bank/               # Bank Marketing Dataset
│   │   ├── european-credit-card-dataset/         # Credit Card Fraud Dataset
│   │   └── paysim-dataset/                       # Additional fraud data
│   ├── helpers/                             # 🛠️ Utility functions
│   ├── model/                               # 💾 Saved models
│   └── requirements.txt                     # 📦 Dependencies
├── llm-business-validation/                  # 🤖 Multi-LLM Research
│   ├── shared/                              # 🔧 Common components
│   │   ├── scripts/multi_llm_validator.py         # Multi-provider runner
│   │   ├── scripts/evaluate_performance.py        # Performance analysis
│   │   └── data/nyc_restaurants_sample_115.csv    # Test dataset
│   ├── experiments/                         # 🧪 Provider-specific experiments
│   │   ├── openai/openai_client.py               # OpenAI GPT-4o implementation
│   │   ├── perplexity/perplexity_client.py       # Perplexity Sonar implementation  
│   │   ├── gemini/gemini_client.py               # Google Gemini implementation
│   │   └── */results/                            # Provider results & analysis
│   ├── COMPREHENSIVE_LLM_COMPARISON.md      # 📋 Complete LLM comparison results
│   └── requirements.txt                     # 📦 LLM dependencies
└── README.md                               # 📖 This file
```

---

## 🚀 Quick Start

### **Option 1: Banking Active Learning (Champion Strategy)**
```bash
# Install dependencies
pip install -r active-learning/requirements.txt

# Run the champion configuration (Config 62)
cd active-learning/experimentation
python simple_active_learning.py
# Manually set CONFIG_NAME = "config62" in the script
```

### **Option 2: Fraud Detection Active Learning**
```bash
# Install dependencies  
pip install -r active-learning/requirements.txt

# Run comprehensive fraud detection experiments
cd active-learning/experimentation-fraud/scripts/configurations
python final_lr_comprehensive.py      # Logistic Regression experiments
python final_lgbm_comprehensive.py    # LightGBM experiments
```

### **Option 3: Multi-LLM Business Validation**
```bash
# Install LLM dependencies
pip install -r llm-business-validation/shared/requirements.txt

# Set up API keys in .env file
export OPENAI_API_KEY=sk-...
export PERPLEXITY_API_KEY=pplx-...
export GOOGLE_API_KEY=...

# Test individual providers
cd llm-business-validation/experiments/perplexity
python perplexity_client.py

# Run multi-provider comparison  
cd llm-business-validation
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers perplexity openai gemini
```

---

## 🧪 Experimental Configurations

### **Banking Active Learning Ranges**
- **Configs 20-41**: Baseline Logistic Regression with binned features (3.95% improvement)
- **Configs 50-70**: Standardized features with regularization (**6.57% champion**)  
- **Configs 80-100**: LightGBM experimentation (4.33% improvement)
- **Configs 110-130**: Unregularized Logistic Regression (5.37% improvement)

### **Fraud Detection Configurations**  
- **LR Configurations (1002-1005)**: 291-614% improvements, high volatility
- **LightGBM Configurations (2001-2005)**: 84-136% improvements, high stability
- **Fair Parallel Methodology**: Eliminates data leakage and temporal bias

### **Multi-LLM Validation Providers**
- **OpenAI GPT-4o**: 79.8% F1-score, 2.4s/request, $0.35 cost  
- **Perplexity Sonar**: 97.6% F1-score, 4.2s/request, $0.58 cost (**Winner**)
- **Google Gemini**: 65.8% F1-score, 2.7s/request, $0.17 cost

---

## 📊 Key Research Insights

### **🎯 Active Learning Strategy Optimization**
1. **Feature standardization is more important than model complexity**
2. **Uncertainty sampling dominates successful strategies (70-80% of iterations)**  
3. **Strategic diversity placement prevents overfitting**
4. **QBC finale provides ensemble-based final refinement**
5. **4-1-2-1-2-1 pattern consistently outperforms other strategies**

### **🚨 Fraud Detection Breakthroughs**
1. **Active Learning excels on highly imbalanced data (0.173% fraud rate)**
2. **Extreme imbalance amplifies AL benefits (300-600% improvements possible)**
3. **Model choice determines risk-reward profile (stable vs extreme gains)**
4. **Fair parallel comparison essential for valid results**

### **🤖 LLM Provider Selection**  
1. **Perplexity dominates with web search capabilities (97.6% F1-score)**
2. **OpenAI provides good balance of cost and performance**
3. **Gemini offers cost-effective option for budget-conscious applications**
4. **Provider-agnostic architecture enables easy switching**

---

## 🛠️ Advanced Usage

### **Custom Banking Experiments**
```python
# Example: Create custom banking configuration
CONFIG = {
    'name': 'custom_config',
    'initial_samples': 300,
    'batch_size': 68,
    'strategies': ['uncertainty', 'uncertainty', 'diversity', 'uncertainty', 'qbc'],
    'model_type': 'logistic_regression',
    'standardize_features': True,
    'regularization': 0.1
}
```

### **Fraud Detection Analysis**
```bash
# Run comprehensive fraud analysis  
cd active-learning/experimentation-fraud/scripts/analysis
python comprehensive_iteration_analysis.py
```

### **LLM Provider Extension**
```python
# Example: Add new LLM provider
class NewProviderClient(BaseLLMClient):
    def _get_api_key(self):
        return os.getenv('NEW_PROVIDER_API_KEY')
    
    def _call_api(self, prompt):
        # Implement provider-specific API call
        pass
```

---

## 🚀 Production Deployment

### **Banking Active Learning Recommendations**
1. **Use Config 62 strategy** (4-1-2-1-2-1 pattern) for optimal performance
2. **Implement global feature standardization** for numerical features
3. **Apply regularized Logistic Regression** over complex models  
4. **Validate on domain-specific data** before production deployment

### **Fraud Detection Recommendations**
1. **Use LightGBM for production** (stable 100%+ improvements)
2. **Consider LR for research** (potential 600%+ gains with volatility management)
3. **Implement fair parallel comparison** for valid performance assessment
4. **Monitor for data drift** in highly imbalanced scenarios

### **LLM Validation Recommendations**  
1. **Use Perplexity for highest accuracy** (97.6% F1-score)
2. **Consider OpenAI for balanced cost/performance** (79.8% F1-score) 
3. **Implement provider fallback** for robustness
4. **Monitor API costs** and rate limits

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👥 Authors

- **Lucas Benevides e Braga** - *Author & Developer* - [lucasbraga461](https://github.com/lucasbraga461) | [ORCID](https://orcid.org/0009-0007-5397-5652)

## 📞 Contact

- **Email**: lucasbraga461@gmail.com
- **LinkedIn**: [Lucas Braga](https://linkedin.com/in/lucasbraga461)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎯 Research Summary**: Comprehensive multi-domain active learning research spanning banking (75 configs), fraud detection (550 experiments), and LLM validation (3 providers). Champion banking strategy achieves 6.57% F1 improvement, fraud detection shows 300-600% gains, and Perplexity AI dominates LLM validation with 97.6% F1-score. All results statistically validated and production-ready.