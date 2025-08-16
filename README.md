# 🧠 Comprehensive Active Learning & LLM Validation Research Repository

A comprehensive research project demonstrating advanced Active Learning strategies and LLM Prompt-Based Validation workflows. This repository contains **75 unique experimental configurations** with rigorous statistical analysis, plus a complete LLM validation framework.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏆 Key Achievements

- **🎯 Champion Strategy Identified**: Config 62 achieves **6.57% F1 improvement** over passive learning
- **📊 Comprehensive Analysis**: 75 unique configurations tested across 4 experimental ranges
- **🔬 Statistical Rigor**: 750 total statistical runs (10 runs per configuration)
- **🏭 Production-Validated**: Strategies validated on real-world financial datasets
- **🤖 LLM Integration**: Complete prompt-based validation framework with OpenAI integration

---

## 📌 What's Inside

### 🧪 **Active Learning Research** (`active-learning/`)
Complete experimental framework with:
- **3 Sampling Strategies**: Uncertainty Sampling, Diversity Sampling (k-NN), Query-by-Committee (QBC)
- **3 Model Types**: Logistic Regression (regularized/unregularized), LightGBM  
- **2 Feature Engineering Approaches**: Binned vs. standardized features
- **Comprehensive Experimentation**: 75 unique configurations with statistical analysis
- **Champion Strategy**: 4-1-2-1-2-1 pattern (uncertainty-diversity-uncertainty-diversity-uncertainty-qbc)

### 🤖 **LLM Prompt-Based Validation** (`llm-prompt-based-validation/`)
Production-ready validation system with:
- **Provider-Agnostic Architecture**: Supports multiple LLM providers
- **Robust JSON Parsing**: Handles malformed responses gracefully
- **Offline/Online Modes**: Deterministic dry-run + OpenAI integration
- **Batched Processing**: Concurrent validation with progress tracking
- **CLI Interface**: Simple command-line operation

---

## 🔍 Research Findings

### **🏆 Champion Strategy (Config 62)**
- **Performance**: 6.57% F1 improvement over passive learning
- **Model**: Logistic Regression with regularization (C=0.1)
- **Features**: Globally standardized numerical features
- **Strategy Pattern**: `uncertainty(4) → diversity(1) → uncertainty(2) → diversity(1) → uncertainty(2) → qbc(1)`

### **📊 Model Performance Ranking**
| Rank | Model Type | Feature Type | Best F1 Improvement | Champion Config |
|------|------------|--------------|-------------------|-----------------|
| 🥇 1st | LR Regularized | Standardized | **6.57%** | Config 62 |
| 🥈 2nd | LR Unregularized | Standardized | **5.37%** | Config 124 |
| 🥉 3rd | LightGBM | Standardized | **4.33%** | Config 95 |
| 4th | LR Regularized | Binned | **3.95%** | Config 23 |

### **🔬 Key Research Insights**
1. **Feature standardization is more important than model complexity**
2. **Uncertainty sampling dominates successful strategies (70-80% of iterations)**
3. **Strategic diversity placement prevents overfitting**
4. **QBC finale provides ensemble-based final refinement**
5. **Regularized models outperform complex alternatives on small labeled datasets**

---

## 📁 Repository Structure

```
active-learning/
├── experimentation/                    # 🧪 Core research experiments
│   ├── COMPREHENSIVE_EXPERIMENT_REPORT.md    # 📊 Complete results analysis
│   ├── simple_active_learning.py            # 🎯 Main experiment script
│   ├── simple_active_learning-lgbm.py       # 🌳 LightGBM variant
│   ├── simple_active_learning-noreg.py      # 📈 Unregularized variant
│   ├── run_configs_*.py                     # ⚙️ Configuration runners
│   ├── configuration_analysis_report_*.md   # 📋 Detailed config reports
│   ├── data/                                # 📊 Experimental results
│   │   ├── logs/                           # 📝 Detailed experiment logs
│   │   └── *_comparison_*.png              # 📈 Performance visualizations
│   └── requirements_lgbm.txt               # 📦 LightGBM dependencies
├── generate_synthetic_data.ipynb           # 🎲 Data generation utilities
├── helpers/                                # 🛠️ Utility functions
└── requirements.txt                        # 📦 Core dependencies

llm-prompt-based-validation/
├── llm_utils.py                            # 🔧 Core utilities & clients
├── llm_validation.py                       # 🎮 CLI runner
├── dummy_dataset.csv                       # 📋 Example dataset
├── results_openai.jsonl                    # 📄 Example results
├── check-results.ipynb                     # 🔍 Results analysis
└── requirements.txt                        # 📦 LLM dependencies
```

---

## 🚀 Quick Start

### **Option 1: Run Champion Strategy (Recommended)**
```bash
# Install dependencies
pip install -r active-learning/requirements.txt

# Run the champion configuration (Config 62)
cd active-learning/experimentation
python simple_active_learning.py
# Manually set CONFIG_NAME = "config62" in the script
```

### **Option 2: Interactive Notebooks**
```bash
# Install dependencies
pip install -r active-learning/requirements.txt

# Launch Jupyter
jupyter notebook active-learning/notebook-active-learning.ipynb
```

### **Option 3: LLM Validation**
```bash
# Install LLM dependencies
pip install -r llm-prompt-based-validation/requirements.txt

# Dry-run mode (offline, deterministic)
python llm-prompt-based-validation/llm_validation.py \
  --input-csv llm-prompt-based-validation/dummy_dataset.csv \
  --output-jsonl results_dryrun.jsonl \
  --dry-run

# OpenAI mode (requires API key in .env)
export OPENAI_API_KEY=sk-...
python llm-prompt-based-validation/llm_validation.py \
  --input-csv llm-prompt-based-validation/dummy_dataset.csv \
  --output-jsonl results_openai.jsonl \
  --provider openai \
  --openai-model gpt-4o
```

---

## 🧪 Experimental Configurations

### **Configuration Ranges**
- **Configs 20-41**: Baseline Logistic Regression with binned features  
- **Configs 50-70**: Standardized features with regularization
- **Configs 80-100**: LightGBM experimentation  
- **Configs 110-130**: Unregularized Logistic Regression

### **Running Specific Configuration Ranges**
```bash
cd active-learning/experimentation

# Run specific configuration ranges
python run_configs_20_40.py     # Baseline experiments
python run_configs_50_70.py     # Standardized features
python run_configs_80_100.py    # LightGBM experiments  
python run_configs_110_130.py   # Unregularized experiments
```

### **Analysis Scripts**
```bash
# Analyze results for specific ranges
python analyze_results-config20-41.py
python analyze_results-config50-70.py
python analyze_results-config80-100.py
python analyze_results-config110-130.py
```

---

## 📊 Comprehensive Results

### **Statistical Validation**
- **Total Experimental Runs**: 750 (75 configs × 10 statistical runs)
- **Dataset**: UCI Bank Marketing (45,211 samples, 11% class imbalance)
- **Statistical Tests**: Paired t-tests, Wilcoxon signed-rank tests
- **Effect Size**: Cohen's d for practical significance
- **Significance Threshold**: p < 0.001 for strong evidence

### **Champion Strategy Details**
**Config 62 (6.57% F1 improvement):**
```
Iterations 1-4:  uncertainty  (build decision boundary confidence)
Iteration 5:     diversity    (strategic exploration)
Iterations 6-7:  uncertainty  (maintain learning momentum)  
Iteration 8:     diversity    (prevent overfitting)
Iterations 9-10: uncertainty  (final boundary refinement)
Iteration 11:    qbc         (ensemble disagreement finale)
```

### **Performance Visualizations**
All experiments include comprehensive visualizations:
- Active vs. Passive learning curves
- Statistical significance comparisons  
- Performance distribution analysis
- Strategy effectiveness heatmaps

### **📁 Accessing Experimental Results**
Complete experimental results are available in the repository:

- **📝 Detailed Logs**: `active-learning/experimentation/data/logs/experiment_log_configXX_*.txt`
  - Full statistical analysis for each of the 75 configurations
  - 10 runs per configuration with complete performance metrics
  - F1 scores, accuracy, precision, recall, and statistical significance tests

- **📈 Statistical Visualizations**: `active-learning/experimentation/data/statistical_comparison_configXX.png`  
  - 104+ comparison charts showing active vs. passive learning performance
  - Box plots with confidence intervals and effect sizes
  - Visual validation of statistical significance for each configuration

**🎯 Key Files for Reproducibility:**
- `experiment_log_config62_*.txt` - Champion strategy detailed results
- `statistical_comparison_config62.png` - Champion performance visualization
- `COMPREHENSIVE_EXPERIMENT_REPORT.md` - Complete analysis of all 75 configurations

---

## 🔬 Research Methodology

### **Active Learning Strategies**
- **Uncertainty Sampling**: Selects samples with lowest model confidence
- **Diversity Sampling**: KNN-based representative sample selection
- **Query by Committee**: Ensemble disagreement-based selection
- **Strategic Combination**: Optimal timing and sequencing patterns

### **Statistical Rigor**
- **Cross-Validation**: Nested cross-validation with GridSearchCV
- **Multiple Runs**: 10 independent runs per configuration
- **Reproducibility**: Fixed random seeds (42-51) for deterministic results
- **Effect Size**: Practical significance beyond statistical significance

### **Feature Engineering**
- **Standardization**: Global Z-score normalization for numerical features
- **Log Transformation**: Applied to skewed distributions
- **Categorical Encoding**: One-hot encoding with rare category aggregation
- **Class Balancing**: Weighted sampling for imbalanced datasets

---

## 💡 LLM Prompt-Based Validation

### **Core Features**
- **Flexible Prompting**: Customizable validation prompts for any domain
- **Robust Parsing**: Handles malformed JSON responses gracefully
- **Provider Support**: OpenAI (extensible to other providers)
- **Offline Testing**: Deterministic rule-based responses for development
- **Batch Processing**: Concurrent requests with progress tracking

### **Production Use Cases**
- Data quality validation
- Content moderation  
- Classification verification
- Anomaly detection confirmation
- Human-in-the-loop workflows

---

## 📚 Documentation

### **Comprehensive Reports**
- [`COMPREHENSIVE_EXPERIMENT_REPORT.md`](active-learning/experimentation/COMPREHENSIVE_EXPERIMENT_REPORT.md): Complete analysis of all 75 configurations
- [`README_CONFIG_OPTIMIZATION.md`](active-learning/experimentation/README_CONFIG_OPTIMIZATION.md): Configuration optimization methodology
- Individual configuration reports for detailed analysis

### **Research Papers & Citations**
Based on production validation in real-world financial datasets. **Paper submitted to IEEE Access**. If you use this work, please cite:

```bibtex
@article{braga2025activelearning,
  title={{Active Learning for Imbalanced Classification: Empirical Insights, Iteration Scheduling, and LLM-Augmented Validation}},
  author={Benevides e Braga, Lucas},
  journal={IEEE Access},
  year={2025},
  note={Submitted},
  url={https://github.com/lucasbraga461/active-learning}
}
```

---

## 🔗 Dataset Access

### **📊 Main Experimental Dataset**
**UCI Bank Marketing Dataset** (used for all 75 configurations):
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Size**: 45,211 samples with 11% positive class
- **Domain**: Financial services customer behavior
- **Use Case**: Marketing campaign optimization

### **🎲 Synthetic Datasets** 
**Custom synthetic datasets for tutorials and examples**:
- **DOI**: [10.21227/29cz-j345](https://dx.doi.org/10.21227/29cz-j345)
- **Purpose**: Educational demonstrations and LLM validation workflows
- **Generated by**: `generate_synthetic_data.ipynb` notebook
- **Use Case**: Tutorial examples and method validation

---

## 🛠️ Advanced Configuration

### **Custom Experiments**
```python
# Example: Create custom configuration
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

### **Environment Variables**
```bash
# Optional configuration
export OPENAI_API_KEY=sk-...        # For LLM validation
export AL_EXPERIMENT_TIMEOUT=1800   # Experiment timeout (seconds)
export AL_MAX_WORKERS=4             # Parallel processing workers
```

---

## 🚀 Production Deployment

### **Key Recommendations**
1. **Use Config 62 strategy** for optimal performance
2. **Implement global feature standardization** for numerical features
3. **Apply regularized Logistic Regression** over complex models
4. **Follow 4-1-2-1-2-1 uncertainty-diversity pattern**
5. **Validate on domain-specific data** before production deployment

### **Production Checklist**
- [ ] Implement champion strategy (Config 62)
- [ ] Set up feature standardization pipeline
- [ ] Configure class balancing for imbalanced datasets
- [ ] Set up monitoring for model performance drift
- [ ] Implement statistical testing for performance validation

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

**🎯 Research Summary**: 75 configurations tested, Champion strategy (Config 62) achieves 6.57% F1 improvement through strategic uncertainty sampling with diversity placement and QBC finale. Feature standardization proved more important than model complexity for active learning effectiveness.
