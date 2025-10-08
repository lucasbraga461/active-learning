# Multi-LLM Business Validation Research

This project compares different LLM providers for business validation tasks, specifically determining whether businesses are legitimate, operating restaurants.

## 🏆 Key Results

**📊 [COMPREHENSIVE LLM COMPARISON](COMPREHENSIVE_LLM_COMPARISON.md)** - Main results document with complete analysis

**Winner: Perplexity AI** - 97.6% F1-score, 99.0% recall, 0.9% failure rate

## 📁 Project Structure

```
llm-business-validation/
├── shared/                          # Shared components
│   ├── data/                       # Common datasets
│   │   ├── nyc_restaurants_sample_115.csv
│   │   └── nyc_validation_115_results-withactuals.csv
│   ├── scripts/                    # Shared utilities
│   │   ├── base_llm_client.py     # Abstract base class
│   │   ├── multi_llm_validator.py # Multi-provider runner
│   │   └── evaluate_performance.py # Performance analysis
│   └── requirements.txt            # Python dependencies
├── experiments/                     # Provider-specific experiments
│   ├── openai/                     # OpenAI GPT experiments
│   │   ├── openai_client.py       # OpenAI implementation
│   │   └── results/               # OpenAI experiment results
│   │       ├── RESULTS.md         # Detailed analysis
│   │       ├── *.jsonl, *.csv     # Raw data
│   │       └── *.md               # Performance metrics
│   ├── perplexity/                # Perplexity AI experiments
│   │   ├── perplexity_client.py   # Perplexity implementation
│   │   └── results/               # Perplexity experiment results
│   │       ├── perplexity_performance_metrics.md
│   │       └── *.jsonl, *.csv     # Raw data
│   └── gemini/                    # Google Gemini experiments
│       ├── gemini_client.py       # Gemini implementation
│       └── results/               # Gemini experiment results
│           ├── gemini_performance_metrics.md
│           └── *.jsonl, *.csv     # Raw data
├── data/                           # Raw datasets
└── .env                           # API keys (not in git)
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment (REQUIRED)
source ../venv-al-llm/bin/activate

# Install dependencies
pip install -r shared/requirements.txt

# Set up API keys in .env file (in project root)
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GOOGLE_API_KEY=your_google_key_here
```

### 2. Test Individual Providers

Each provider can be tested individually with a sample business:

```bash
# Test OpenAI GPT-4o (no web search)
cd experiments/openai
python openai_client.py

# Test Perplexity Sonar (with web search)
cd experiments/perplexity  
python perplexity_client.py

# Test Gemini 1.5 Pro (no web search)
cd experiments/gemini
python gemini_client.py
```

**Expected Output:** Each test validates "STARBUCKS, 157 LAFAYETTE STREET, New York" and shows:
- Provider name
- Validation result (True/False)
- Confidence score (0-100)
- Reasoning snippet

### 3. Run Multi-LLM Validation

#### Single Provider Validation
```bash
# Run OpenAI only
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers openai

# Run Perplexity only (with rate limiting)
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers perplexity \
  --max-workers 1

# Run Gemini only
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers gemini
```

#### Multi-Provider Comparison
```bash
# Run all providers (WARNING: Takes ~15 minutes, costs ~$1.10)
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers openai perplexity gemini

# Test with limited records first
python shared/scripts/multi_llm_validator.py \
  --input-csv shared/data/nyc_restaurants_sample_115.csv \
  --providers openai perplexity gemini \
  --limit 5
```

**Input Requirements:**
- CSV file with columns: `business_name`, `address`, `city`
- Each row represents a business to validate

**Output Generated:**
- `experiments/{provider}/results/{provider}_{count}_results.jsonl` - Raw API responses
- `experiments/{provider}/results/{provider}_{count}_results.csv` - Structured results
- Results include: business info, is_valid, confidence, reasoning, provider

### 4. Performance Analysis

Evaluate LLM performance against ground truth labels:

```bash
# Analyze individual provider (requires ground truth labels)
python shared/scripts/evaluate_performance.py \
  --input experiments/openai/results/openai_115_results-withactuals.csv \
  --output experiments/openai/results/openai_performance_metrics.md

python shared/scripts/evaluate_performance.py \
  --input experiments/perplexity/results/perplexity_115_results-withactuals.csv \
  --output experiments/perplexity/results/perplexity_performance_metrics.md

python shared/scripts/evaluate_performance.py \
  --input experiments/gemini/results/gemini_115_results-withactuals.csv \
  --output experiments/gemini/results/gemini_performance_metrics.md
```

**Input Requirements:**
- CSV with LLM predictions AND `true_label_is_valid` column
- Columns: `business_name`, `address`, `city`, `is_valid`, `confidence`, `reasoning`, `true_label_is_valid`

**Output Generated:**
- Detailed performance report with metrics, confusion matrix, false positives/negatives
- Markdown file with comprehensive analysis

### 5. Speed & Cost Analysis

Measure API performance and cost estimation:

```bash
python shared/scripts/measure_performance.py
```

**Output:** Speed comparison, cost per request, and extrapolated costs for larger datasets.

## ⚠️ Important Usage Notes

### Virtual Environment
**CRITICAL:** Always activate the virtual environment before running any Python scripts:
```bash
source ../venv-al-llm/bin/activate
```

### Rate Limiting
- **Perplexity**: 50 requests/minute limit - use `--max-workers 1` for large datasets
- **OpenAI**: Higher limits - can use default `--max-workers 3`
- **Gemini**: Moderate limits - default settings work fine

### Cost Considerations
- **Full 115 samples**: ~$1.10 total ($0.17 Gemini + $0.35 OpenAI + $0.58 Perplexity)
- **Test with `--limit 5`** first to verify setup
- **Perplexity most expensive** but best performance

### File Organization
Results are automatically saved to provider-specific folders:
```
experiments/
├── openai/results/openai_115_results.csv
├── perplexity/results/perplexity_115_results.csv
└── gemini/results/gemini_115_results.csv
```

## 🔧 Troubleshooting

### Common Issues
```bash
# API key not found
export OPENAI_API_KEY=your_key_here  # or add to .env

# Rate limit errors (Perplexity)
# Use --max-workers 1 and expect ~8 minutes for 115 samples

# Module not found
source ../venv-al-llm/bin/activate  # Always activate venv first

# File not found
# Ensure you're in the llm-business-validation directory
```

### Verification Steps
1. Test individual providers first
2. Run with `--limit 5` before full dataset
3. Check API keys are loaded: `echo $OPENAI_API_KEY`
4. Verify virtual environment: `which python`

## 📊 Expected Performance

Based on 115 NYC restaurant samples:

| Provider | F1-Score | Speed | Cost | Best For |
|----------|----------|--------|------|----------|
| **Perplexity** | 97.6% | 4.2s/req | $0.58 | Production (best accuracy) |
| **OpenAI** | 79.8% | 2.4s/req | $0.35 | Budget-conscious |
| **Gemini** | 65.8% | 2.7s/req | $0.17 | Cost-sensitive |

## 🔧 Extending the Framework

To add a new LLM provider:

1. Create `experiments/new_provider/new_provider_client.py`
2. Inherit from `BaseLLMClient` 
3. Implement `_get_api_key()` and `_call_api()` methods
4. Add to `multi_llm_validator.py`

## 📄 License

This project is licensed under the MIT License.