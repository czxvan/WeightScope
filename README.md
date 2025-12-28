# WeightScope

WeightScope is a Python toolkit designed to analyze Large Language Model (LLM) weights to identify potential defects, anomalies, and opportunities for optimization.

## Features

- **Basic Statistics**: Analyze mean, std, min, max, L2 norm, and sparsity of weights per layer
- **Spectral Analysis**: Perform SVD to check singular values, condition numbers, stable rank, and effective rank
- **Quantization Sensitivity**: Identify outliers, estimate quantization errors (MSE, SQNR), and assess INT8 readiness
- **Sparsity Analysis**: Detect zero weights, dead neurons, and structured sparsity patterns
- **Model Comparison**: Compare two models or checkpoints to identify weight changes
- **Parallel Processing**: Multi-threaded analysis for 3-4x faster processing on multi-core CPUs
- **Rich Visualizations**: Generate beautiful terminal reports with color-coded issue severity
- **Metrics & Plots Guide**: See [`METRICS_AND_PLOTS.md`](./METRICS_AND_PLOTS.md) for detailed metric + visualization explanations

## Installation

```bash
cd weightscope
pip install -e .
```

## Usage

### Command Line Interface

**Basic scan (statistics only):**
```bash
weightscope scan --model openai-community/gpt2
```

**Run all analyses:**
```bash
weightscope scan --model openai-community/gpt2 --methods all
```

**Specific analyses:**
```bash
weightscope scan --model qwen/Qwen-7B --methods spectral quantization --top-issues 10
```

**Save results to JSON:**
```bash
weightscope scan --model openai-community/gpt2 --methods all --output report.json
```

### Python API

```python
from weightscope import Scope

# Load a model (supports ModelScope paths or local paths)
scope = Scope("openai-community/gpt2")

# Run comprehensive analysis
results = scope.scan(methods=["basic_stats", "spectral", "quantization", "sparsity"])

# Or run all analyses with parallel processing
results = scope.scan(methods=["all"], parallel=True, num_workers=4)

# Compare with another model
comparison = scope.compare_with("other_model_path", methods=["basic_stats", "spectral"])

```

## Analysis Methods

### 1. Basic Statistics (`basic_stats`)
- Mean, Standard Deviation, Min, Max
- L2 Norm
- Zero percentage

### 2. Spectral Analysis (`spectral`)
- **Condition Number**: Ratio of max/min singular values (high values indicate numerical instability)
- **Stable Rank**: Effective dimensionality of the weight matrix
- **Effective Rank**: Number of significant singular values
- **Singular Value Spectrum**: Distribution of singular values

### 3. Quantization Sensitivity (`quantization`)
- **Outlier Detection**: Identifies extreme weight values using IQR method
- **SQNR (Signal-to-Quantization-Noise Ratio)**: Measures quantization quality
- **Cosine Similarity**: Post-quantization similarity to original weights
- **Dynamic Range**: Ratio of max absolute value to mean absolute value

### 4. Sparsity Analysis (`sparsity`)
- **Near-Zero Analysis**: Weights below various thresholds (1e-6, 1e-4, 1e-2)
- **Dead Neurons**: Rows/columns that are entirely zero
- **Magnitude Percentiles**: Distribution of weight magnitudes

### 5. Model Comparison (`compare`)
- **Layer-wise Differences**: Compare statistics between two models
- **Weight Movement**: L2 distance and relative changes
- **Cosine Similarity**: Measure of weight preservation
- **Change Detection**: Identify most significantly changed layers

### 6. Parallel Processing
- **Multi-threading**: Analyze multiple layers simultaneously
- **3-4x Speedup**: On typical multi-core CPUs
- **Automatic Load Balancing**: Distributes work across available cores
- **Memory Efficient**: Streams results to avoid memory spikes

## Example Output

```
                        Top 3 Layers by Condition Number                        
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer              ┃ Condition # ┃ Stable Rank ┃ Effective Rank ┃ Total Rank ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ transformer.h.9.a… │   129498.63 │       21.90 │            724 │        768 │
│ transformer.h.0.a… │   103109.80 │       18.01 │            559 │        768 │
│ transformer.h.1.a… │    81291.45 │        8.44 │            656 │        768 │
└────────────────────┴─────────────┴─────────────┴────────────────┴────────────┘
```

## Interpreting Results

### High Condition Numbers (>1000)
- Indicates numerical instability
- May cause issues during fine-tuning or inference
- Consider layer normalization or weight regularization

### High Outlier Percentage (>1%)
- Challenging for quantization
- May require per-channel quantization or SmoothQuant-style techniques
- High dynamic range suggests uneven weight distribution

### Low Effective Rank
- Indicates redundancy in the layer
- Good candidate for low-rank decomposition (LoRA)
- May be over-parameterized

### Dead Neurons (>0)
- Completely inactive weights
- Can be pruned without loss
- May indicate training issues

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- modelscope
- numpy, scipy
- rich (for beautiful terminal output)

## License

MIT License
