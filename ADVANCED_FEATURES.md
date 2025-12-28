# WeightScope 新功能说明

## 1. 模型比较功能 (Model Comparison)

### 功能描述
比较两个模型（例如基础模型 vs 微调模型，或不同训练阶段的 checkpoint）的权重差异。

### Python API
```python
from weightscope import Scope

scope1 = Scope("model_before")
comparison = scope1.compare_with("model_after", methods=["basic_stats", "spectral"])

# 查找变化最大的层
from weightscope.core.comparator import find_most_changed_layers
most_changed = find_most_changed_layers(comparison, top_k=10)
```

### CLI 命令
```bash
weightscope compare --model1 base_model --model2 finetuned_model --top-changes 10
```

### 比较指标
- **L2 Distance**: 权重的欧几里得距离
- **Cosine Similarity**: 权重向量的余弦相似度
- **Mean/Std Difference**: 统计量的变化
- **L2 Norm Ratio**: 范数的比值
- **Significant Change Percentage**: 显著变化的权重百分比

### 使用场景
1. **训练监控**: 比较不同 epoch 的 checkpoint，观察权重演化
2. **微调评估**: 对比微调前后的权重变化，识别哪些层被显著修改
3. **模型合并**: 在合并多个模型前，评估它们的差异程度
4. **知识蒸馏**: 验证学生模型与教师模型的权重分布差异

---

## 2. 并行分析功能 (Parallel Processing)

### 功能描述
使用多线程并行分析多个层，在多核 CPU 上可实现 **3-4倍加速**。

### Python API
```python
from weightscope import Scope

scope = Scope("openai-community/gpt2")

# 使用并行分析（自动检测 CPU 核心数）
results = scope.scan(methods=["all"], parallel=True)

# 指定工作线程数
results = scope.scan(methods=["all"], parallel=True, num_workers=4)
```

### CLI 命令
```bash
# 使用并行处理
weightscope scan --model gpt2 --methods all --parallel

# 指定线程数
weightscope scan --model gpt2 --methods all --parallel --workers 8
```

### 性能测试结果（GPT-2）
| 分析方法 | 顺序执行 | 并行执行 (4 workers) | 加速比 |
|---------|---------|---------------------|--------|
| basic_stats + spectral | 67.24s | 20.36s | 3.30x |
| 所有方法 | ~120s | ~35s | ~3.4x |

### 技术实现
- **线程池**: 使用 `ThreadPoolExecutor` 避免 CUDA/模块序列化问题
- **自动负载均衡**: 动态分配任务给空闲线程
- **内存优化**: 权重张量提前转移到 CPU，避免 GPU 内存溢出
- **结果一致性**: 并行和顺序分析产生完全相同的结果

### 使用建议
- **小模型 (<1B)**: 并行可能没有明显优势，因为开销较大
- **中大型模型 (1B-70B)**: 强烈推荐使用并行，加速明显
- **GPU 服务器**: 建议使用 4-8 个 workers，过多会导致 CPU 竞争
- **分析方法**: SVD 和量化分析最耗时，从并行中获益最大

---

## 3. 集成使用示例

### 完整工作流
```python
from weightscope import Scope
from weightscope.core.comparator import find_most_changed_layers
import json

# 1. 分析基础模型
print("Analyzing base model...")
base_scope = Scope("llama2-7b")
base_results = base_scope.scan(methods=["all"], parallel=True, num_workers=8)

# 保存基础分析
with open("base_analysis.json", 'w') as f:
    json.dump(base_results, f, indent=2)

# 2. 分析微调模型
print("Analyzing fine-tuned model...")
finetuned_scope = Scope("llama2-7b-chat")
finetuned_results = finetuned_scope.scan(methods=["all"], parallel=True, num_workers=8)

# 3. 比较两个模型
print("Comparing models...")
comparison = base_scope.compare_with(finetuned_results, methods=["basic_stats", "spectral"])

# 4. 找出变化最大的层
most_changed = find_most_changed_layers(comparison, top_k=10)

print("\nTop 10 most changed layers:")
for i, (layer, score, metrics) in enumerate(most_changed, 1):
    print(f"{i}. {layer}: change_score={score:.4f}")
    
# 5. 识别量化敏感层
print("\nQuantization-sensitive layers in fine-tuned model:")
for layer, stats in finetuned_results.items():
    if "quantization" in stats:
        outliers = stats["quantization"]["extreme_outlier_percentage"]
        if outliers > 1.0:
            print(f"  - {layer}: {outliers:.2f}% outliers")
```

### CLI 完整流程
```bash
# 1. 分析基础模型
weightscope scan --model base_model \
  --methods all \
  --parallel --workers 8 \
  --output base_analysis.json

# 2. 分析微调模型
weightscope scan --model finetuned_model \
  --methods all \
  --parallel --workers 8 \
  --output finetuned_analysis.json

# 3. 比较两个模型
weightscope compare \
  --model1 base_model \
  --model2 finetuned_model \
  --methods basic_stats spectral quantization \
  --top-changes 20 \
  --output comparison.json
```

---

## 4. 高级应用场景

### 场景 1: 监控训练稳定性
```python
# 在训练过程中定期检查权重健康度
import glob

checkpoint_files = sorted(glob.glob("checkpoints/step_*.pt"))

for ckpt in checkpoint_files[-5:]:  # 检查最近5个checkpoint
    scope = Scope(ckpt)
    results = scope.scan(methods=["spectral"], parallel=True)
    
    # 检查是否有层的条件数爆炸
    for layer, stats in results.items():
        cond = stats["spectral"]["condition_number"]
        if cond > 1e6:
            print(f"WARNING: {layer} has very high condition number: {cond:.2e}")
```

### 场景 2: 选择最佳量化策略
```python
scope = Scope("your_model")
results = scope.scan(methods=["quantization"], parallel=True)

# 根据异常值百分比决定量化策略
for layer, stats in results.items():
    outlier_pct = stats["quantization"]["extreme_outlier_percentage"]
    
    if outlier_pct > 5:
        print(f"{layer}: Use SmoothQuant or per-channel quantization")
    elif outlier_pct > 1:
        print(f"{layer}: Use mixed precision (keep in FP16)")
    else:
        print(f"{layer}: Safe for INT8 quantization")
```

### 场景 3: 追踪微调过程中的权重漂移
```python
base_scope = Scope("pretrained_model")
finetuned_scope = Scope("finetuned_model")

comparison = base_scope.compare_with("finetuned_model")

# 识别"冻结"层（几乎没变化）
frozen_layers = []
adapted_layers = []

for layer, metrics in comparison["layer_comparisons"].items():
    l2_ratio = metrics["l2_norm_ratio"]
    
    if 0.99 < l2_ratio < 1.01:
        frozen_layers.append(layer)
    elif l2_ratio > 1.1 or l2_ratio < 0.9:
        adapted_layers.append(layer)

print(f"Frozen layers: {len(frozen_layers)}")
print(f"Heavily adapted layers: {len(adapted_layers)}")
```

---

## 5. 性能优化建议

### 内存优化
- 对于超大模型 (>70B)，使用 `device="cpu"` 避免 GPU OOM
- 批量处理：如果内存不足，可以分批分析不同层

### 速度优化
- 并行分析可提速 3-4x，强烈推荐
- 对于只需要基础统计的场景，不要运行 `spectral` 分析（SVD 很慢）
- 使用 `--top-issues N` 限制输出，避免处理大量数据

### 准确性
- SVD 在 float16 下可能不够稳定，内部会自动转换为 float32
- 量化分析使用 numpy percentile 而非 torch.quantile，避免大张量问题
