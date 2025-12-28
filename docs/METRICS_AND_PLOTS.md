# WeightScope 指标与图表解读（为什么有用）

本文档面向“用权重分析定位问题/做决策”的场景，逐项解释 WeightScope 输出的核心指标与可视化图。

- 适用对象：训练/微调工程师、量化/部署工程师、模型压缩/性能优化工程师
- 适用模型：绝大多数 Transformer（含 GPT-2 类 Conv1D/Linear/Embedding）
- 数据来源：**仅权重**（不依赖数据集、不需要前向推理），因此能做“上线前体检/离线诊断”

---

## 1. 使用方式（你会得到什么）

### 1.1 扫描输出结构
`Scope.scan()` 返回结构：

```python
{
  "layer.full.name": {
    "basic_stats": {...},
    "spectral": {...},
    "quantization": {...},
    "sparsity": {...}
  },
  ...
}
```

### 1.2 生成图表
示例脚本：`examples/visualization_demo.py`，默认输出到：`visualization_outputs/`

常用命令：
```bash
# 在 weightscope 目录下
conda run -n llm python examples/visualization_demo.py
```

---

## 2. Basic Stats（基础统计）

代码：`weightscope/analyzers/basic.py` → `analyze_basic_stats()`

### 2.1 指标字段与含义
- `mean`：均值。用于检查权重是否有明显偏置（大多数层应接近 0，Embedding/部分层可能偏离）。
- `std`：标准差。权重尺度的粗略刻画；过大/过小都可能导致训练不稳定或表达能力不足。
- `min` / `max`：极值。用于发现异常尖峰（可能导致量化 clipping、数值不稳定）。
- `l2_norm`：$\|W\|_2$（这里是张量的 2 范数）。可用于粗略衡量权重“能量”。
- `zeros_percentage`：精确为 0 的比例（百分比）。

### 2.2 为什么有用
- 快速发现“尺度不一致”：某一层 `std` 明显高于其它层 → 可能是训练崩溃前兆、量化高风险点。
- 快速发现“异常极值”：`max`/`min` 极端离群 → 通常意味着 outlier-heavy，后续 INT8 会很难。

---

## 3. Spectral（谱分析：SVD/条件数/有效秩）

代码：`weightscope/analyzers/spectral.py` → `analyze_spectral()`

谱分析把权重矩阵看作线性映射 $y = Wx$，通过奇异值分解（SVD）理解它的数值性质与表达自由度。

### 3.1 指标字段与含义
- `max_singular_value`：最大奇异值 $\sigma_{\max}$（谱范数）。映射的最大放大倍数。
- `min_singular_value`：最小奇异值 $\sigma_{\min}$。接近 0 意味着映射在某些方向几乎“压扁”。
- `condition_number`：条件数 $\kappa = \sigma_{\max}/\sigma_{\min}$。
  - 直觉：**越大越病**，表示映射在不同方向放大倍数差异巨大。
- `stable_rank`：稳定秩 $\|W\|_F^2/\|W\|_2^2$。
  - 直觉：比“数学秩”更稳定，越小表示能量集中在少数主方向（低秩/可压缩）。
- `effective_rank`：有效秩（这里定义为奇异值 > `0.01 * max_sv` 的数量）。
- `total_rank`：奇异值数量（等价于 $\min(m,n)$，是矩阵最大可能秩）。
- `top_singular_values`：前 `top_k` 个奇异值，用于画谱图/观察“长尾”。
- `singular_value_ratio`：$\sigma_{\min}/\sigma_{\max}$（与条件数相反）。

### 3.2 常用阈值（经验）
这些阈值不是“真理”，但非常适合做预警/排序：
- 条件数 `condition_number`：
  - $\kappa < 10^3$：通常安全
  - $10^3 \sim 10^4$：需要注意（混合精度/量化可能更敏感）
  - $10^4 \sim 10^5$：高风险
  - $> 10^5$：极高风险（训练 NaN/Inf、推理不稳定的概率显著上升）

- 秩利用率（可自己算）：
  - `stable_rank / total_rank` 很小（例如 < 0.3）：常见“可压缩/低秩”信号

### 3.3 为什么有用
- **数值稳定性诊断**：条件数高 → 反向传播时某些方向梯度可能被放大，混合精度更容易溢出。
- **量化前体检**：条件数高的层常常对数值噪声敏感，量化误差更容易放大。
- **压缩决策**：`stable_rank`/`effective_rank` 低 → 表明存在大量冗余，可考虑 LoRA、小秩分解等。

---

## 4. Quantization（量化敏感度/异常值/SQNR）

代码：`weightscope/analyzers/quantization.py` → `analyze_quantization_sensitivity()`

该分析做了两件事：
1) 用 IQR 检测 outlier（异常值）比例
2) 做一次“对称量化模拟”（按 `abs_max` 缩放，round+clamp）估计误差

### 4.1 指标字段与含义
- Outlier（异常值）相关：
  - `outlier_percentage`：IQR(1.5×) 规则下的异常值百分比
  - `extreme_outlier_percentage`：IQR(3×) 的极端异常值百分比（更重要）
  - `outlier_count` / `extreme_outlier_count`：对应计数

- 量化模拟相关：
  - `bits` 固定为 8（INT8）时：
  - `abs_max`：$\max |w|$，决定量化 scale
  - `quantization_scale`：对称量化的 scale（$abs\_max / 127$）
  - `quantization_mse` / `quantization_mae`：量化误差（越小越好）
  - `cosine_similarity`：原始权重向量与反量化权重向量的余弦相似度（越接近 1 越好）
  - `sqnr_db`：SQNR（dB），$10\log_{10}(\text{signal}/\text{noise})$，越高越好
  - `dynamic_range`：$abs\_max / mean(|w|)$（粗略动态范围），越大往往越 outlier-heavy

### 4.2 常用阈值（经验）
- `sqnr_db`：
  - > 40 dB：通常较安全
  - 30–40 dB：中风险（可能需要 per-channel、平滑量化等）
  - < 30 dB：高风险（naive INT8 很可能掉点明显）

- `extreme_outlier_percentage`：
  - < 0.1%：通常友好
  - 0.1–1%：需要策略（per-channel、clipping、SmoothQuant）
  - > 1%：高风险
  - > 10%：非常高风险（强烈不建议 naive INT8）

### 4.3 为什么有用
- **在量化前就能预测灾难层**：outlier-heavy 层会迫使 scale 由极端值决定，导致大多数权重被“压扁到同一格”，误差暴增。
- **指导混合精度/混合 bit 宽**：不是所有层都用 INT8。该分析能自动给出“哪些层保持 FP16/用 INT16”的证据。

---

## 5. Sparsity（稀疏度/结构化稀疏/小权重分布）

代码：`weightscope/analyzers/sparsity.py` → `analyze_sparsity()`

### 5.1 指标字段与含义
- `sparsity_levels`：不同阈值下“近似为 0”的比例（百分比）
  - `threshold_0.0`：严格 0
  - `threshold_1e-6` / `1e-4` / `1e-2`：把很小的权重当作“可剪”
- `structured_sparsity`（仅 2D 权重矩阵）：
  - `dead_rows` / `dead_rows_percentage`：整行几乎全 0（死神经元/通道）
  - `dead_columns` / `dead_columns_percentage`：整列几乎全 0
- `magnitude_percentiles`：|w| 的分位数（p10/p25/p50/p75/p90），描述“小权重占比/尾部厚度”
- `mean_abs_weight` / `median_abs_weight`：|w| 的均值/中位数

### 5.2 为什么有用
- **剪枝/稀疏化可行性评估**：小权重比例高 → 更适合幅值剪枝。
- **结构化剪枝线索**：dead rows/cols 高 → 可能有直接可裁剪的通道。

---

## 6. 图表解读（每张图看什么）

代码：`weightscope/visualizers/plot.py`（图函数都在这里）

> 注：为了避免“标签只剩 c_proj/文字重叠”，图中会对层名做语义缩写（例如 `h9.attn.c_proj`），并对多层情况做 tick 抽样。

### 6.1 `condition_heatmap.png`（条件数热力图）
函数：`plot_condition_number_heatmap(results)`

- y 轴：层名（语义缩写）
- x 轴：$\log_{10}(\kappa)$
- 颜色：
  - 绿：$\kappa < 10^3$
  - 黄：$10^3 \sim 10^4$
  - 橙：$10^4 \sim 10^5$
  - 红：$> 10^5$

怎么看：
- 先看红色条：这是“数值不稳定首要怀疑层”。
- 再看橙色条：量化/混合精度训练时经常是二级风险点。

为什么有用：
- **训练崩溃定位**：当你遇到 NaN/Inf，优先排查这些层（FP32 保留、加 norm、降 lr、梯度裁剪）。
- **量化策略**：高条件数层对噪声敏感，通常不适合激进低比特。

### 6.2 `quantization_risk.png`（量化风险矩阵）
函数：`plot_quantization_risk_matrix(results)`

- x 轴：`extreme_outlier_percentage`（%）
- y 轴：`sqnr_db`（dB）
- 图中水平/垂直参考线：
  - SQNR 40 dB：相对安全线
  - outlier 1%：策略线
- 红色背景区域：SQNR < 30 dB 代表高风险

怎么看：
- **右下角（outlier 大、SQNR 低）**：强烈不建议 naive INT8。
- **左上角（outlier 小、SQNR 高）**：较适合 INT8。

为什么有用：
- 一眼看到“哪些层需要特殊量化策略”。
- 避免把时间浪费在盲目尝试（尤其是大模型量化试一次成本很高）。

### 6.3 `rank_efficiency.png`（秩效率/压缩机会）
函数：`plot_rank_efficiency(results)`

包含三块：
1) 顶部条形图：每层的秩利用率（这里用 `stable_rank/total_rank` 的百分比表达）
2) 左下散点：`total_rank` vs `stable_rank`
3) 右下饼图：压缩机会分布（高/中/低/已高效）

怎么看：
- 低于 30%：通常意味着“可压缩空间大”。
- 极低（如 < 10%）：常见于 embedding/lm_head 或某些投影层，适合 LoRA / 低秩分解。

为什么有用：
- **直接指导压缩/微调 rank**：不用盲选 LoRA rank，可以从稳定秩得到一个“有数据支撑的起点”。

### 6.4 `health_dashboard.png`（模型健康仪表盘）
函数：`plot_model_health_dashboard(results)`

包含：
- 条件数分布直方图（稳定性）
- outlier 分布直方图（量化风险）
- 秩利用率分布直方图（压缩空间）
- SQNR 分布直方图（量化质量）
- sparsity 分布直方图（稀疏/剪枝空间）
- 健康分级饼图（Excellent/Good/Warning/Critical）
- Top-10 风险层条形图（综合风险排序）

怎么看：
- 先看 Top-10：优先处理这些层。
- 再看分布：判断“问题是集中在少数层还是全局系统性问题”。

为什么有用：
- **上线前体检/回归对比**：改了训练策略/合并了 checkpoint，仪表盘能快速判断是否引入了系统性退化。

### 6.5 `layer_details/*_distribution.png`（权重分布直方图）
函数：`plot_weight_distribution(tensor)`

- 左：线性频数直方图
- 右：对数频数直方图（更容易看到尾部/极端值）

怎么看：
- 如果对数图里尾部很长/极端值密集：往往是 outlier-heavy，量化和数值稳定性都更难。

### 6.6 `layer_details/*_singular_values.png`（奇异值谱图）
函数：`plot_singular_values(tensor)`

- 左：Top-K 奇异值（线性）
- 右：全量奇异值（对数）

怎么看：
- 右图如果快速衰减并很早贴近 0：低秩信号强。
- 如果尾部长期不衰减：更“满秩”，压缩空间相对小。

---

## 7. 为什么“只看权重”也能发现问题？

### 7.1 权重刻画了模型的三类工程风险
- 数值风险（条件数/谱范数）：会影响训练稳定性、混合精度、噪声敏感度
- 表达风险（秩/能量集中）：会影响容量利用率、可压缩性、微调方式选择
- 部署风险（outlier/SQNR）：会影响量化是否掉点、需要何种量化策略

### 7.2 它是“必要但不充分”的诊断
- 仅权重分析无法替代：激活分布、数据漂移、任务性能评估
- 但它能：
  - 在没有数据集/推理成本很高时，给出**快速、可排序、可行动**的信号
  - 把排查范围从“全模型”缩小到“少数层”

---

## 8. 常见行动指南（从指标到动作）

### 8.1 训练/微调不稳定（NaN/Inf/爆炸）
优先看：`condition_heatmap.png` + `health_dashboard.png` 的 Top-10
- 对高条件数层：
  - 保持 FP32（或 BF16）
  - 降学习率、启用梯度裁剪
  - 检查是否缺少 norm/初始化异常

### 8.2 INT8 掉点明显
优先看：`quantization_risk.png`
- 对 outlier-heavy 层：
  - per-channel 量化
  - clipping / SmoothQuant
  - 混合 bit：该层保持 FP16/INT16

### 8.3 需要压缩/加速（LoRA/蒸馏/分解）
优先看：`rank_efficiency.png`
- 低秩强的层：更适合 LoRA 或低秩分解
- “已高效”的层：压缩收益可能小，优先级靠后

---

## 9. 限制与注意事项
- 条件数对“接近零的最小奇异值”非常敏感；当 `min_singular_value` 极小会导致 `inf`。
- 量化分析目前是“对称量化模拟”，与具体部署框架（GPTQ/AWQ/RTN/PerChannel）会有差异，但**用于筛风险层非常有效**。
- 稀疏分析默认阈值是幅值阈值，不等价于结构化稀疏训练得到的稀疏图谱。

---

## 10. 快速索引（图 ↔ 函数 ↔ 指标）

- `condition_heatmap.png` → `plot_condition_number_heatmap()` → `spectral.condition_number`
- `quantization_risk.png` → `plot_quantization_risk_matrix()` → `quantization.extreme_outlier_percentage`, `quantization.sqnr_db`
- `rank_efficiency.png` → `plot_rank_efficiency()` → `spectral.stable_rank`, `spectral.total_rank`
- `health_dashboard.png` → `plot_model_health_dashboard()` → 综合以上 + `sparsity.*`
- `layer_details/*_distribution.png` → `plot_weight_distribution()` → 权重分布/尾部
- `layer_details/*_singular_values.png` → `plot_singular_values()` → 奇异值谱
