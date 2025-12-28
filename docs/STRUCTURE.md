# WeightScope 项目结构

```
weightscope/
├── pyproject.toml              # 项目配置和依赖
├── README.md                   # 项目文档
│
├── weightscope/                # 主包
│   ├── __init__.py            # Scope 主类
│   ├── cli.py                 # 命令行接口
│   │
│   ├── core/                  # 核心模块
│   │   ├── loader.py          # 模型加载器（ModelScope + Transformers）
│   │   └── walker.py          # 层遍历器（支持 Linear, Conv1D, Embedding）
│   │
│   ├── analyzers/             # 分析器模块
│   │   ├── basic.py           # 基础统计（均值、方差、范数）
│   │   ├── spectral.py        # 频谱分析（SVD、条件数、秩）
│   │   ├── quantization.py    # 量化敏感度（异常值、SQNR）
│   │   └── sparsity.py        # 稀疏性分析（零值、死神经元）
│   │
│   ├── visualizers/           # 可视化模块
│   │   ├── plot.py            # Matplotlib 绘图
│   │   └── terminal.py        # Rich 终端输出
│   │
│   └── report/                # 报告生成（未来扩展）
│
├── tests/                     # 测试文件
│   └── test_analyzers.py     # 分析器单元测试
│
└── examples/                  # 示例脚本
    └── analyze_gpt2.py       # GPT-2 分析示例
```

## 核心组件说明

### 1. Scope 主类 (`__init__.py`)
- 提供统一的 API 接口
- 协调 Loader 和 Walker
- 调用各种分析器
- 返回结构化结果

### 2. Loader (`core/loader.py`)
- 使用 ModelScope 下载模型
- 通过 Transformers 加载模型
- 支持 AutoModelForCausalLM 和 AutoModel
- 自动处理本地/远程路径

### 3. Walker (`core/walker.py`)
- 遍历模型的所有层
- 支持多种层类型：Linear, Conv1D, Conv2d, Embedding
- 自动检测 GPT-2 风格的 Conv1D 层

### 4. 分析器 (`analyzers/`)

#### Basic (`basic.py`)
- 均值、标准差、最小值、最大值
- L2 范数
- 零值百分比

#### Spectral (`spectral.py`)
- 奇异值分解 (SVD)
- 条件数 (Condition Number)
- 稳定秩 (Stable Rank)
- 有效秩 (Effective Rank)

#### Quantization (`quantization.py`)
- IQR 方法检测异常值
- 模拟 INT8 量化
- 计算 MSE、MAE、余弦相似度
- SQNR (信号与量化噪声比)
- 动态范围分析

#### Sparsity (`sparsity.py`)
- 多阈值稀疏度统计
- 结构化稀疏性（行/列级别）
- 权重幅度分布

### 5. CLI (`cli.py`)
- 命令行参数解析
- Rich 表格美化输出
- JSON 结果导出
- 按问题严重程度排序

## 使用场景

1. **模型训练监控**：检查训练过程中的权重健康度
2. **量化前评估**：识别对量化敏感的层
3. **模型压缩**：发现低秩层和稀疏层
4. **模型比较**：对比不同 checkpoint 的权重变化
5. **问题诊断**：定位训练崩溃、梯度爆炸的原因
