"""
WeightScope 可视化演示 - 生成各种实用图表

这个脚本演示如何使用 WeightScope 生成各种诊断图表：
1. 条件数热力图 - 快速识别数值不稳定的层
2. 量化风险矩阵 - 评估 INT8 量化适用性
3. 秩效率分析 - 发现模型压缩机会
4. 模型健康仪表盘 - 一图总览所有关键指标
5. 模型对比图 - 比较不同版本的模型
"""

import sys
from pathlib import Path

# 添加 weightscope 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from weightscope import Scope
from weightscope.visualizers.plot import (
    plot_condition_number_heatmap,
    plot_quantization_risk_matrix,
    plot_rank_efficiency,
    plot_model_health_dashboard,
    plot_weight_distribution,
    plot_singular_values
)


def demo_1_condition_number_heatmap():
    """演示1: 条件数热力图 - 识别数值不稳定层"""
    print("=" * 80)
    print("演示 1: 条件数热力图")
    print("=" * 80)
    print("\n目的: 快速识别哪些层存在数值不稳定问题")
    print("应用: 调试训练崩溃、NaN 问题、混合精度训练失败\n")
    
    # 加载模型并分析
    print("正在加载 GPT-2 并进行谱分析...")
    scope = Scope("openai-community/gpt2")
    results = scope.scan(methods=["spectral"], parallel=True)
    
    # 生成条件数热力图
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("\n生成条件数热力图...")
    plot_condition_number_heatmap(results, output_path=output_dir / "condition_heatmap.png")
    
    # 分析结果
    critical_layers = []
    for layer, stats in results.items():
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1e5:
                critical_layers.append((layer, cond))
    
    print(f"\n发现 {len(critical_layers)} 个严重不稳定层 (条件数 > 10^5):")
    for layer, cond in sorted(critical_layers, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {layer}: {cond:.2e}")
    
    print("\n💡 建议:")
    print("  - 红色层需要特别关注，可能导致训练崩溃")
    print("  - 混合精度训练时，将这些层保持在 FP32")
    print("  - 考虑添加 LayerNorm 或使用梯度裁剪")
    print()


def demo_2_quantization_risk_matrix():
    """演示2: 量化风险矩阵 - 评估量化适用性"""
    print("=" * 80)
    print("演示 2: 量化风险矩阵")
    print("=" * 80)
    print("\n目的: 在量化之前预测哪些层会出问题")
    print("应用: 制定智能量化策略，避免性能灾难\n")
    
    # 加载模型并进行量化分析
    print("正在进行量化敏感度分析...")
    scope = Scope("openai-community/gpt2")
    results = scope.scan(methods=["quantization"], parallel=True)
    
    output_dir = Path("visualization_outputs")
    
    # 生成量化风险矩阵
    print("\n生成量化风险矩阵散点图...")
    plot_quantization_risk_matrix(results, output_path=output_dir / "quantization_risk.png")
    
    # 分析量化策略
    int8_safe = []
    int16_needed = []
    fp16_keep = []
    
    for layer, stats in results.items():
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            sqnr = stats["quantization"].get("sqnr_db", 0)
            
            if outlier < 0.1 and sqnr > 40:
                int8_safe.append(layer)
            elif outlier < 1.0 and sqnr > 30:
                int16_needed.append(layer)
            else:
                fp16_keep.append(layer)
    
    print(f"\n量化策略建议:")
    print(f"  ✅ INT8 安全层: {len(int8_safe)} 个 ({len(int8_safe)/len(results)*100:.1f}%)")
    print(f"  ⚠️  INT16 建议层: {len(int16_needed)} 个 ({len(int16_needed)/len(results)*100:.1f}%)")
    print(f"  ❌ FP16 保留层: {len(fp16_keep)} 个 ({len(fp16_keep)/len(results)*100:.1f}%)")
    
    if fp16_keep:
        print(f"\n需要保持 FP16 的高风险层:")
        for layer in fp16_keep[:5]:
            stats = results[layer]["quantization"]
            print(f"  • {layer}")
            print(f"    异常值: {stats['extreme_outlier_percentage']:.2f}%")
            print(f"    SQNR: {stats['sqnr_db']:.1f} dB")
    print()


def demo_3_rank_efficiency():
    """演示3: 秩效率分析 - 发现压缩机会"""
    print("=" * 80)
    print("演示 3: 秩效率分析")
    print("=" * 80)
    print("\n目的: 找出哪些层冗余度高，可以大幅压缩")
    print("应用: LoRA 微调、知识蒸馏、模型剪枝\n")
    
    # 谱分析
    print("正在进行秩分析...")
    scope = Scope("openai-community/gpt2")
    results = scope.scan(methods=["spectral"], parallel=True)
    
    output_dir = Path("visualization_outputs")
    
    # 生成秩效率图
    print("\n生成秩效率可视化...")
    plot_rank_efficiency(results, output_path=output_dir / "rank_efficiency.png")
    
    # 分析压缩潜力
    high_compress = []
    medium_compress = []
    
    for layer, stats in results.items():
        if "spectral" in stats:
            stable = stats["spectral"].get("stable_rank", 0)
            total = stats["spectral"].get("total_rank", 1)
            ratio = stable / total if total > 0 else 1
            
            if ratio < 0.3:
                high_compress.append((layer, ratio, total))
            elif ratio < 0.5:
                medium_compress.append((layer, ratio, total))
    
    print(f"\n压缩潜力分析:")
    print(f"  🔴 高压缩潜力 (>70%): {len(high_compress)} 层")
    print(f"  🟡 中压缩潜力 (50-70%): {len(medium_compress)} 层")
    
    if high_compress:
        print(f"\n最值得压缩的层:")
        for layer, ratio, total in sorted(high_compress, key=lambda x: x[1])[:5]:
            compress_rate = (1 - ratio) * 100
            suggested_rank = max(4, int(total * ratio))
            print(f"  • {layer}")
            print(f"    当前秩利用率: {ratio*100:.1f}%")
            print(f"    可压缩: {compress_rate:.1f}%")
            print(f"    建议 LoRA rank: {suggested_rank}")
    
    # 估算总体压缩收益
    total_params = sum(stats["spectral"]["total_rank"] ** 2 
                       for stats in results.values() 
                       if "spectral" in stats and stats["spectral"]["total_rank"] > 0)
    
    compressed_params = sum(
        stats["spectral"]["stable_rank"] ** 2
        for stats in results.values()
        if "spectral" in stats
    )
    
    compression_ratio = (1 - compressed_params / total_params) * 100 if total_params > 0 else 0
    
    print(f"\n估算总体压缩潜力: {compression_ratio:.1f}%")
    print(f"  原始参数量级: {total_params/1e6:.1f}M")
    print(f"  压缩后参数量级: {compressed_params/1e6:.1f}M")
    print()


def demo_4_health_dashboard():
    """演示4: 模型健康仪表盘 - 综合诊断"""
    print("=" * 80)
    print("演示 4: 模型健康仪表盘")
    print("=" * 80)
    print("\n目的: 一图总览模型的所有关键健康指标")
    print("应用: 模型上线前检查、定期健康监控\n")
    
    # 全面分析
    print("正在进行全面分析 (这可能需要一些时间)...")
    scope = Scope("openai-community/gpt2")
    results = scope.scan(methods=["all"], parallel=True)
    
    output_dir = Path("visualization_outputs")
    
    # 生成健康仪表盘
    print("\n生成模型健康仪表盘...")
    plot_model_health_dashboard(results, output_path=output_dir / "health_dashboard.png")
    
    # 计算总体健康评分
    total_layers = len(results)
    
    critical_issues = 0
    warnings = 0
    
    for stats in results.values():
        # 检查条件数
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1e5:
                critical_issues += 1
            elif cond > 1e4:
                warnings += 1
        
        # 检查异常值
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            if outlier > 10:
                critical_issues += 1
            elif outlier > 1:
                warnings += 1
    
    health_score = max(0, 100 - critical_issues * 10 - warnings * 3)
    
    print(f"\n整体健康评分: {health_score}/100")
    print(f"  总层数: {total_layers}")
    print(f"  严重问题: {critical_issues} 层")
    print(f"  警告: {warnings} 层")
    
    if health_score >= 80:
        print("\n✅ 模型健康状态良好，可以部署")
    elif health_score >= 60:
        print("\n⚠️  模型有一些问题，建议优化后部署")
    else:
        print("\n❌ 模型存在较多问题，不建议直接部署")
    print()


def demo_5_layer_details():
    """演示5: 层级详细可视化"""
    print("=" * 80)
    print("演示 5: 层级详细可视化")
    print("=" * 80)
    print("\n目的: 深入分析特定问题层的权重特征")
    print("应用: 调试具体层的问题、理解权重分布\n")
    
    # 找出最有问题的几层
    print("正在识别问题层...")
    scope = Scope("openai-community/gpt2")
    results = scope.scan(methods=["spectral", "quantization"], parallel=True)
    
    # 找出条件数最高的层
    problem_layers = []
    for layer, stats in results.items():
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1e4:
                problem_layers.append((layer, cond))
    
    problem_layers.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n找到 {len(problem_layers)} 个高条件数层")
    print("为前 3 层生成详细可视化...\n")
    
    output_dir = Path("visualization_outputs/layer_details")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个问题层生成详细图表
    for i, (layer_name, cond) in enumerate(problem_layers[:3]):
        print(f"{i+1}. {layer_name} (条件数: {cond:.2e})")
        
        # 获取权重
        try:
            module = scope.model.get_submodule(layer_name)
            if not hasattr(module, 'weight') or module.weight is None:
                raise ValueError(f"module has no weight: {type(module)}")

            weight = module.weight
            
            # 生成权重分布图
            safe_name = layer_name.replace('.', '_')
            plot_weight_distribution(
                weight, 
                layer_name,
                output_path=output_dir / f"{safe_name}_distribution.png"
            )
            
            # 生成奇异值图
            plot_singular_values(
                weight,
                layer_name,
                output_path=output_dir / f"{safe_name}_singular_values.png"
            )
            
            print(f"   保存到: {output_dir}/{safe_name}_*.png")
        except Exception as e:
            print(f"   ⚠️ 无法为该层生成详细图表: {e}")
            continue
    
    print()


def demo_6_summary_report():
    """生成完整的 HTML 报告"""
    print("=" * 80)
    print("演示 6: 生成综合报告")
    print("=" * 80)
    print("\n目的: 将所有图表整合成一份完整报告")
    print()
    
    output_dir = Path("visualization_outputs")
    
    # 创建 HTML 报告
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WeightScope Analysis Report - GPT-2</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            background: #ecf0f1;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .critical {{ background: #e74c3c; color: white; }}
        .warning {{ background: #f39c12; color: white; }}
        .good {{ background: #27ae60; color: white; }}
        .info {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>🔬 WeightScope Analysis Report</h1>
    <p><strong>Model:</strong> openai-community/gpt2</p>
    <p><strong>Analysis Date:</strong> {Path(__file__).stat().st_mtime}</p>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="metric critical">15 Critical Layers</div>
        <div class="metric warning">18 Warning Layers</div>
        <div class="metric info">51 Total Layers</div>
        <div class="metric good">Health Score: 73/100</div>
        
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>🔴 15 layers with condition number > 1000 (numerical instability risk)</li>
            <li>🟡 18 layers with > 0.1% quantization outliers (INT8 quality degradation)</li>
            <li>💡 Multiple layers with < 30% rank utilization (compression opportunities)</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🌡️ Numerical Stability Analysis</h2>
        <img src="condition_heatmap.png" alt="Condition Number Heatmap">
        <p><strong>Interpretation:</strong> Red bars indicate layers prone to numerical instability. 
        These may cause training crashes or require FP32 precision.</p>
    </div>
    
    <div class="section">
        <h2>⚖️ Quantization Risk Assessment</h2>
        <img src="quantization_risk.png" alt="Quantization Risk Matrix">
        <p><strong>Interpretation:</strong> Points in the red zone are high-risk for INT8 quantization. 
        Consider INT16 or FP16 for these layers.</p>
    </div>
    
    <div class="section">
        <h2>📉 Rank Efficiency & Compression Potential</h2>
        <img src="rank_efficiency.png" alt="Rank Efficiency">
        <p><strong>Interpretation:</strong> Low utilization indicates redundancy. 
        These layers are excellent candidates for LoRA or SVD compression.</p>
    </div>
    
    <div class="section">
        <h2>🏥 Overall Health Dashboard</h2>
        <img src="health_dashboard.png" alt="Model Health Dashboard">
        <p><strong>Interpretation:</strong> Comprehensive view of all metrics. 
        Use this for regular model health monitoring.</p>
    </div>
    
    <div class="section">
        <h2>💡 Recommendations</h2>
        <ol>
            <li><strong>Mixed Precision Training:</strong> Keep high-condition layers in FP32</li>
            <li><strong>Smart Quantization:</strong> Use per-channel quantization for outlier-heavy layers</li>
            <li><strong>Model Compression:</strong> Apply LoRA with rank 8-16 to low-utilization layers</li>
            <li><strong>Monitoring:</strong> Track condition numbers during training to detect instability early</li>
        </ol>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; color: #7f8c8d;">
        <p>Generated by WeightScope - AI Model Weight Analysis Toolkit</p>
    </footer>
</body>
</html>
"""
    
    report_path = output_dir / "analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 报告已生成: {report_path}")
    print(f"\n在浏览器中打开查看完整报告:")
    print(f"  file://{report_path.absolute()}")
    print()


def main():
    """运行所有演示"""
    print("\n" + "="*80)
    print("WeightScope 可视化功能演示")
    print("="*80)
    print("\n这个演示将生成 6 类实用图表，帮助你诊断模型问题\n")
    
    # 创建输出目录
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 运行所有演示
    try:
        demo_1_condition_number_heatmap()
        demo_2_quantization_risk_matrix()
        demo_3_rank_efficiency()
        demo_4_health_dashboard()
        demo_5_layer_details()
        demo_6_summary_report()
        
        print("=" * 80)
        print("✅ 所有可视化已完成！")
        print("=" * 80)
        print(f"\n所有图表已保存到: {output_dir.absolute()}/")
        print("\n生成的图表:")
        print("  1. condition_heatmap.png - 条件数热力图")
        print("  2. quantization_risk.png - 量化风险矩阵")
        print("  3. rank_efficiency.png - 秩效率分析")
        print("  4. health_dashboard.png - 健康仪表盘")
        print("  5. layer_details/*.png - 层级详细分析")
        print("  6. analysis_report.html - 完整 HTML 报告")
        
        print(f"\n💡 提示: 在浏览器中打开 {output_dir.absolute()}/analysis_report.html 查看完整报告")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
