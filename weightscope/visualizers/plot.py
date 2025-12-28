import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from pathlib import Path
import math
import re

# Best-effort font config (won't error if fonts are missing)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


_H_BLOCK_RE = re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)")


def _short_layer_label(full_name: str, max_len: int = 28) -> str:
    """Create a short, meaningful label for a layer path.

    Goal: keep semantic context (block index + sub-structure) and avoid duplicates like 'c_proj'.
    """
    if not full_name:
        return ""

    name = str(full_name)
    parts = name.split('.')

    # Extract transformer block index if present: transformer.h.<idx>...
    block_match = _H_BLOCK_RE.search(name)
    block = f"h{block_match.group(1)}" if block_match else ""

    # Common GPT-style modules
    if "attn" in parts:
        # e.g. transformer.h.9.attn.c_proj -> h9.attn.c_proj
        tail = parts[-2:] if len(parts) >= 2 else parts
        core = [p for p in [block, "attn", *tail] if p]
    elif "mlp" in parts:
        tail = parts[-2:] if len(parts) >= 2 else parts
        core = [p for p in [block, "mlp", *tail] if p]
    else:
        # embeddings / lm_head / others
        tail = parts[-3:] if len(parts) >= 3 else parts
        core = [p for p in [block, *tail] if p]

    label = ".".join(core)

    # If still too long, compress from the left.
    if len(label) > max_len:
        label = "…" + label[-(max_len - 1):]
    return label


def _dedupe_labels(full_names, max_len: int = 28):
    """Generate short labels and ensure uniqueness with minimal suffixing."""
    labels = [_short_layer_label(n, max_len=max_len) for n in full_names]
    seen = {}
    out = []
    for full, label in zip(full_names, labels):
        key = label
        if key not in seen:
            seen[key] = 1
            out.append(label)
            continue
        seen[key] += 1
        # Add a small disambiguating suffix using trailing path parts.
        parts = str(full).split('.')
        suffix = parts[-3] if len(parts) >= 3 else parts[0]
        candidate = f"{label}·{suffix}"
        if len(candidate) > max_len:
            candidate = candidate[: max_len - 1] + "…"
        out.append(candidate)
    return out


def _tick_step(n: int, target: int = 40) -> int:
    return max(1, int(math.ceil(n / max(1, target))))

def plot_weight_distribution(tensor: torch.Tensor, layer_name: str, output_path=None):
    """
    Plot histogram of weight distribution.
    """
    weights = tensor.detach().float().cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(weights, bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Weight Distribution: {layer_name}')
    axes[0].grid(True, alpha=0.3)
    
    # Log-scale histogram for better visibility
    axes[1].hist(weights, bins=100, edgecolor='black', alpha=0.7, log=True)
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title(f'Weight Distribution (Log Scale): {layer_name}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_singular_values(tensor: torch.Tensor, layer_name: str, output_path=None, top_k=50):
    """
    Plot singular value spectrum.
    """
    t = tensor.detach().float().cpu()
    
    # Reshape if needed
    if len(t.shape) > 2:
        t = t.reshape(-1, t.shape[-1])
    elif len(t.shape) == 1:
        t = t.unsqueeze(1)
    
    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(t, full_matrices=False)
        singular_values = S.numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        k = min(top_k, len(singular_values))
        axes[0].plot(range(1, k+1), singular_values[:k], marker='o', markersize=3)
        axes[0].set_xlabel('Singular Value Index')
        axes[0].set_ylabel('Singular Value')
        axes[0].set_title(f'Top {k} Singular Values: {layer_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Log scale
        axes[1].semilogy(range(1, len(singular_values)+1), singular_values, marker='o', markersize=2)
        axes[1].set_xlabel('Singular Value Index')
        axes[1].set_ylabel('Singular Value (log scale)')
        axes[1].set_title(f'All Singular Values (Log Scale): {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Error plotting singular values: {e}")


def plot_weight_heatmap(tensor: torch.Tensor, layer_name: str, output_path=None, max_size=100):
    """
    Plot heatmap of weight matrix (downsampled if too large).
    """
    t = tensor.detach().float().cpu().numpy()
    
    # For high-dimensional tensors, reshape to 2D
    if len(t.shape) > 2:
        t = t.reshape(t.shape[0], -1)
    elif len(t.shape) == 1:
        t = t.reshape(-1, 1)
    
    # Downsample if too large
    if t.shape[0] > max_size or t.shape[1] > max_size:
        stride_0 = max(1, t.shape[0] // max_size)
        stride_1 = max(1, t.shape[1] // max_size)
        t = t[::stride_0, ::stride_1]
        title_suffix = f" (downsampled {stride_0}x{stride_1})"
    else:
        title_suffix = ""
    
    plt.figure(figsize=(10, 8))
    plt.imshow(t, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Weight Value')
    plt.title(f'Weight Heatmap: {layer_name}{title_suffix}')
    plt.xlabel('Output Dimension')
    plt.ylabel('Input Dimension')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_analysis_report(results: dict, output_dir: str):
    """
    Create a comprehensive visual report with multiple plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visual report in {output_dir}...")
    
    # Summary statistics plot
    if results:
        layer_names = []
        means = []
        stds = []
        
        for layer_name, stats in results.items():
            if "basic_stats" in stats:
                layer_names.append(layer_name)
                means.append(stats["basic_stats"]["mean"])
                stds.append(stats["basic_stats"]["std"])
        
        if means:
            fig, ax = plt.subplots(figsize=(max(14, len(layer_names) * 0.35), 6))
            x = np.arange(len(layer_names))
            ax.bar(x, stds, alpha=0.7, label='Std Dev')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Weight Standard Deviation Across Layers')

            labels = _dedupe_labels(layer_names, max_len=18)
            step = _tick_step(len(labels), target=45)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=90, fontsize=7)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.35)
            plt.savefig(output_path / "summary_std.png", dpi=150)
            plt.close()
            
            print(f"Created summary plots in {output_dir}")

    # Extended report plots
    try:
        plot_condition_number_heatmap(results, output_path=output_path / "condition_heatmap.png")
    except Exception as e:
        print(f"Failed to create condition heatmap: {e}")

    try:
        plot_quantization_risk_matrix(results, output_path=output_path / "quantization_risk.png")
    except Exception as e:
        print(f"Failed to create quantization risk matrix: {e}")

    try:
        plot_rank_efficiency(results, output_path=output_path / "rank_efficiency.png")
    except Exception as e:
        print(f"Failed to create rank efficiency plot: {e}")

    try:
        plot_model_health_dashboard(results, output_path=output_path / "health_dashboard.png")
    except Exception as e:
        print(f"Failed to create health dashboard: {e}")


def plot_condition_number_heatmap(results: dict, output_path=None):
    """
    绘制条件数热力图，快速识别数值不稳定的层
    """
    layer_names = []
    condition_numbers = []
    layer_types = []
    
    for layer_name, stats in results.items():
        if "spectral" in stats and "condition_number" in stats["spectral"]:
            cond = stats["spectral"]["condition_number"]
            if not np.isnan(cond) and not np.isinf(cond):
                layer_names.append(layer_name)
                condition_numbers.append(cond)
                
                # 识别层类型
                if "attn" in layer_name.lower():
                    layer_types.append("Attention")
                elif "mlp" in layer_name.lower() or "fc" in layer_name.lower():
                    layer_types.append("MLP")
                elif "embed" in layer_name.lower():
                    layer_types.append("Embedding")
                else:
                    layer_types.append("Other")
    
    if not condition_numbers:
        print("No condition number data available")
        return
    
    # Sort by condition number so the plot is interpretable
    pairs = sorted(zip(layer_names, condition_numbers, layer_types), key=lambda x: x[1], reverse=True)
    layer_names = [p[0] for p in pairs]
    condition_numbers = [p[1] for p in pairs]
    layer_types = [p[2] for p in pairs]

    # 创建图表
    fig, ax = plt.subplots(figsize=(16, max(8, len(layer_names) * 0.33)))
    
    # 对条件数取对数以便可视化
    log_conds = np.log10(np.array(condition_numbers))
    
    # 创建颜色映射
    colors = []
    for cond in condition_numbers:
        if cond > 1e5:
            colors.append('#d62728')  # 红色：严重
        elif cond > 1e4:
            colors.append('#ff7f0e')  # 橙色：警告
        elif cond > 1e3:
            colors.append('#ffdd57')  # 黄色：注意
        else:
            colors.append('#2ca02c')  # 绿色：安全
    
    # 绘制水平条形图
    y_pos = np.arange(len(layer_names))
    bars = ax.barh(y_pos, log_conds, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    y_labels = _dedupe_labels(layer_names, max_len=32)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Condition Number (log10)', fontsize=12)
    ax.set_title('Layer Condition Number Heatmap - Numerical Stability Analysis', 
                 fontsize=14, fontweight='bold')
    
    # 添加参考线
    ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, label='Safe (10³)')
    ax.axvline(x=4, color='orange', linestyle='--', alpha=0.5, label='Warning (10⁴)')
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Critical (10⁵)')
    
    # 添加数值标签（仅前 N 个，避免文字堆叠）
    for i, (bar, cond) in enumerate(zip(bars, condition_numbers)):
        if i >= 25:
            break
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f'{cond:.1e}', va='center', fontsize=7)
    
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    # Leave room for long y labels
    fig.subplots_adjust(left=0.32)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved condition number heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quantization_risk_matrix(results: dict, output_path=None):
    """
    绘制量化风险矩阵（异常值比例 vs SQNR）
    """
    layer_names = []
    outlier_pcts = []
    sqnrs = []
    layer_types = []
    
    for layer_name, stats in results.items():
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            sqnr = stats["quantization"].get("sqnr_db", 0)
            
            if sqnr > 0:  # 有效数据
                layer_names.append(layer_name)
                outlier_pcts.append(outlier)
                sqnrs.append(sqnr)
                
                if "attn" in layer_name.lower():
                    layer_types.append("Attention")
                elif "mlp" in layer_name.lower() or "fc" in layer_name.lower():
                    layer_types.append("MLP")
                elif "embed" in layer_name.lower():
                    layer_types.append("Embedding")
                else:
                    layer_types.append("Other")
    
    if not layer_names:
        print("No quantization data available")
        return
    
    # 创建散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 根据层类型设置颜色
    type_colors = {
        "Attention": '#1f77b4',
        "MLP": '#ff7f0e',
        "Embedding": '#2ca02c',
        "Other": '#7f7f7f'
    }
    
    for ltype in set(layer_types):
        mask = [t == ltype for t in layer_types]
        x = [outlier_pcts[i] for i in range(len(mask)) if mask[i]]
        y = [sqnrs[i] for i in range(len(mask)) if mask[i]]
        ax.scatter(x, y, c=type_colors[ltype], label=ltype, s=100, alpha=0.6, edgecolors='black')
    
    # 添加风险区域
    ax.axhspan(0, 30, alpha=0.1, color='red', label='High Risk (SQNR < 30 dB)')
    ax.axhspan(30, 40, alpha=0.1, color='yellow', label='Medium Risk (30-40 dB)')
    ax.axvspan(1, 100, alpha=0.05, color='red')
    
    # 添加决策边界线
    ax.axhline(y=40, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Safe SQNR Threshold')
    ax.axvline(x=1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Outlier Threshold')
    
    ax.set_xlabel('Extreme Outlier Percentage (%)', fontsize=12)
    ax.set_ylabel('SQNR (dB)', fontsize=12)
    ax.set_title('Quantization Risk Matrix - INT8 Suitability Analysis', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, max(outlier_pcts) * 1.1)
    ax.set_ylim(min(sqnrs) * 0.9, max(sqnrs) * 1.05)
    
    # 标注高风险层：限制数量避免文字堆叠
    risk_scores = []
    for name, outlier, sqnr in zip(layer_names, outlier_pcts, sqnrs):
        score = 0
        if outlier > 5:
            score += outlier
        if sqnr < 30:
            score += (30 - sqnr) * 2
        if score > 0:
            risk_scores.append((score, name, outlier, sqnr))
    risk_scores.sort(reverse=True)
    for idx, (_, name, outlier, sqnr) in enumerate(risk_scores[:12]):
        short_name = _short_layer_label(name, max_len=26)
        ax.annotate(short_name, (outlier, sqnr), fontsize=7, alpha=0.8,
                    xytext=(6, 6 + (idx % 3) * 6), textcoords='offset points')

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved quantization risk matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rank_efficiency(results: dict, output_path=None):
    """
    绘制秩利用率图，识别可压缩的层
    """
    layer_names = []
    rank_ratios = []
    stable_ranks = []
    total_ranks = []
    
    for layer_name, stats in results.items():
        if "spectral" in stats:
            stable_rank = stats["spectral"].get("stable_rank", 0)
            total_rank = stats["spectral"].get("total_rank", 1)
            
            if total_rank > 0:
                layer_names.append(layer_name)
                rank_ratios.append(stable_rank / total_rank * 100)
                stable_ranks.append(stable_rank)
                total_ranks.append(total_rank)
    
    if not layer_names:
        print("No spectral data available")
        return
    
    # 创建双子图
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 子图1：秩利用率条形图
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(layer_names))
    colors_rank = ['#d62728' if r < 10 else '#ff7f0e' if r < 30 else '#2ca02c' 
                   for r in rank_ratios]
    
    bars = ax1.bar(x_pos, rank_ratios, color=colors_rank, alpha=0.8)
    ax1.axhline(y=30, color='orange', linestyle='--', label='30% Threshold', alpha=0.7)
    ax1.axhline(y=10, color='red', linestyle='--', label='10% Critical', alpha=0.7)
    
    ax1.set_ylabel('Rank Utilization (%)', fontsize=12)
    ax1.set_title('Layer Rank Efficiency - Compression Potential Analysis', 
                  fontsize=14, fontweight='bold')
    labels = _dedupe_labels(layer_names, max_len=16)
    step = _tick_step(len(labels), target=35)
    ax1.set_xticks(x_pos[::step])
    ax1.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                        rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加压缩潜力标签（稀疏标注，避免重叠）
    step_note = _tick_step(len(rank_ratios), target=18)
    for i, (bar, ratio) in enumerate(zip(bars, rank_ratios)):
        if ratio < 30 and (i % step_note == 0):
            compression = 100 - ratio
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f'-{compression:.0f}%', ha='center', fontsize=7, color='red')
    
    # 子图2：稳定秩 vs 总秩散点图
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(total_ranks, stable_ranks, c=rank_ratios, 
                         cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
    
    # 添加对角线（完美利用）
    max_rank = max(total_ranks)
    ax2.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.3, label='Perfect Utilization')
    
    ax2.set_xlabel('Total Rank', fontsize=11)
    ax2.set_ylabel('Stable Rank', fontsize=11)
    ax2.set_title('Stable Rank vs Total Rank', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Utilization %', fontsize=10)
    
    # 子图3：压缩建议饼图
    ax3 = fig.add_subplot(gs[1, 1])
    
    compression_categories = {
        'High Potential (>70% compress)': sum(1 for r in rank_ratios if r < 30),
        'Medium Potential (50-70%)': sum(1 for r in rank_ratios if 30 <= r < 50),
        'Low Potential (30-50%)': sum(1 for r in rank_ratios if 50 <= r < 70),
        'Already Efficient': sum(1 for r in rank_ratios if r >= 70)
    }
    
    colors_pie = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c']
    values = list(compression_categories.values())
    labels_pie = list(compression_categories.keys())
    wedges, _texts, autotexts = ax3.pie(
        values,
        labels=None,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        textprops={'fontsize': 9}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax3.set_title('Compression Opportunity Distribution', fontsize=12)
    legend_labels = [f"{k} ({v})" for k, v in zip(labels_pie, values)]
    ax3.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved rank efficiency plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_health_dashboard(results: dict, output_path=None):
    """
    创建模型健康度仪表盘，一图总览所有关键指标
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 收集所有数据
    layer_names = list(results.keys())
    
    # 1. 条件数分布（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    cond_nums = []
    for stats in results.values():
        if "spectral" in stats and "condition_number" in stats["spectral"]:
            cond = stats["spectral"]["condition_number"]
            if not np.isnan(cond) and not np.isinf(cond):
                cond_nums.append(cond)
    
    if cond_nums:
        ax1.hist(np.log10(cond_nums), bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax1.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Safe')
        ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax1.set_xlabel('Condition Number (log10)')
        ax1.set_ylabel('Layer Count')
        ax1.set_title('Numerical Stability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 异常值百分比分布（中上）
    ax2 = fig.add_subplot(gs[0, 1])
    outlier_pcts = []
    for stats in results.values():
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            outlier_pcts.append(outlier)
    
    if outlier_pcts:
        ax2.hist(outlier_pcts, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
        ax2.axvline(x=1, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax2.set_xlabel('Outlier Percentage (%)')
        ax2.set_ylabel('Layer Count')
        ax2.set_title('Quantization Outlier Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 秩利用率分布（右上）
    ax3 = fig.add_subplot(gs[0, 2])
    rank_ratios = []
    for stats in results.values():
        if "spectral" in stats:
            stable = stats["spectral"].get("stable_rank", 0)
            total = stats["spectral"].get("total_rank", 1)
            if total > 0:
                rank_ratios.append(stable / total * 100)
    
    if rank_ratios:
        ax3.hist(rank_ratios, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax3.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Compress Threshold')
        ax3.set_xlabel('Rank Utilization (%)')
        ax3.set_ylabel('Layer Count')
        ax3.set_title('Rank Efficiency Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. SQNR分布（左中）
    ax4 = fig.add_subplot(gs[1, 0])
    sqnrs = []
    for stats in results.values():
        if "quantization" in stats:
            sqnr = stats["quantization"].get("sqnr_db", 0)
            if sqnr > 0:
                sqnrs.append(sqnr)
    
    if sqnrs:
        ax4.hist(sqnrs, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
        ax4.axvline(x=40, color='green', linestyle='--', alpha=0.7, label='Safe (>40 dB)')
        ax4.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Risk (<30 dB)')
        ax4.set_xlabel('SQNR (dB)')
        ax4.set_ylabel('Layer Count')
        ax4.set_title('Quantization Quality (SQNR)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 稀疏度分布（中中）
    ax5 = fig.add_subplot(gs[1, 1])
    sparsities = []
    for stats in results.values():
        if "sparsity" in stats:
            sparsity = stats["sparsity"].get("sparsity_0.001", 0)
            sparsities.append(sparsity * 100)
    
    if sparsities:
        ax5.hist(sparsities, bins=30, color='#8c564b', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Sparsity (%)')
        ax5.set_ylabel('Layer Count')
        ax5.set_title('Weight Sparsity Distribution (threshold=0.001)')
        ax5.grid(True, alpha=0.3)
    
    # 6. 健康度评分饼图（右中）
    ax6 = fig.add_subplot(gs[1, 2])
    
    health_scores = {
        'Excellent': 0,
        'Good': 0,
        'Warning': 0,
        'Critical': 0
    }
    
    for stats in results.values():
        score = 100
        
        # 检查条件数
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1e5:
                score -= 40
            elif cond > 1e4:
                score -= 20
        
        # 检查异常值
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            if outlier > 10:
                score -= 30
            elif outlier > 1:
                score -= 15
        
        # 分类
        if score >= 80:
            health_scores['Excellent'] += 1
        elif score >= 60:
            health_scores['Good'] += 1
        elif score >= 40:
            health_scores['Warning'] += 1
        else:
            health_scores['Critical'] += 1
    
    colors_health = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    values = list(health_scores.values())
    labels = list(health_scores.keys())
    wedges, _texts, autotexts = ax6.pie(
        values,
        labels=None,
        colors=colors_health,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        textprops={'fontsize': 9}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax6.set_title('Overall Layer Health Distribution')
    ax6.legend(wedges, [f"{k} ({v})" for k, v in zip(labels, values)],
               loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)
    
    # 7. Top-10 问题层（底部，横跨所有列）
    ax7 = fig.add_subplot(gs[2, :])
    
    # 计算每层的综合风险分数
    layer_risks = []
    for name, stats in results.items():
        risk = 0
        
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1e5:
                risk += 50
            elif cond > 1e4:
                risk += 25
        
        if "quantization" in stats:
            outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
            risk += outlier * 3
        
        layer_risks.append((name, risk))
    
    # 排序并取前10
    layer_risks.sort(key=lambda x: x[1], reverse=True)
    top_10 = layer_risks[:10]
    
    if top_10:
        names = _dedupe_labels([item[0] for item in top_10], max_len=30)
        risks = [item[1] for item in top_10]
        
        colors_risk = ['#d62728' if r > 50 else '#ff7f0e' if r > 25 else '#ffdd57' 
                      for r in risks]
        
        y_pos = np.arange(len(names))
        bars = ax7.barh(y_pos, risks, color=colors_risk, alpha=0.8)
        
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(names, fontsize=10)
        ax7.set_xlabel('Risk Score', fontsize=12)
        ax7.set_title('Top 10 Highest Risk Layers', fontsize=13, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
        ax7.invert_yaxis()
        # Leave extra room for labels
        ax7.margins(y=0.05)
        
        # 添加风险分数标签
        for bar, risk in zip(bars, risks):
                ax7.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{risk:.1f}', va='center', fontsize=9)
    
    # 总标题
    fig.suptitle('Model Health Dashboard - Comprehensive Analysis',
                 fontsize=18, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved health dashboard to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_layer_comparison(results1: dict, results2: dict, 
                          model1_name: str = "Model 1", 
                          model2_name: str = "Model 2",
                          output_path=None):
    """
    对比两个模型的层级指标
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 找到共同的层
    common_layers = set(results1.keys()) & set(results2.keys())
    if not common_layers:
        print("No common layers found")
        return
    
    common_layers = sorted(list(common_layers))[:20]  # 最多20层
    
    # 1. 条件数对比
    ax1 = axes[0, 0]
    cond1 = []
    cond2 = []
    layer_labels = []
    
    for layer in common_layers:
        if ("spectral" in results1[layer] and "spectral" in results2[layer]):
            c1 = results1[layer]["spectral"].get("condition_number", 0)
            c2 = results2[layer]["spectral"].get("condition_number", 0)
            if c1 > 0 and c2 > 0:
                cond1.append(np.log10(c1))
                cond2.append(np.log10(c2))
                layer_labels.append(_short_layer_label(layer, max_len=18))
    
    if cond1:
        x = np.arange(len(layer_labels))
        width = 0.35
        
        ax1.bar(x - width/2, cond1, width, label=model1_name, alpha=0.8, color='#1f77b4')
        ax1.bar(x + width/2, cond2, width, label=model2_name, alpha=0.8, color='#ff7f0e')
        
        ax1.set_ylabel('Condition Number (log10)')
        ax1.set_title('Condition Number Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. 秩利用率对比
    ax2 = axes[0, 1]
    rank1 = []
    rank2 = []
    
    for layer in common_layers:
        if ("spectral" in results1[layer] and "spectral" in results2[layer]):
            s1 = results1[layer]["spectral"].get("stable_rank", 0)
            t1 = results1[layer]["spectral"].get("total_rank", 1)
            s2 = results2[layer]["spectral"].get("stable_rank", 0)
            t2 = results2[layer]["spectral"].get("total_rank", 1)
            
            if t1 > 0 and t2 > 0:
                rank1.append(s1/t1 * 100)
                rank2.append(s2/t2 * 100)
    
    if rank1:
        x = np.arange(len(rank1))
        width = 0.35
        
        ax2.bar(x - width/2, rank1, width, label=model1_name, alpha=0.8, color='#2ca02c')
        ax2.bar(x + width/2, rank2, width, label=model2_name, alpha=0.8, color='#d62728')
        
        ax2.set_ylabel('Rank Utilization (%)')
        ax2.set_title('Rank Efficiency Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_labels[:len(rank1)], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. SQNR对比
    ax3 = axes[1, 0]
    sqnr1 = []
    sqnr2 = []
    
    for layer in common_layers:
        if ("quantization" in results1[layer] and "quantization" in results2[layer]):
            sq1 = results1[layer]["quantization"].get("sqnr_db", 0)
            sq2 = results2[layer]["quantization"].get("sqnr_db", 0)
            
            if sq1 > 0 and sq2 > 0:
                sqnr1.append(sq1)
                sqnr2.append(sq2)
    
    if sqnr1:
        x = np.arange(len(sqnr1))
        width = 0.35
        
        ax3.bar(x - width/2, sqnr1, width, label=model1_name, alpha=0.8, color='#9467bd')
        ax3.bar(x + width/2, sqnr2, width, label=model2_name, alpha=0.8, color='#8c564b')
        
        ax3.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Safe Threshold')
        ax3.set_ylabel('SQNR (dB)')
        ax3.set_title('Quantization Quality Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_labels[:len(sqnr1)], rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. 整体指标对比雷达图
    ax4 = axes[1, 1]
    
    # 计算综合指标
    metrics = {
        'Numerical\nStability': 0,
        'Quantization\nQuality': 0,
        'Rank\nEfficiency': 0,
        'Sparsity': 0,
        'Overall\nHealth': 0
    }
    
    def calc_score(results):
        scores = []
        
        # 数值稳定性得分
        conds = [s["spectral"].get("condition_number", 0) for s in results.values() 
                if "spectral" in s]
        conds = [c for c in conds if not np.isnan(c) and not np.isinf(c) and c > 0]
        if conds:
            avg_cond = np.mean(np.log10(conds))
            scores.append(max(0, 100 - (avg_cond - 2) * 20))  # 越低越好
        
        # 量化质量得分
        sqnrs = [s["quantization"].get("sqnr_db", 0) for s in results.values() 
                if "quantization" in s]
        sqnrs = [sq for sq in sqnrs if sq > 0]
        if sqnrs:
            scores.append(min(100, np.mean(sqnrs) * 2))  # 越高越好
        
        # 秩效率得分
        ranks = []
        for s in results.values():
            if "spectral" in s:
                stable = s["spectral"].get("stable_rank", 0)
                total = s["spectral"].get("total_rank", 1)
                if total > 0:
                    ranks.append(stable / total * 100)
        if ranks:
            scores.append(np.mean(ranks))
        
        # 稀疏度得分
        sparsities = [s["sparsity"].get("sparsity_0.001", 0) * 100 
                     for s in results.values() if "sparsity" in s]
        if sparsities:
            scores.append(np.mean(sparsities))
        
        # 总体健康得分
        if scores:
            scores.append(np.mean(scores))
        
        return scores
    
    scores1 = calc_score(results1)
    scores2 = calc_score(results2)
    
    if scores1 and scores2:
        categories = list(metrics.keys())
        N = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores1 += scores1[:1]
        scores2 += scores2[:1]
        angles += angles[:1]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, scores1, 'o-', linewidth=2, label=model1_name, color='#1f77b4')
        ax4.fill(angles, scores1, alpha=0.25, color='#1f77b4')
        ax4.plot(angles, scores2, 'o-', linewidth=2, label=model2_name, color='#ff7f0e')
        ax4.fill(angles, scores2, alpha=0.25, color='#ff7f0e')
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=9)
        ax4.set_ylim(0, 100)
        ax4.set_title('Overall Metrics Comparison', fontsize=12, pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_evolution(checkpoint_results: dict, output_path=None):
    """
    可视化训练过程中权重指标的演化
    
    Args:
        checkpoint_results: {checkpoint_name: results_dict}
    """
    if len(checkpoint_results) < 2:
        print("Need at least 2 checkpoints for evolution plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    checkpoints = sorted(checkpoint_results.keys())
    
    # 选择几个代表性的层进行追踪
    first_results = checkpoint_results[checkpoints[0]]
    tracked_layers = list(first_results.keys())[:5]  # 追踪前5层
    
    # 1. 条件数演化
    ax1 = axes[0, 0]
    for layer in tracked_layers:
        conds = []
        steps = []
        
        for i, ckpt in enumerate(checkpoints):
            if layer in checkpoint_results[ckpt]:
                stats = checkpoint_results[ckpt][layer]
                if "spectral" in stats:
                    cond = stats["spectral"].get("condition_number", 0)
                    if cond > 0 and not np.isnan(cond) and not np.isinf(cond):
                        conds.append(np.log10(cond))
                        steps.append(i)
        
        if conds:
            ax1.plot(steps, conds, marker='o', label=_short_layer_label(layer, max_len=18))
    
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Condition Number (log10)')
    ax1.set_title('Numerical Stability Evolution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 秩利用率演化
    ax2 = axes[0, 1]
    for layer in tracked_layers:
        ranks = []
        steps = []
        
        for i, ckpt in enumerate(checkpoints):
            if layer in checkpoint_results[ckpt]:
                stats = checkpoint_results[ckpt][layer]
                if "spectral" in stats:
                    stable = stats["spectral"].get("stable_rank", 0)
                    total = stats["spectral"].get("total_rank", 1)
                    if total > 0:
                        ranks.append(stable / total * 100)
                        steps.append(i)
        
        if ranks:
            ax2.plot(steps, ranks, marker='o', label=_short_layer_label(layer, max_len=18))
    
    ax2.set_xlabel('Checkpoint')
    ax2.set_ylabel('Rank Utilization (%)')
    ax2.set_title('Rank Efficiency Evolution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. SQNR演化
    ax3 = axes[1, 0]
    for layer in tracked_layers:
        sqnrs = []
        steps = []
        
        for i, ckpt in enumerate(checkpoints):
            if layer in checkpoint_results[ckpt]:
                stats = checkpoint_results[ckpt][layer]
                if "quantization" in stats:
                    sqnr = stats["quantization"].get("sqnr_db", 0)
                    if sqnr > 0:
                        sqnrs.append(sqnr)
                        steps.append(i)
        
        if sqnrs:
            ax3.plot(steps, sqnrs, marker='o', label=_short_layer_label(layer, max_len=18))
    
    ax3.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Safe Threshold')
    ax3.set_xlabel('Checkpoint')
    ax3.set_ylabel('SQNR (dB)')
    ax3.set_title('Quantization Quality Evolution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 整体健康度演化
    ax4 = axes[1, 1]
    
    overall_health = []
    for ckpt in checkpoints:
        results = checkpoint_results[ckpt]
        
        # 计算健康度分数
        scores = []
        for stats in results.values():
            score = 100
            
            if "spectral" in stats:
                cond = stats["spectral"].get("condition_number", 0)
                if cond > 1e5:
                    score -= 40
                elif cond > 1e4:
                    score -= 20
            
            if "quantization" in stats:
                outlier = stats["quantization"].get("extreme_outlier_percentage", 0)
                if outlier > 10:
                    score -= 30
                elif outlier > 1:
                    score -= 15
            
            scores.append(max(0, score))
        
        overall_health.append(np.mean(scores) if scores else 0)
    
    ax4.plot(range(len(checkpoints)), overall_health, marker='o', 
            linewidth=2, markersize=8, color='#1f77b4')
    ax4.fill_between(range(len(checkpoints)), overall_health, alpha=0.3, color='#1f77b4')
    
    ax4.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good')
    ax4.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax4.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Critical')
    
    ax4.set_xlabel('Checkpoint')
    ax4.set_ylabel('Health Score')
    ax4.set_title('Overall Model Health Evolution')
    ax4.set_xticks(range(len(checkpoints)))
    ax4.set_xticklabels(checkpoints, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved evolution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

