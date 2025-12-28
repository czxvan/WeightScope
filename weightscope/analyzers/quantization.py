import torch
import numpy as np

def analyze_quantization_sensitivity(tensor: torch.Tensor, bits=8):
    """
    Analyze how sensitive this layer is to quantization.
    
    Args:
        tensor: Weight tensor
        bits: Target quantization bits (default: 8 for INT8)
    
    Returns:
        dict: Outlier statistics and quantization error estimates
    """
    t = tensor.detach().float().cpu()
    
    # Outlier detection using IQR method
    flat = t.flatten()
    
    # Use numpy for quantile calculation to avoid tensor size issues
    flat_np = flat.numpy()
    q1 = np.percentile(flat_np, 25)
    q3 = np.percentile(flat_np, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = ((flat < lower_bound) | (flat > upper_bound)).sum().item()
    outlier_percentage = outliers / flat.numel() * 100
    
    # Extreme outlier detection (3x IQR)
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr
    extreme_outliers = ((flat < extreme_lower) | (flat > extreme_upper)).sum().item()
    extreme_outlier_percentage = extreme_outliers / flat.numel() * 100
    
    # Simulate symmetric quantization
    abs_max = flat.abs().max().item()
    qmax = 2 ** (bits - 1) - 1  # e.g., 127 for INT8
    scale = abs_max / qmax if qmax > 0 else 0
    
    # Quantize and dequantize
    quantized = torch.round(flat / scale).clamp(-qmax - 1, qmax)
    dequantized = quantized * scale
    
    # Calculate error metrics
    mse = ((flat - dequantized) ** 2).mean().item()
    mae = (flat - dequantized).abs().mean().item()
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        flat.unsqueeze(0), 
        dequantized.unsqueeze(0)
    ).item()
    
    # Signal-to-Quantization-Noise Ratio (SQNR)
    signal_power = (flat ** 2).mean().item()
    sqnr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
    
    return {
        "outlier_percentage": outlier_percentage,
        "extreme_outlier_percentage": extreme_outlier_percentage,
        "outlier_count": outliers,
        "extreme_outlier_count": extreme_outliers,
        "quantization_scale": scale,
        "quantization_mse": mse,
        "quantization_mae": mae,
        "cosine_similarity": cos_sim,
        "sqnr_db": sqnr_db,
        "abs_max": abs_max,
        "dynamic_range": abs_max / (flat.abs().mean().item() + 1e-10)
    }


def detect_outliers(tensor: torch.Tensor, method="iqr", threshold=1.5):
    """
    Detect outlier weights using various methods.
    
    Args:
        tensor: Weight tensor
        method: "iqr" (Interquartile Range) or "zscore" (Z-score)
        threshold: Multiplier for IQR or Z-score threshold
    
    Returns:
        dict: Outlier locations and statistics
    """
    t = tensor.detach().float().cpu()
    flat = t.flatten()
    
    if method == "iqr":
        q1 = torch.quantile(flat, 0.25)
        q3 = torch.quantile(flat, 0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (flat < lower) | (flat > upper)
    
    elif method == "zscore":
        mean = flat.mean()
        std = flat.std()
        z_scores = ((flat - mean) / std).abs()
        outlier_mask = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_indices = outlier_mask.nonzero(as_tuple=True)[0]
    outlier_values = flat[outlier_indices].numpy().tolist()
    
    return {
        "outlier_count": len(outlier_indices),
        "outlier_percentage": len(outlier_indices) / len(flat) * 100,
        "max_outlier": max(outlier_values) if outlier_values else 0,
        "min_outlier": min(outlier_values) if outlier_values else 0,
        "method": method
    }
