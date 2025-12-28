import torch
import numpy as np

def analyze_sparsity(tensor: torch.Tensor, thresholds=[0.0, 1e-6, 1e-4, 1e-2]):
    """
    Analyze sparsity patterns in weight tensor.
    
    Args:
        tensor: Weight tensor
        thresholds: List of thresholds to consider as "zero"
    
    Returns:
        dict: Sparsity metrics at different thresholds
    """
    t = tensor.detach().float().cpu()
    total_elements = t.numel()
    
    sparsity_at_thresholds = {}
    for thresh in thresholds:
        near_zero = (t.abs() <= thresh).sum().item()
        sparsity_percentage = near_zero / total_elements * 100
        sparsity_at_thresholds[f"threshold_{thresh}"] = sparsity_percentage
    
    # Structured sparsity analysis (for 2D tensors)
    structured_sparsity = {}
    if len(t.shape) == 2:
        # Row-wise sparsity
        row_zeros = (t.abs() <= 1e-6).all(dim=1).sum().item()
        structured_sparsity["dead_rows"] = row_zeros
        structured_sparsity["dead_rows_percentage"] = row_zeros / t.shape[0] * 100
        
        # Column-wise sparsity
        col_zeros = (t.abs() <= 1e-6).all(dim=0).sum().item()
        structured_sparsity["dead_columns"] = col_zeros
        structured_sparsity["dead_columns_percentage"] = col_zeros / t.shape[1] * 100
    
    # Distribution of small weights
    abs_weights = t.abs().numpy()
    percentiles = {
        "p10": float(np.percentile(abs_weights, 10)),
        "p25": float(np.percentile(abs_weights, 25)),
        "p50": float(np.percentile(abs_weights, 50)),
        "p75": float(np.percentile(abs_weights, 75)),
        "p90": float(np.percentile(abs_weights, 90)),
    }
    
    return {
        "sparsity_levels": sparsity_at_thresholds,
        "structured_sparsity": structured_sparsity,
        "magnitude_percentiles": percentiles,
        "mean_abs_weight": float(np.mean(abs_weights)),
        "median_abs_weight": float(np.median(abs_weights))
    }
