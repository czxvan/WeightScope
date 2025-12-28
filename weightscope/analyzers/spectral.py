import torch
import numpy as np

def analyze_spectral(tensor: torch.Tensor, top_k=10):
    """
    Perform spectral analysis on a weight tensor using SVD.
    
    Returns:
        dict: Contains singular values, condition number, stable rank, etc.
    """
    # Convert to float32 on CPU for numerical stability
    t = tensor.detach().float().cpu()
    
    # Reshape to 2D matrix if needed
    original_shape = t.shape
    if len(t.shape) > 2:
        # Flatten all dims except last into first dim
        t = t.reshape(-1, t.shape[-1])
    elif len(t.shape) == 1:
        # For 1D tensors (bias), treat as column vector
        t = t.unsqueeze(1)
    
    # Perform SVD
    try:
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        singular_values = S.numpy()
        
        # Calculate metrics
        max_sv = singular_values[0]
        min_sv = singular_values[-1] if len(singular_values) > 0 else 0
        
        # Condition number (max/min singular value)
        condition_number = max_sv / min_sv if min_sv > 1e-10 else float('inf')
        
        # Stable rank: ||A||_F^2 / ||A||_2^2
        frobenius_norm_sq = (singular_values ** 2).sum()
        spectral_norm_sq = max_sv ** 2
        stable_rank = frobenius_norm_sq / spectral_norm_sq if spectral_norm_sq > 0 else 0
        
        # Effective rank (number of singular values > threshold)
        threshold = 0.01 * max_sv
        effective_rank = (singular_values > threshold).sum()
        
        # Top-k singular values
        top_singular_values = singular_values[:min(top_k, len(singular_values))].tolist()
        
        return {
            "condition_number": float(condition_number),
            "stable_rank": float(stable_rank),
            "effective_rank": int(effective_rank),
            "total_rank": len(singular_values),
            "max_singular_value": float(max_sv),
            "min_singular_value": float(min_sv),
            "top_singular_values": top_singular_values,
            "singular_value_ratio": float(singular_values[-1] / singular_values[0]) if len(singular_values) > 0 else 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "condition_number": None,
            "stable_rank": None,
            "effective_rank": None
        }
