import torch
import numpy as np

def analyze_basic_stats(tensor: torch.Tensor):
    """
    Calculate basic statistics for a weight tensor.
    """
    # Ensure tensor is on CPU and float32 for statistics
    t = tensor.detach().float().cpu()
    
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "l2_norm": torch.norm(t, p=2).item(),
        "zeros_percentage": (t == 0).sum().item() / t.numel() * 100
    }
