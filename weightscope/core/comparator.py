"""
Comparison utilities for analyzing differences between two models.
"""

import torch
import numpy as np
from typing import Dict, Any


def compare_weights(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Dict[str, Any]:
    """
    Compare two weight tensors and compute difference metrics.
    
    Args:
        tensor1: First weight tensor
        tensor2: Second weight tensor
        
    Returns:
        dict: Comparison metrics including L2 distance, cosine similarity, etc.
    """
    if tensor1.shape != tensor2.shape:
        return {
            "error": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}",
            "comparable": False
        }
    
    t1 = tensor1.detach().float().cpu()
    t2 = tensor2.detach().float().cpu()
    
    # Compute difference
    diff = t1 - t2
    
    # L2 distance (Euclidean distance)
    l2_distance = torch.norm(diff, p=2).item()
    
    # Relative L2 distance
    l2_norm_1 = torch.norm(t1, p=2).item()
    relative_l2 = l2_distance / (l2_norm_1 + 1e-10)
    
    # Cosine similarity
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        t1_flat.unsqueeze(0), 
        t2_flat.unsqueeze(0)
    ).item()
    
    # Mean absolute difference
    mae = diff.abs().mean().item()
    
    # Max absolute difference
    max_abs_diff = diff.abs().max().item()
    
    # Percentage of weights that changed significantly (> 1% relative change)
    relative_diff = (diff.abs() / (t1.abs() + 1e-10))
    significant_changes = (relative_diff > 0.01).sum().item()
    significant_change_pct = significant_changes / t1.numel() * 100
    
    # Statistical divergence
    diff_stats = {
        "mean": diff.mean().item(),
        "std": diff.std().item(),
        "min": diff.min().item(),
        "max": diff.max().item()
    }
    
    return {
        "comparable": True,
        "l2_distance": l2_distance,
        "relative_l2_distance": relative_l2,
        "cosine_similarity": cos_sim,
        "mean_absolute_difference": mae,
        "max_absolute_difference": max_abs_diff,
        "significant_change_percentage": significant_change_pct,
        "difference_stats": diff_stats,
        "shape": list(tensor1.shape)
    }


def compare_models(results1: Dict, results2: Dict) -> Dict[str, Any]:
    """
    Compare analysis results from two models.
    
    Args:
        results1: Results from first model
        results2: Results from second model
        
    Returns:
        dict: Comparison summary
    """
    common_layers = set(results1.keys()) & set(results2.keys())
    only_in_1 = set(results1.keys()) - set(results2.keys())
    only_in_2 = set(results2.keys()) - set(results1.keys())
    
    layer_comparisons = {}
    
    for layer in common_layers:
        stats1 = results1[layer].get("basic_stats", {})
        stats2 = results2[layer].get("basic_stats", {})
        
        if stats1 and stats2:
            # Compare basic statistics
            layer_comparisons[layer] = {
                "mean_diff": stats2["mean"] - stats1["mean"],
                "std_diff": stats2["std"] - stats1["std"],
                "l2_norm_ratio": stats2["l2_norm"] / (stats1["l2_norm"] + 1e-10),
                "max_abs_change": max(abs(stats2["max"]), abs(stats2["min"])) - 
                                 max(abs(stats1["max"]), abs(stats1["min"]))
            }
            
            # Compare spectral properties if available
            if "spectral" in results1[layer] and "spectral" in results2[layer]:
                spec1 = results1[layer]["spectral"]
                spec2 = results2[layer]["spectral"]
                
                layer_comparisons[layer]["spectral_diff"] = {
                    "condition_number_ratio": spec2.get("condition_number", 0) / 
                                             (spec1.get("condition_number", 1e-10)),
                    "stable_rank_diff": spec2.get("stable_rank", 0) - spec1.get("stable_rank", 0)
                }
    
    return {
        "common_layers": len(common_layers),
        "only_in_model1": list(only_in_1),
        "only_in_model2": list(only_in_2),
        "layer_comparisons": layer_comparisons
    }


def find_most_changed_layers(comparison: Dict, top_k: int = 10) -> list:
    """
    Find the layers that changed the most between two models.
    
    Args:
        comparison: Comparison results from compare_models
        top_k: Number of top changed layers to return
        
    Returns:
        list: Top-k most changed layers with their metrics
    """
    layer_comparisons = comparison.get("layer_comparisons", {})
    
    # Calculate a composite change score for each layer
    scored_layers = []
    for layer_name, metrics in layer_comparisons.items():
        # Combine multiple metrics into a single score
        score = 0
        
        # Normalize and add different metrics
        if "mean_diff" in metrics:
            score += abs(metrics["mean_diff"]) * 10
        if "std_diff" in metrics:
            score += abs(metrics["std_diff"]) * 10
        if "l2_norm_ratio" in metrics:
            score += abs(metrics["l2_norm_ratio"] - 1.0)
        
        scored_layers.append((layer_name, score, metrics))
    
    # Sort by score descending
    scored_layers.sort(key=lambda x: x[1], reverse=True)
    
    return scored_layers[:top_k]
