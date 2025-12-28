"""
Simple tests for WeightScope analyzers
"""

import torch
from weightscope.analyzers.basic import analyze_basic_stats
from weightscope.analyzers.spectral import analyze_spectral
from weightscope.analyzers.quantization import analyze_quantization_sensitivity
from weightscope.analyzers.sparsity import analyze_sparsity


def test_basic_stats():
    """Test basic statistics analyzer"""
    # Create a simple tensor
    tensor = torch.randn(100, 100)
    stats = analyze_basic_stats(tensor)
    
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "l2_norm" in stats
    
    print("✓ Basic stats test passed")


def test_spectral():
    """Test spectral analyzer"""
    tensor = torch.randn(50, 100)
    spectral = analyze_spectral(tensor)
    
    assert "condition_number" in spectral
    assert "stable_rank" in spectral
    assert "effective_rank" in spectral
    assert "max_singular_value" in spectral
    
    print("✓ Spectral analysis test passed")


def test_quantization():
    """Test quantization analyzer"""
    tensor = torch.randn(100, 100)
    # Add some outliers
    tensor[0, 0] = 100.0
    tensor[1, 1] = -100.0
    
    quant = analyze_quantization_sensitivity(tensor)
    
    assert "outlier_percentage" in quant
    assert "quantization_mse" in quant
    assert "sqnr_db" in quant
    assert quant["outlier_percentage"] > 0
    
    print("✓ Quantization analysis test passed")


def test_sparsity():
    """Test sparsity analyzer"""
    tensor = torch.randn(100, 100)
    # Make it sparse
    tensor[tensor.abs() < 0.5] = 0
    
    sparse = analyze_sparsity(tensor)
    
    assert "sparsity_levels" in sparse
    assert "structured_sparsity" in sparse
    assert sparse["sparsity_levels"]["threshold_0.0"] > 0
    
    print("✓ Sparsity analysis test passed")


if __name__ == "__main__":
    print("Running WeightScope tests...\n")
    test_basic_stats()
    test_spectral()
    test_quantization()
    test_sparsity()
    print("\n✓ All tests passed!")
