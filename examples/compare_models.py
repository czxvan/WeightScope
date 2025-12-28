"""
Example: Comparing two model checkpoints or fine-tuned versions
"""

from weightscope import Scope
import json

def main():
    # For demonstration, we'll compare GPT-2 with itself
    # In practice, you would compare different checkpoints or model versions
    model1 = "openai-community/gpt2"
    model2 = "openai-community/gpt2"  # Replace with actual second model
    
    print("=" * 70)
    print("Model Comparison Example")
    print("=" * 70)
    
    print(f"\nModel 1: {model1}")
    print(f"Model 2: {model2}")
    
    # Load first model
    print("\n1. Loading and analyzing first model...")
    scope1 = Scope(model1)
    
    # Compare with second model
    print("\n2. Comparing with second model...")
    comparison = scope1.compare_with(model2, methods=["basic_stats", "spectral"])
    
    # Display results
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    print(f"\nCommon layers: {comparison['common_layers']}")
    print(f"Layers only in Model 1: {len(comparison['only_in_model1'])}")
    print(f"Layers only in Model 2: {len(comparison['only_in_model2'])}")
    
    # Find most changed layers
    from weightscope.core.comparator import find_most_changed_layers
    most_changed = find_most_changed_layers(comparison, top_k=5)
    
    print("\nTop 5 Most Changed Layers:")
    for i, (layer_name, score, metrics) in enumerate(most_changed, 1):
        print(f"\n{i}. {layer_name}")
        print(f"   Change Score: {score:.4f}")
        print(f"   Mean Difference: {metrics.get('mean_diff', 0):.6f}")
        print(f"   Std Difference: {metrics.get('std_diff', 0):.6f}")
        print(f"   L2 Norm Ratio: {metrics.get('l2_norm_ratio', 1.0):.4f}")
    
    # Save comparison results
    output_file = "model_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nFull comparison saved to {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
