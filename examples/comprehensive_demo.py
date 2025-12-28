"""
Comprehensive example: Full workflow with WeightScope
Demonstrates all major features including parallel processing and comparison
"""

from weightscope import Scope
from weightscope.core.comparator import find_most_changed_layers
import json
import time

def analyze_model(model_name, parallel=True):
    """Analyze a model with all methods"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")
    
    scope = Scope(model_name)
    
    start = time.time()
    results = scope.scan(
        methods=["all"],
        parallel=parallel,
        num_workers=4
    )
    duration = time.time() - start
    
    print(f"\nAnalysis completed in {duration:.2f} seconds")
    print(f"Analyzed {len(results)} layers")
    
    # Find problematic layers
    print("\n--- Potential Issues ---")
    
    # High condition numbers
    high_cond = []
    for layer, stats in results.items():
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1000:
                high_cond.append((layer, cond))
    
    high_cond.sort(key=lambda x: x[1], reverse=True)
    print(f"\n1. Layers with high condition numbers (>1000): {len(high_cond)}")
    for layer, cond in high_cond[:3]:
        print(f"   - {layer}: {cond:.2f}")
    
    # Quantization-sensitive layers
    sensitive = []
    for layer, stats in results.items():
        if "quantization" in stats:
            outliers = stats["quantization"].get("extreme_outlier_percentage", 0)
            if outliers > 0.1:
                sensitive.append((layer, outliers))
    
    sensitive.sort(key=lambda x: x[1], reverse=True)
    print(f"\n2. Quantization-sensitive layers (>0.1% outliers): {len(sensitive)}")
    for layer, pct in sensitive[:3]:
        print(f"   - {layer}: {pct:.3f}%")
    
    return results


def compare_models_detailed(model1_name, model2_name):
    """Compare two models in detail"""
    print(f"\n{'='*70}")
    print(f"Comparing Models")
    print(f"{'='*70}")
    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")
    
    # Analyze both models
    scope1 = Scope(model1_name)
    comparison = scope1.compare_with(model2_name, methods=["basic_stats", "spectral"])
    
    print(f"\nCommon layers: {comparison['common_layers']}")
    
    # Find most changed
    most_changed = find_most_changed_layers(comparison, top_k=5)
    
    if most_changed:
        print("\nTop 5 Most Changed Layers:")
        for i, (layer, score, metrics) in enumerate(most_changed, 1):
            print(f"\n{i}. {layer}")
            print(f"   Change Score: {score:.4f}")
            if "mean_diff" in metrics:
                print(f"   Mean Δ: {metrics['mean_diff']:.6f}")
            if "l2_norm_ratio" in metrics:
                print(f"   L2 Norm Ratio: {metrics['l2_norm_ratio']:.4f}")
    
    return comparison


def main():
    """Main workflow demonstration"""
    print("=" * 70)
    print("WeightScope Comprehensive Demo")
    print("=" * 70)
    
    model = "openai-community/gpt2"
    
    # Step 1: Analyze a model with parallel processing
    print("\n[Step 1] Analyzing model with parallel processing...")
    results = analyze_model(model, parallel=True)
    
    # Step 2: Save results
    output_file = "comprehensive_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Step 2] Results saved to {output_file}")
    
    # Step 3: Model comparison (comparing with itself as demo)
    print("\n[Step 3] Demonstrating model comparison...")
    print("(In real usage, you would compare different checkpoints)")
    comparison = compare_models_detailed(model, model)
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)
    print("\nKey Capabilities Demonstrated:")
    print("✓ Comprehensive multi-method analysis")
    print("✓ Parallel processing for speed")
    print("✓ Issue detection (condition numbers, quantization sensitivity)")
    print("✓ Model comparison")
    print("✓ Results export to JSON")
    print("\nUse these insights to:")
    print("- Optimize quantization strategies")
    print("- Identify layers for pruning or LoRA")
    print("- Monitor training stability")
    print("- Compare model versions")


if __name__ == "__main__":
    main()
