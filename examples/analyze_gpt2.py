"""
Example: Analyzing GPT-2 weights with WeightScope
"""

from weightscope import Scope
import json

def main():
    print("=" * 60)
    print("WeightScope Example: Analyzing GPT-2")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    scope = Scope("openai-community/gpt2", device="cpu")
    
    # Run comprehensive analysis
    print("\n2. Running comprehensive analysis...")
    results = scope.scan(methods=["all"])
    
    # Find problematic layers
    print("\n3. Analyzing results...")
    
    # Find layers with high condition numbers
    high_condition = []
    for layer_name, stats in results.items():
        if "spectral" in stats:
            cond = stats["spectral"].get("condition_number", 0)
            if cond > 1000:
                high_condition.append((layer_name, cond))
    
    high_condition.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   Found {len(high_condition)} layers with high condition numbers (>1000):")
    for layer, cond in high_condition[:5]:
        print(f"   - {layer}: {cond:.2f}")
    
    # Find layers with many outliers
    outlier_layers = []
    for layer_name, stats in results.items():
        if "quantization" in stats:
            outlier_pct = stats["quantization"].get("extreme_outlier_percentage", 0)
            if outlier_pct > 0.1:
                outlier_layers.append((layer_name, outlier_pct))
    
    outlier_layers.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   Found {len(outlier_layers)} layers with significant outliers (>0.1%):")
    for layer, pct in outlier_layers[:5]:
        print(f"   - {layer}: {pct:.3f}%")
    
    # Save detailed results
    output_file = "gpt2_analysis_detailed.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n4. Detailed results saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
