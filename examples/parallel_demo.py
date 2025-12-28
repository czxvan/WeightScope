"""
Example: Using parallel processing for faster analysis
"""

from weightscope import Scope
import time

def main():
    model_name = "openai-community/gpt2"
    
    print("=" * 70)
    print("Parallel Processing Comparison")
    print("=" * 70)
    
    # Sequential analysis
    print("\n1. Running sequential analysis...")
    scope = Scope(model_name)
    
    start = time.time()
    results_seq = scope.scan(methods=["spectral", "quantization"], parallel=False)
    seq_time = time.time() - start
    
    print(f"   Sequential analysis took: {seq_time:.2f} seconds")
    
    # Parallel analysis
    print("\n2. Running parallel analysis...")
    scope2 = Scope(model_name)
    
    start = time.time()
    results_par = scope2.scan(methods=["spectral", "quantization"], parallel=True)
    par_time = time.time() - start
    
    print(f"   Parallel analysis took: {par_time:.2f} seconds")
    
    # Speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n   Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print(f"Parallel processing is {speedup:.1f}x faster!")
    print("=" * 70)

if __name__ == "__main__":
    main()
