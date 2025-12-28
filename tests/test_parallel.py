"""
Test parallel vs sequential analysis performance
"""
import time
from weightscope import Scope

def test_performance():
    model = "openai-community/gpt2"
    methods = ["basic_stats", "spectral"]
    
    print("=" * 70)
    print("Performance Test: Sequential vs Parallel Analysis")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Methods: {methods}")
    
    # Sequential
    print("\n1. Sequential Analysis...")
    scope_seq = Scope(model)
    start = time.time()
    results_seq = scope_seq.scan(methods=methods, parallel=False)
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.2f}s")
    print(f"   Layers analyzed: {len(results_seq)}")
    
    # Parallel
    print("\n2. Parallel Analysis...")
    scope_par = Scope(model)
    start = time.time()
    results_par = scope_par.scan(methods=methods, parallel=True, num_workers=4)
    par_time = time.time() - start
    print(f"   Time: {par_time:.2f}s")
    print(f"   Layers analyzed: {len(results_par)}")
    
    # Results
    speedup = seq_time / par_time if par_time > 0 else 1.0
    print("\n" + "=" * 70)
    print(f"Speedup: {speedup:.2f}x")
    print("=" * 70)
    
    # Verify results are the same
    print("\n3. Verifying results match...")
    matches = 0
    for layer in results_seq:
        if layer in results_par:
            if "basic_stats" in results_seq[layer] and "basic_stats" in results_par[layer]:
                seq_mean = results_seq[layer]["basic_stats"]["mean"]
                par_mean = results_par[layer]["basic_stats"]["mean"]
                if abs(seq_mean - par_mean) < 1e-6:
                    matches += 1
    
    print(f"   Matching layers: {matches}/{len(results_seq)}")
    print(f"   Results are {'identical' if matches == len(results_seq) else 'different'}!")
    
if __name__ == "__main__":
    test_performance()
