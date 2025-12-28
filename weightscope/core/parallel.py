"""
Parallel processing utilities for faster analysis.
"""

import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple, Any
import multiprocessing as mp


def analyze_layer_parallel(layer_data: Tuple[str, torch.nn.Module, List[str]]) -> Tuple[str, Dict]:
    """
    Analyze a single layer with multiple methods.
    This function is designed to be picklable for multiprocessing.
    
    Args:
        layer_data: Tuple of (layer_name, module, methods)
        
    Returns:
        Tuple of (layer_name, results)
    """
    from weightscope.analyzers.basic import analyze_basic_stats
    from weightscope.analyzers.spectral import analyze_spectral
    from weightscope.analyzers.quantization import analyze_quantization_sensitivity
    from weightscope.analyzers.sparsity import analyze_sparsity
    
    layer_name, module, methods = layer_data
    results = {}
    
    if hasattr(module, 'weight') and module.weight is not None:
        weight = module.weight
        
        if "basic_stats" in methods:
            results["basic_stats"] = analyze_basic_stats(weight)
        
        if "spectral" in methods:
            results["spectral"] = analyze_spectral(weight)
        
        if "quantization" in methods:
            results["quantization"] = analyze_quantization_sensitivity(weight)
        
        if "sparsity" in methods:
            results["sparsity"] = analyze_sparsity(weight)
    
    return layer_name, results


class ParallelAnalyzer:
    """
    Parallel analyzer using multiprocessing or threading.
    """
    
    def __init__(self, num_workers: int = None, use_threads: bool = True):
        """
        Args:
            num_workers: Number of worker processes/threads. None = CPU count
            use_threads: If True, use ThreadPoolExecutor instead of ProcessPoolExecutor
                        (Default: True for better compatibility)
        """
        self.num_workers = num_workers or min(mp.cpu_count(), 8)  # Cap at 8 for safety
        self.use_threads = use_threads
    
    def analyze_layers(self, layers: List[Tuple], methods: List[str]) -> Dict:
        """
        Analyze multiple layers in parallel.
        
        Args:
            layers: List of (layer_name, module) tuples
            methods: List of analysis methods to apply
            
        Returns:
            dict: Results for all layers
        """
        # Prepare layer data for parallel processing
        # Note: We need to extract weight tensors to avoid pickling issues with modules
        layer_data_list = []
        for layer_name, module in layers:
            if hasattr(module, 'weight') and module.weight is not None:
                # Clone weight to CPU to avoid CUDA serialization issues
                weight_cpu = module.weight.detach().cpu()
                layer_data_list.append((layer_name, weight_cpu, methods))
        
        results = {}
        
        # Use ThreadPoolExecutor for better compatibility
        # (ProcessPoolExecutor can have issues with CUDA tensors and module pickling)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_layer = {
                executor.submit(analyze_layer_worker, data): data[0] 
                for data in layer_data_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_layer):
                layer_name = future_to_layer[future]
                try:
                    name, layer_results = future.result()
                    if layer_results:
                        results[name] = layer_results
                except Exception as e:
                    print(f"Error analyzing layer {layer_name}: {e}")
                    results[layer_name] = {"error": str(e)}
        
        return results


def analyze_layer_worker(layer_data: Tuple[str, torch.Tensor, List[str]]) -> Tuple[str, Dict]:
    """
    Worker function for parallel analysis.
    Operates on tensors directly to avoid module serialization issues.
    
    Args:
        layer_data: Tuple of (layer_name, weight_tensor, methods)
        
    Returns:
        Tuple of (layer_name, results)
    """
    from weightscope.analyzers.basic import analyze_basic_stats
    from weightscope.analyzers.spectral import analyze_spectral
    from weightscope.analyzers.quantization import analyze_quantization_sensitivity
    from weightscope.analyzers.sparsity import analyze_sparsity
    
    layer_name, weight, methods = layer_data
    results = {}
    
    try:
        if "basic_stats" in methods:
            results["basic_stats"] = analyze_basic_stats(weight)
        
        if "spectral" in methods:
            results["spectral"] = analyze_spectral(weight)
        
        if "quantization" in methods:
            results["quantization"] = analyze_quantization_sensitivity(weight)
        
        if "sparsity" in methods:
            results["sparsity"] = analyze_sparsity(weight)
    except Exception as e:
        results["error"] = str(e)
    
    return layer_name, results


def batch_analyze(layer_batches: List[List[Tuple]], methods: List[str], 
                  num_workers: int = None) -> Dict:
    """
    Analyze layers in batches for better memory management.
    
    Args:
        layer_batches: List of batches, each batch is a list of (name, module) tuples
        methods: Analysis methods to apply
        num_workers: Number of parallel workers
        
    Returns:
        dict: Combined results from all batches
    """
    analyzer = ParallelAnalyzer(num_workers=num_workers)
    all_results = {}
    
    for i, batch in enumerate(layer_batches):
        print(f"Processing batch {i+1}/{len(layer_batches)}...")
        batch_results = analyzer.analyze_layers(batch, methods)
        all_results.update(batch_results)
    
    return all_results
