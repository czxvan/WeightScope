from .core.loader import ModelLoader
from .core.walker import ModelWalker
from .core.comparator import compare_models, compare_weights, find_most_changed_layers
from .core.parallel import ParallelAnalyzer
from .analyzers.basic import analyze_basic_stats
from .analyzers.spectral import analyze_spectral
from .analyzers.quantization import analyze_quantization_sensitivity, detect_outliers
from .analyzers.sparsity import analyze_sparsity

class Scope:
    def __init__(self, model_name_or_path, device="cpu"):
        self.loader = ModelLoader(model_name_or_path, device)
        self.walker = ModelWalker(self.loader.model)
        self.model_name = model_name_or_path

    @property
    def model(self):
        return self.loader.model
        
    def scan(self, methods=None, parallel=False, num_workers=None):
        """
        Run analysis on the model.
        
        Args:
            methods: List of analysis methods. Options:
                - "basic_stats": Mean, std, min, max, norms
                - "spectral": SVD, condition number, stable rank
                - "quantization": Quantization sensitivity, outliers
                - "sparsity": Sparsity levels, dead neurons
                - "all": Run all analyses
            parallel: Use parallel processing for faster analysis
            num_workers: Number of parallel workers (None = CPU count)
        """
        if methods is None:
            methods = ["basic_stats"]
        elif "all" in methods:
            methods = ["basic_stats", "spectral", "quantization", "sparsity"]
            
        print(f"Scanning model: {self.loader.model_name} with methods: {methods}")
        if parallel:
            print(f"Using parallel processing with {num_workers or 'auto'} workers...")
        
        if parallel:
            # Use parallel analysis
            analyzer = ParallelAnalyzer(num_workers=num_workers)
            layers = list(self.walker.walk())
            results = analyzer.analyze_layers(layers, methods)
            layer_count = len(layers)
        else:
            # Sequential analysis
            results = {}
            layer_count = 0
            for name, module in self.walker.walk():
                layer_count += 1
                layer_results = {}
                
                # Assuming module.weight exists for Linear/Conv2d/Embedding
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight
                    
                    if "basic_stats" in methods:
                        layer_results["basic_stats"] = analyze_basic_stats(weight)
                    
                    if "spectral" in methods:
                        layer_results["spectral"] = analyze_spectral(weight)
                    
                    if "quantization" in methods:
                        layer_results["quantization"] = analyze_quantization_sensitivity(weight)
                    
                    if "sparsity" in methods:
                        layer_results["sparsity"] = analyze_sparsity(weight)
                
                if layer_results:
                    results[name] = layer_results
        
        print(f"Found {layer_count} layers to analyze, {len(results)} with valid weights.")
        return results
    
    def compare_with(self, other_model_or_results, methods=None):
        """
        Compare this model with another model or results.
        
        Args:
            other_model_or_results: Either a model path/name or pre-computed results dict
            methods: Analysis methods to use for comparison
            
        Returns:
            dict: Comparison results
        """
        if methods is None:
            methods = ["basic_stats", "spectral"]
        
        # Get results for this model
        results1 = self.scan(methods=methods)
        
        # Get results for the other model
        if isinstance(other_model_or_results, dict):
            results2 = other_model_or_results
        else:
            other_scope = Scope(other_model_or_results, device=self.loader.device)
            results2 = other_scope.scan(methods=methods)
        
        # Compare the results
        comparison = compare_models(results1, results2)
        
        return comparison

    def print_summary(self):
        print("Scan complete. (Summary implementation pending)")
