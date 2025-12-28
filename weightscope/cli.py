import argparse
from weightscope import Scope
from weightscope.core.comparator import find_most_changed_layers
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

def main():
    parser = argparse.ArgumentParser(description="WeightScope: LLM Weight Analysis Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a model for defects")
    scan_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    scan_parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")
    scan_parser.add_argument("--methods", type=str, nargs="+", 
                           default=["basic_stats"],
                           choices=["basic_stats", "spectral", "quantization", "sparsity", "all"],
                           help="Analysis methods to run")
    scan_parser.add_argument("--output", type=str, help="Output JSON file path")
    scan_parser.add_argument("--top-issues", type=int, default=10, 
                           help="Show top N problematic layers")
    scan_parser.add_argument("--parallel", action="store_true",
                           help="Use parallel processing for faster analysis")
    scan_parser.add_argument("--workers", type=int, default=None,
                           help="Number of parallel workers (default: CPU count)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model1", type=str, required=True, help="First model name or path")
    compare_parser.add_argument("--model2", type=str, required=True, help="Second model name or path")
    compare_parser.add_argument("--device", type=str, default="cpu", help="Device to load models on")
    compare_parser.add_argument("--methods", type=str, nargs="+",
                              default=["basic_stats", "spectral"],
                              choices=["basic_stats", "spectral", "quantization", "sparsity", "all"],
                              help="Analysis methods for comparison")
    compare_parser.add_argument("--output", type=str, help="Output JSON file path")
    compare_parser.add_argument("--top-changes", type=int, default=10,
                              help="Show top N most changed layers")

    args = parser.parse_args()

    console = Console()

    if args.command == "scan":
        console.print(f"[bold green]Starting scan for model: {args.model}[/bold green]")
        
        try:
            scope = Scope(args.model, device=args.device)
            results = scope.scan(methods=args.methods, parallel=args.parallel, num_workers=args.workers)
            
            # Save to JSON if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]Results saved to {args.output}[/green]")
            
            # Display results based on analysis type
            all_methods = args.methods if "all" not in args.methods else ["basic_stats", "spectral", "quantization", "sparsity"]
            
            if "basic_stats" in all_methods:
                display_basic_stats(console, results)
            
            if "spectral" in all_methods:
                display_spectral_analysis(console, results, args.top_issues)
            
            if "quantization" in all_methods:
                display_quantization_analysis(console, results, args.top_issues)
            
            if "sparsity" in all_methods:
                display_sparsity_analysis(console, results, args.top_issues)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            import traceback
            console.print(traceback.format_exc())
    
    elif args.command == "compare":
        console.print(f"[bold cyan]Comparing models:[/bold cyan]")
        console.print(f"  Model 1: {args.model1}")
        console.print(f"  Model 2: {args.model2}")
        
        try:
            scope1 = Scope(args.model1, device=args.device)
            comparison = scope1.compare_with(args.model2, methods=args.methods)
            
            # Save to JSON if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(comparison, f, indent=2)
                console.print(f"[green]Comparison results saved to {args.output}[/green]")
            
            # Display comparison results
            display_comparison(console, comparison, args.top_changes)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            import traceback
            console.print(traceback.format_exc())


def display_basic_stats(console, results):
    table = Table(title="Basic Statistics")
    table.add_column("Layer", style="cyan", max_width=40)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("L2 Norm", justify="right")

    for layer_name, stats in results.items():
        if "basic_stats" in stats:
            s = stats["basic_stats"]
            table.add_row(
                layer_name, 
                f"{s['mean']:.4f}", 
                f"{s['std']:.4f}", 
                f"{s['min']:.4f}", 
                f"{s['max']:.4f}",
                f"{s['l2_norm']:.2f}"
            )
    
    console.print(table)


def display_spectral_analysis(console, results, top_n=10):
    # Find layers with highest condition numbers
    layers_with_issues = []
    for layer_name, stats in results.items():
        if "spectral" in stats and stats["spectral"].get("condition_number"):
            cond = stats["spectral"]["condition_number"]
            if cond != float('inf'):
                layers_with_issues.append((layer_name, cond, stats["spectral"]))
    
    layers_with_issues.sort(key=lambda x: x[1], reverse=True)
    
    table = Table(title=f"Top {top_n} Layers by Condition Number")
    table.add_column("Layer", style="cyan", max_width=40)
    table.add_column("Condition #", justify="right")
    table.add_column("Stable Rank", justify="right")
    table.add_column("Effective Rank", justify="right")
    table.add_column("Total Rank", justify="right")
    
    for layer_name, cond, spectral in layers_with_issues[:top_n]:
        style = "red" if cond > 1000 else "yellow" if cond > 100 else "white"
        table.add_row(
            layer_name,
            f"[{style}]{cond:.2f}[/{style}]",
            f"{spectral['stable_rank']:.2f}",
            f"{spectral['effective_rank']}",
            f"{spectral['total_rank']}"
        )
    
    console.print(table)


def display_quantization_analysis(console, results, top_n=10):
    # Find layers most sensitive to quantization
    layers_with_issues = []
    for layer_name, stats in results.items():
        if "quantization" in stats:
            quant = stats["quantization"]
            layers_with_issues.append((
                layer_name, 
                quant["extreme_outlier_percentage"],
                quant
            ))
    
    layers_with_issues.sort(key=lambda x: x[1], reverse=True)
    
    table = Table(title=f"Top {top_n} Quantization-Sensitive Layers")
    table.add_column("Layer", style="cyan", max_width=40)
    table.add_column("Outliers %", justify="right")
    table.add_column("SQNR (dB)", justify="right")
    table.add_column("Cos Sim", justify="right")
    table.add_column("Dynamic Range", justify="right")
    
    for layer_name, outlier_pct, quant in layers_with_issues[:top_n]:
        style = "red" if outlier_pct > 1 else "yellow" if outlier_pct > 0.1 else "white"
        table.add_row(
            layer_name,
            f"[{style}]{outlier_pct:.3f}%[/{style}]",
            f"{quant['sqnr_db']:.2f}",
            f"{quant['cosine_similarity']:.4f}",
            f"{quant['dynamic_range']:.2f}"
        )
    
    console.print(table)


def display_sparsity_analysis(console, results, top_n=10):
    # Find sparsest layers
    layers_with_sparsity = []
    for layer_name, stats in results.items():
        if "sparsity" in stats:
            sparse = stats["sparsity"]
            sparsity_pct = sparse["sparsity_levels"].get("threshold_0.0", 0)
            layers_with_sparsity.append((layer_name, sparsity_pct, sparse))
    
    layers_with_sparsity.sort(key=lambda x: x[1], reverse=True)
    
    table = Table(title=f"Top {top_n} Sparse Layers")
    table.add_column("Layer", style="cyan", max_width=40)
    table.add_column("Exact Zero %", justify="right")
    table.add_column("Near-Zero %", justify="right")
    table.add_column("Dead Rows", justify="right")
    table.add_column("Dead Cols", justify="right")
    
    for layer_name, zero_pct, sparse in layers_with_sparsity[:top_n]:
        near_zero = sparse["sparsity_levels"].get("threshold_1e-06", 0)
        structured = sparse.get("structured_sparsity", {})
        
        table.add_row(
            layer_name,
            f"{zero_pct:.2f}%",
            f"{near_zero:.2f}%",
            str(structured.get("dead_rows", "N/A")),
            str(structured.get("dead_columns", "N/A"))
        )
    
    console.print(table)


def display_comparison(console, comparison, top_n=10):
    """Display comparison results between two models."""
    
    # Summary panel
    summary_text = f"""
[bold]Common Layers:[/bold] {comparison['common_layers']}
[bold]Only in Model 1:[/bold] {len(comparison['only_in_model1'])}
[bold]Only in Model 2:[/bold] {len(comparison['only_in_model2'])}
    """
    console.print(Panel(summary_text, title="Comparison Summary", border_style="cyan"))
    
    # Find most changed layers
    from weightscope.core.comparator import find_most_changed_layers
    most_changed = find_most_changed_layers(comparison, top_k=top_n)
    
    if most_changed:
        table = Table(title=f"Top {top_n} Most Changed Layers")
        table.add_column("Layer", style="cyan", max_width=40)
        table.add_column("Change Score", justify="right")
        table.add_column("Mean Δ", justify="right")
        table.add_column("Std Δ", justify="right")
        table.add_column("L2 Norm Ratio", justify="right")
        
        for layer_name, score, metrics in most_changed:
            style = "red" if score > 1.0 else "yellow" if score > 0.1 else "white"
            
            table.add_row(
                layer_name,
                f"[{style}]{score:.4f}[/{style}]",
                f"{metrics.get('mean_diff', 0):.4f}",
                f"{metrics.get('std_diff', 0):.4f}",
                f"{metrics.get('l2_norm_ratio', 1.0):.4f}"
            )
        
        console.print(table)
    
    # Show layers only in one model
    if comparison['only_in_model1']:
        console.print(f"\n[yellow]Layers only in Model 1:[/yellow] {', '.join(comparison['only_in_model1'][:5])}")
        if len(comparison['only_in_model1']) > 5:
            console.print(f"  ... and {len(comparison['only_in_model1']) - 5} more")
    
    if comparison['only_in_model2']:
        console.print(f"\n[yellow]Layers only in Model 2:[/yellow] {', '.join(comparison['only_in_model2'][:5])}")
        if len(comparison['only_in_model2']) > 5:
            console.print(f"  ... and {len(comparison['only_in_model2']) - 5} more")


if __name__ == "__main__":
    main()
