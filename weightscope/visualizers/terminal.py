from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

def print_summary_table(results: dict, console=None):
    """
    Print a formatted summary table to terminal.
    """
    if console is None:
        console = Console()
    
    table = Table(title="Weight Analysis Summary")
    table.add_column("Layer", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Max Abs", justify="right")
    
    for layer_name, stats in results.items():
        if "basic_stats" in stats:
            s = stats["basic_stats"]
            max_abs = max(abs(s["min"]), abs(s["max"]))
            table.add_row(
                layer_name[:50],  # Truncate long names
                f"{s['mean']:.4f}",
                f"{s['std']:.4f}",
                f"{max_abs:.4f}"
            )
    
    console.print(table)
