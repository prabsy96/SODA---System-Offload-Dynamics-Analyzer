"""
Print utilities for SODA
"""

from typing import Any, List
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel
from rich.padding import Padding

def bool_to_match(match: bool) -> str:
    """Convert boolean match to colored tick mark."""
    return "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"

def iter_start(title: str, style: str = "cyan") -> None:
    """Print a title using rich Rule with dot for iteration start.
    
    Args:
        title: Title text to display
        style: Style/color for the rule (default: "cyan")
    """
    console = Console()
    console.print(Rule(f"\n{title}", style=style, align="left", characters="·"))

def iter_end(style: str = "cyan") -> None:
    """Print a separator using rich Rule with dot for iteration end.
    
    Args:
        style: Style/color for the rule (default: "cyan", matches iter_start)
    """
    console = Console()
    console.print(Rule(characters="·", style=style))

def subsection(title: str, level: int = 1, style: str = "cyan") -> None:
    """Print a subsection using Panel with indentation.
    
    Args:
        title: Subsection title text
        level: Indentation level (0 = no indent, 1 = 2 spaces, 2 = 4 spaces, etc.)
        style: Style/color for the panel border (default: "cyan")
    """
    console = Console()
    indent = level * 2  # 2 spaces per level
    panel = Panel(title, border_style=style, style="bold")
    console.print(Padding(panel, (0, 0, 0, indent)))

def comp_table(title: str, headers: List[str], data: List[List[Any]]):
    """Print a rich table with automatic boolean formatting and justification.
    
    Args:
        title: Table title
        headers: List of column header names
        data: List of rows, where each row is a list of values
    """
    console = Console()
    table = Table(title=title, title_style="bold", show_header=True, show_lines=True)
    
    # Detect boolean columns and auto-justify them
    if data:
        is_bool_column = [any(isinstance(row[i], bool) for row in data if i < len(row)) for i in range(len(headers))]
    else:
        is_bool_column = [False] * len(headers)
    
    # Add columns with automatic justification for boolean columns
    for i, header in enumerate(headers):
        justify_val = "center" if is_bool_column[i] else None
        table.add_column(header, justify=justify_val)
    
    # Add rows, formatting booleans automatically
    for row in data:
        formatted_row = []
        for value in row:
            if isinstance(value, bool):
                formatted_row.append(bool_to_match(value))
            else:
                formatted_row.append(str(value))
        table.add_row(*formatted_row)
    
    console.print(table)

