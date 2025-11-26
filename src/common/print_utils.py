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
        title: Title text to display (will be colored with style)
        style: Style/color for the rule and title (default: "cyan")
    """
    console = Console()
    # Color the title text with the style
    colored_title = f"[{style}]{title}[/{style}]"
    console.print(Rule(f"\n{colored_title}", style=style, align="right", characters="·"))

def iter_end(style: str = "cyan") -> None:
    """Print a separator using rich Rule with dot for iteration end.

    Args:
        style: Style/color for the rule (default: "cyan", matches iter_start)
    """
    console = Console()
    console.print(Rule(characters="·", style=style))

def section_end(title: str = "", style: str = "cyan") -> None:
    """Print a strong separator using rich Rule with heavy line for section end.
    
    Args:
        title: Optional title to display as </Title> (default: empty for just a line)
        style: Style/color for the rule (default: "cyan")
    """
    console = Console()
    if title:
        colored_title = f"[{style}]</{title}>[/{style}]"
        console.print(Rule(colored_title, style=style, characters="━"))
    else:
        console.print(Rule(characters="━", style=style))
    console.print("\n")

def section_start(title: str, style: str = "cyan") -> None:
    """Print a section_start heading using Rule with double line separator.
    
    Args:
        title: section_start title text (will be displayed as <Title>)
        style: Style/color for the rule (default: "cyan")
    """
    console = Console()
    indent = 0
    colored_title = f"[{style}]<{title}>[/{style}]"
    rule = Rule(colored_title, style=style, characters="═")
    console.print(Padding(rule, (0, 0, 0, indent)))

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
            if value is None:
                formatted_row.append("-")
            elif isinstance(value, bool):
                formatted_row.append(bool_to_match(value))
            elif isinstance(value, (list, tuple)) and all(isinstance(x, bool) for x in value):
                # Format list of bools as space-separated checkmarks
                formatted_row.append(" ".join(bool_to_match(x) for x in value))
            else:
                formatted_row.append(str(value))
        table.add_row(*formatted_row)
    
    console.print(table)
