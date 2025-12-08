"""
Utility functions for advanced demo scripts.

This module provides common functionality used across all advanced demo scripts,
including formatting, error handling, and progress reporting.
"""

import sys
import traceback
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(title: str, width: int = 70) -> None:
    """
    Print a formatted header for demo sections.

    Args:
        title: The title to display
        width: Width of the header line
    """
    print(f"\n{Colors.BOLD}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * width}{Colors.ENDC}\n")


def print_section(title: str, level: int = 1) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        level: Section level (1 for main sections, 2 for subsections)
    """
    if level == 1:
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.BOLD}--- {title} ---{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC}  {message}")


def print_info(message: str, indent: int = 0) -> None:
    """Print an info message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}{message}")


def print_metric(name: str, value: Any, unit: str = "", indent: int = 1) -> None:
    """
    Print a formatted metric.

    Args:
        name: Metric name
        value: Metric value
        unit: Optional unit string
        indent: Indentation level
    """
    prefix = "  " * indent
    if isinstance(value, float):
        if unit == "%":
            print(f"{prefix}{name}: {value:.1%}")
        else:
            print(f"{prefix}{name}: {value:.4f}{unit}")
    else:
        print(f"{prefix}{name}: {value}{unit}")


def print_dict(
    data: Dict[str, Any], title: Optional[str] = None, indent: int = 1
) -> None:
    """
    Print a dictionary in a formatted way.

    Args:
        data: Dictionary to print
        title: Optional title
        indent: Indentation level
    """
    if title:
        print_info(f"\n{Colors.BOLD}{title}:{Colors.ENDC}", indent - 1)

    for key, value in data.items():
        if isinstance(value, dict):
            print_info(f"{key}:", indent)
            print_dict(value, indent=indent + 1)
        elif isinstance(value, (list, tuple)):
            print_info(f"{key}: {', '.join(map(str, value))}", indent)
        else:
            print_info(f"{key}: {value}", indent)


def safe_import(module_name: str, class_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module or class with error handling.

    Args:
        module_name: Name of the module to import
        class_name: Optional class name to import from the module

    Returns:
        The imported module or class, or None if import fails
    """
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_name)
    except (ImportError, AttributeError) as e:
        print_warning(f"Could not import {module_name}.{class_name or ''}: {e}")
        return None


def check_dependencies(required_modules: List[str]) -> Dict[str, bool]:
    """
    Check if required modules are available.

    Args:
        required_modules: List of module names to check

    Returns:
        Dictionary mapping module names to availability status
    """
    results = {}
    for module in required_modules:
        try:
            __import__(module)
            results[module] = True
        except ImportError:
            results[module] = False
    return results


def handle_demo_error(error: Exception, demo_name: str) -> None:
    """
    Handle and display demo errors gracefully.

    Args:
        error: The exception that was raised
        demo_name: Name of the demo that failed
    """
    print_error(f"Error in {demo_name}: {str(error)}")
    print_info("\nError details:", 0)
    print_info(traceback.format_exc(), 1)


def run_demo_safely(
    demo_func: Callable, demo_name: str, skip_on_error: bool = True
) -> bool:
    """
    Run a demo function with error handling.

    Args:
        demo_func: The demo function to run
        demo_name: Name of the demo
        skip_on_error: Whether to continue on error

    Returns:
        True if the demo succeeded, False otherwise
    """
    try:
        demo_func()
        return True
    except Exception as e:
        handle_demo_error(e, demo_name)
        if not skip_on_error:
            raise
        return False


def print_demo_summary(demos: List[Dict[str, Any]]) -> None:
    """
    Print a summary of demo results.

    Args:
        demos: List of demo results with 'name' and 'success' keys
    """
    print_section("Demo Summary", level=1)

    total = len(demos)
    successful = sum(1 for d in demos if d.get("success", False))
    failed = total - successful

    print_info(f"Total demos: {total}")
    print_success(f"Successful: {successful}")
    if failed > 0:
        print_error(f"Failed: {failed}")

    if failed > 0:
        print_info("\nFailed demos:")
        for demo in demos:
            if not demo.get("success", False):
                print_info(f"  - {demo['name']}", 1)


def confirm_continue(message: str = "Press Enter to continue...") -> bool:
    """
    Ask user to confirm before continuing.

    Args:
        message: The confirmation message

    Returns:
        True if user confirms, False otherwise
    """
    try:
        input(f"\n{message}")
        return True
    except (KeyboardInterrupt, EOFError):
        print_warning("\nDemo interrupted by user")
        return False


def get_timestamp() -> str:
    """Get a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def print_feature_not_implemented(
    feature_name: str, coming_in: Optional[str] = None
) -> None:
    """
    Print a message indicating that a feature is not yet implemented.

    Args:
        feature_name: Name of the feature
        coming_in: Optional version/phase where feature will be available
    """
    msg = f"Feature '{feature_name}' is not yet implemented"
    if coming_in:
        msg += f" (coming in {coming_in})"
    print_warning(msg)
    print_info("This is a demonstration of planned functionality", 1)


def create_mock_result(result_type: str = "success", **kwargs) -> Dict[str, Any]:
    """
    Create a mock result for demonstration purposes.

    Args:
        result_type: Type of result ('success', 'error', 'warning')
        **kwargs: Additional fields to include in the result

    Returns:
        Mock result dictionary
    """
    result = {
        "status": result_type,
        "timestamp": get_timestamp(),
        "demo_mode": True,
    }
    result.update(kwargs)
    return result


def print_next_steps(steps: List[str], title: str = "Next Steps") -> None:
    """
    Print a list of next steps.

    Args:
        steps: List of next step descriptions
        title: Title for the next steps section
    """
    print_section(title, level=2)
    for i, step in enumerate(steps, 1):
        print_info(f"{i}. {step}", 0)


def print_key_features(features: List[str], title: str = "Key Features") -> None:
    """
    Print a list of key features.

    Args:
        features: List of feature descriptions
        title: Title for the features section
    """
    print_section(title, level=2)
    for feature in features:
        print_success(feature)
