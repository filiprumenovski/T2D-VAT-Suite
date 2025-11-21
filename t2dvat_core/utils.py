"""
Utility functions for common operations.

Helper functions for data manipulation, file I/O, and general utilities.
"""

from pathlib import Path


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
