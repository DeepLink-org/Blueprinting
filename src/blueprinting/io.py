"""IO utilities for blueprinting."""

import json
import os
from typing import Any, Dict

__all__ = ["read_json_file", "write_json_file", "is_json_extension"]


def read_json_file(filepath: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with file contents
    """
    with open(filepath, "r") as f:
        return json.load(f)


def write_json_file(data: Dict[str, Any], filepath: str) -> None:
    """Write data to a JSON file.

    Args:
        data: Dictionary to write
        filepath: Path to the output file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def is_json_extension(filepath: str) -> bool:
    """Check if a filepath has a JSON extension.

    Args:
        filepath: Path to check

    Returns:
        True if the file has a .json extension
    """
    _, ext = os.path.splitext(filepath)
    return ext.lower() == ".json"
