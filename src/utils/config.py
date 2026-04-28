"""
Load YAML configuration files into a nested dict.
"""

import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dict mirroring the YAML structure.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        yaml.YAMLError: if the YAML is malformed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)