"""
Feature configuration loader and validator.
Supports YAML-based feature variant configurations.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_feature_config(config_path: str) -> Dict[str, Any]:
    """
    Load feature configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['name', 'extractor', 'normalization', 'segment']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    return config



def save_config_to_output(config: Dict[str, Any], output_dir: Path):
    """
    Save configuration file to output directory for reproducibility.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config as YAML
    config_file = output_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
