"""Read and save the public training configuration."""
from pathlib import Path
import yaml


def load_training_config(config_path):
    with open(config_path) as stream:
        return yaml.safe_load(stream)


def save_config_to_output(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'training_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
