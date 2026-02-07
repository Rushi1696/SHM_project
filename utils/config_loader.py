"""Configuration loader utility."""

import yaml


def load_config(path):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load(path):
    """Alias for load_config for backward compatibility"""
    return load_config(path)
