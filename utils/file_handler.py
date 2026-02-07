"""File utilities."""

import os
import json
import pickle


def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def save_file(path, data, format='json'):
    """Save data to file"""
    ensure_dir(os.path.dirname(path) or '.')
    
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(path, 'w') as f:
            f.write(str(data))


def load_file(path, format='json'):
    """Load data from file"""
    if not os.path.exists(path):
        return None
    
    if format == 'json':
        with open(path, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'r') as f:
            return f.read()
