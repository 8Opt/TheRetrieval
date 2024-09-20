import os
import sys
import json
import yaml
from pathlib import Path


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML: {e}")


def json_to_str(payload: dict) -> str:
    return json.dumps(payload)


def str_to_json(payload: str) -> dict:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def get_to_root(): 
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # Root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def ignore_warning():
    import warnings
    warnings.filterwarnings("ignore")