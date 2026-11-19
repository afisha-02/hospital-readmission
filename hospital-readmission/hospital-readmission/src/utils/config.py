"""Configuration loader and schema validator."""
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_project_root() -> Path:
    return PROJECT_ROOT

def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
            
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

def resolve_path(rel_or_abs_path: str) -> Path:
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
