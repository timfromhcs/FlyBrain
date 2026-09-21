"""Model Registry v8 Loader & Validator.

Loads models/registry.yaml, validates schema fields, and queries availability.
"""
import os
import yaml
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_YAML = os.path.join(PROJECT_ROOT, "models", "registry.yaml")


def load_registry(yaml_path: str = REGISTRY_YAML) -> List[Dict[str, Any]]:
    """Loads and validates structured model registry."""
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = data.get("models", [])
    
    # Check local disk presence for each entry
    for m in models:
        # Check explicit path relative to models/ or by task dir
        rel_path = m.get("path") or os.path.join(m["task"].lower(), m["file"])
        target_file = os.path.join(PROJECT_ROOT, "models", rel_path)
        m["local_path"] = target_file
        m["present"] = os.path.exists(target_file)
        if m["present"]:
            m["actual_bytes"] = os.path.getsize(target_file)
        else:
            m["actual_bytes"] = 0
            
    return models


def get_model_spec(model_id: str) -> Optional[Dict[str, Any]]:
    models = load_registry()
    for m in models:
        if m["id"] == model_id:
            return m
    return None
