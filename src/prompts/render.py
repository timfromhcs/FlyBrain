"""Render + validate prompt templates (prompts/templates.json).

Rules enforced: every template declares system/user/schema/temperature/
max_tokens; every render fills all placeholders; every parsed output must
contain the schema keys (validated by callers via check_output).
"""
import json
import os
import string
from typing import Any, Dict

TEMPLATE_PATH = os.path.join("prompts", "templates.json")


def load_templates(path: str = TEMPLATE_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "version" in data and isinstance(data["version"], str)
    return data


def render(name: str, templates: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    if name not in templates or not isinstance(templates[name], dict):
        raise KeyError(f"unknown template {name!r}")
    t = templates[name]
    for field in ("system", "user", "schema", "temperature", "max_tokens"):
        if field not in t:
            raise ValueError(f"template {name!r} missing {field!r}")
    needed = {n for _, n, _, _ in string.Formatter().parse(t["user"]) if n}
    missing = needed - set(kwargs)
    if missing:
        raise ValueError(f"template {name!r} missing params {sorted(missing)}")
    return {"system": t["system"], "user": t["user"].format(**kwargs),
            "schema": list(t["schema"]), "temperature": float(t["temperature"]),
            "max_tokens": int(t["max_tokens"]), "template_version": templates["version"]}


def extract_json(text: str, key: str) -> Dict[str, Any]:
    """Last-balanced-object extraction for thinking models."""
    start = text.rfind('{"' + key + '"')
    if start < 0:
        start = text.find("{")
    depth, end = 0, -1
    for i in range(max(start, 0), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if start < 0 or end <= start:
        raise ValueError(f"no JSON with {key!r} in output")
    data = json.loads(text[start:end])
    if key not in data:
        raise ValueError(f"output missing {key!r}")
    return data


def check_output(data: Dict[str, Any], schema: list) -> Dict[str, Any]:
    missing = [k for k in schema if k not in data]
    if missing:
        raise ValueError(f"output missing schema keys {missing}")
    return data
