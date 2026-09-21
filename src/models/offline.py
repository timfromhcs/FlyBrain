"""Offline enforcement (V6 phase_28): FLYBRAIN_OFFLINE=1 propagates to every
library that might phone home (HF hub, transformers, datasets, telemetry).
Call apply_offline_env() at process startup (server, tests, scripts).
"""
import os

OFFLINE_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


def is_offline() -> bool:
    return os.environ.get("FLYBRAIN_OFFLINE", "0") == "1"


def apply_offline_env() -> bool:
    if not is_offline():
        return False
    for k, v in OFFLINE_VARS.items():
        os.environ.setdefault(k, v)
    return True


apply_offline_env()
