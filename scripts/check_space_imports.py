#!/usr/bin/env python3
"""Simulate the slim Space env: block heavy modules, then import the server chain."""
import builtins
import sys

sys.path.insert(0, ".")
BLOCKED = {"torch", "diffusers", "transformers", "llama_cpp", "sounddevice",
           "soundfile", "vulkan"}

real_import = builtins.__import__


def guard(name, *a, **k):
    if name.split(".")[0] in BLOCKED:
        raise ModuleNotFoundError("blocked for Space simulation: " + name)
    return real_import(name, *a, **k)


builtins.__import__ = guard
try:
    import src.ui.server  # noqa
    print("server imports OK without heavy deps")
except Exception as e:
    print("SERVER IMPORT FAILS:", type(e).__name__, str(e)[:500])
