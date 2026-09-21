#!/usr/bin/env python3
"""Identify which local-model load touches the network."""
import os
import socket
import sys
import traceback
sys.path.insert(0, os.path.abspath("."))

_real = socket.socket.connect


def _guard(self, *a, **k):
    print(f"CONNECT attempt to {a[0] if a else '?'}", flush=True)
    traceback.print_stack(limit=8)
    raise RuntimeError("blocked")


socket.socket.connect = _guard
for name in ("embedding", "text"):
    try:
        if name == "embedding":
            from src.models.loaders import load_embedding
            m = load_embedding()
            print(name, "OK, no connect")
            del m
        else:
            from src.models.loaders import load_text
            m = load_text()
            print(name, "OK, no connect")
    except RuntimeError as e:
        print(name, "CONNECTS:", e)
    except Exception as e:
        print(name, "other error:", type(e).__name__, str(e)[:150])
