#!/usr/bin/env python3
"""V6 llm_verification: real local GGUF load + structured decision output.

Asserts: file present, llama_cpp loads, tokens generate, JSON parses,
goal schema validates, no shell in output. Writes diagnostics/v6/llm_verification.json.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "llm_verification.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


GOAL_SCHEMA = {"goal", "urgency"}


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()}
    path = os.path.join(PROJECT_ROOT, "models", "text_model", "Qwen3-0.6B-Q8_0.gguf")
    if not os.path.exists(path):
        rep.update({"status": "UNAVAILABLE", "reason": "GGUF file absent; run scripts/acquire_models.py text"})
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
        print("LLM UNAVAILABLE: file absent")
        return
    rep.update({"model_id": "Qwen/Qwen3-0.6B-GGUF", "file": os.path.basename(path),
                "sha256": sha256(path), "size": os.path.getsize(path),
                "backend": "llama_cpp"})
    try:
        from llama_cpp import Llama
        t0 = time.time()
        llm = Llama(model_path=path, n_ctx=2048, n_threads=6, verbose=False)
        rep["load_s"] = round(time.time() - t0, 1)
        text = ""
        for attempt, ntok in enumerate((256, 512)):
            prompt = ("Output ONLY compact JSON, no thinking, no prose: "
                      '{"goal": "explore", "urgency": 0.5}')
            t0 = time.time()
            out = llm.create_chat_completion(
                messages=[{"role": "system",
                           "content": "Terse robot. JSON only. Never explain."},
                          {"role": "user", "content": prompt}],
                max_tokens=ntok, temperature=0.0, seed=42)
            if attempt == 0:
                rep["gen_s"] = round(time.time() - t0, 1)
            text = out["choices"][0]["message"]["content"]
            if "{" in text and '"goal"' in text:
                break
        rep["raw"] = text[:300]
        # last balanced {...} span (think traces may contain braces)
        start = text.rfind('{"goal"')
        depth, end = 0, -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        assert start >= 0 and end > start, "no goal JSON in output"
        data = json.loads(text[start:end])
        assert GOAL_SCHEMA <= set(data), f"schema violation: {data}"
        assert isinstance(data["urgency"], (int, float)) and 0 <= data["urgency"] <= 1
        for bad in ("os.system", "subprocess", "rm -", "eval("):
            assert bad not in text, f"unsafe content: {bad}"
        rep["structured"] = data
        rep["status"] = "PASS"
    except Exception as e:  # noqa: BLE001
        rep.update({"status": "FAIL", "error": f"{type(e).__name__}: {e}"})
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"LLM {rep['status']}: load={rep.get('load_s')}s gen={rep.get('gen_s')}s "
          f"out={rep.get('structured')}")


if __name__ == "__main__":
    main()
