#!/usr/bin/env python3
"""V6 performance: real latencies for world/physics/perception/nav/LLM/
image(FAST+BALANCED+QUALITY)/STT/TTS/memory/save/restore/load-unload.
Writes diagnostics/v6/performance_report.json. No retained optimization
without measurement; failures recorded, never skipped silently.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "performance_report.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

R = {}


def timed(name, fn, rounds=1):
    lat = []
    out = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        out = fn()
        lat.append(time.perf_counter() - t0)
    lat = sorted(lat)
    R[name] = {"median_s": round(lat[len(lat) // 2], 4),
               "min_s": round(lat[0], 4), "max_s": round(lat[-1], 4),
               "rounds": rounds}
    return out


def ram():
    import psutil
    vm = psutil.virtual_memory()
    return {"percent": vm.percent, "avail_gb": round(vm.available / 1024 ** 3, 2)}


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip(),
           "ram_start": ram()}
    # world + physics + perception + nav (fast)
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=1, db_path=os.path.join(td, "w.db"))
        try:
            ag = svc.agents["hero"]
            timed("world_tick_2agents_s", lambda: svc.step(1), rounds=5)
            timed("physics_substep_s",
                  lambda: svc.world.physics.step(0.01), rounds=50)
            timed("perception_observe_s",
                  lambda: ag.eye.observe(svc.world, "hero", 0.0), rounds=5)
            from src.world3d.navigation import Navigator
            nav = Navigator(svc.world.spec)
            nav.observe_walkable(0.0, 4.0, radius=6.0)
            nav.observe_walkable(0.0, 10.0, radius=6.0)
            timed("nav_plan_s", lambda: nav.plan(0.0, 4.0, 0.0, 8.0), rounds=5)
            timed("memory_place_search_s",
                  lambda: ag.spatial.search_places("forest"), rounds=5)
            timed("world_save_s", lambda: svc.snapshot(), rounds=3)
            snap = svc.snapshot()
            timed("world_restore_s", lambda: svc.restore(snap), rounds=3)
        finally:
            svc.spatial.close()
    rep["ram_after_world"] = ram()
    # models (each timed with load/unload)
    try:
        from src.models.loaders import load_text
        t0 = time.time()
        llm = load_text()
        R["llm_load_s"] = {"median_s": round(time.time() - t0, 1), "rounds": 1}
        timed("llm_goal_gen_s", lambda _llm=llm: _llm.create_chat_completion(
            messages=[{"role": "user", "content": 'Reply ONLY: {"goal": "explore"}'}],
            max_tokens=64, temperature=0.0, seed=1), rounds=2)
        llm = None
    except Exception as e:
        R["llm"] = {"status": f"UNAVAILABLE:{type(e).__name__}"}
    import gc
    gc.collect()
    rep["ram_after_llm"] = ram()
    try:
        from src.models.image_adapter import LocalImageModel
        t0 = time.time()
        img = LocalImageModel()
        img.load()
        R["image_load_s"] = {"median_s": round(time.time() - t0, 1), "rounds": 1}
        for mode in ("FAST", "BALANCED", "QUALITY"):
            try:
                timed(f"image_{mode}_s", lambda m=mode: img.generate(
                    "a pine forest", mode=m, seed=5,
                    out_path=os.path.join(PROJECT_ROOT, "diagnostics", "v6",
                                          f"bench_{m.lower()}.png")), rounds=1)
            except Exception as e:
                R[f"image_{mode}_s"] = {"status": f"FAILED:{type(e).__name__}"}
        img.unload()
    except Exception as e:
        R["image"] = {"status": f"UNAVAILABLE:{type(e).__name__}"}
    gc.collect()
    rep["ram_after_image"] = ram()
    try:
        from src.models.loaders import load_stt
        t0 = time.time()
        stt = load_stt()
        R["stt_load_s"] = {"median_s": round(time.time() - t0, 1), "rounds": 1}
        import soundfile as sf
        wav = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "stt_probe.wav")
        data, sr = sf.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            import math as _m
            g = _m.gcd(sr, 16000)
            data = resample_poly(data, 16000 // g, sr // g).astype(data.dtype)
            sr = 16000
        feats = stt["processor"](data, sampling_rate=sr, return_tensors="pt").input_features
        import torch
        timed("stt_transcribe_s", lambda _stt=stt: _stt["model"].generate(feats, max_new_tokens=32),
              rounds=2)
        stt = None
    except Exception as e:
        R["stt"] = {"status": f"UNAVAILABLE:{type(e).__name__}"}
    gc.collect()
    try:
        from src.models.loaders import load_tts
        t0 = time.time()
        tts = load_tts()
        R["tts_load_s"] = {"median_s": round(time.time() - t0, 1), "rounds": 1}
        import torch as _t
        timed("tts_synth_s", lambda _tts=tts: list(_tts("Hello world.", voice="af_heart")), rounds=2)
        tts = None
    except Exception as e:
        R["tts"] = {"status": f"UNAVAILABLE:{type(e).__name__}"}
    gc.collect()
    rep["ram_end"] = ram()
    rep["results"] = R
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    for k, v in R.items():
        print(f"{k}: {v}")
    print("->", OUT)


if __name__ == "__main__":
    main()
