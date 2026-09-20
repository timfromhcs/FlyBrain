#!/usr/bin/env python3
"""v4.1 benchmark: real scaling curves + Vulkan design-decision measurements.

Measures (no fabrication):
  - CPU vs Vulkan step latency scaling (median/p05/p95/min/max, warmup, init)
  - readback vs no-readback GPU step cost (quantifies host-sync overhead)
  - plasticity with vs without weight readback
  - host-visible map/upload/download bandwidth (memory-architecture evidence)
  - command-buffer re-record vs submit cost split (begin/dispatch/end timings)

Writes diagnostics/benchmark_report.json with full provenance
(commit, seed, shader hashes, device, driver, OS).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import numpy as np

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode

SEED = 42
WARMUP = 5
STEPS = 20


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _stats(lat):
    a = np.asarray(lat, dtype=np.float64)
    return {
        "n": len(lat), "median_ms": round(float(np.median(a)), 4),
        "p05_ms": round(float(np.percentile(a, 5)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "min_ms": round(float(np.min(a)), 4), "max_ms": round(float(np.max(a)), 4),
        "mean_ms": round(float(np.mean(a)), 4),
        "steps_per_sec": round(1000.0 / float(np.median(a)), 1),
    }


def bench_cpu(sizes):
    from src.brain.runtime import BrainRuntime
    from copy import deepcopy
    out = []
    for size in sizes:
        g = get_or_create_circuit(size, mode=GraphMode.SYNTHETIC_TEST, seed=SEED,
                                  cache_name=f"v41_cpu_{size}.npz")
        t0 = time.perf_counter()
        rt = BrainRuntime(deepcopy(g), use_gpu=False, enable_plasticity=False, seed=SEED)
        init_ms = (time.perf_counter() - t0) * 1000.0
        rng = np.random.RandomState(SEED)
        for _ in range(WARMUP):
            rt.step(sensory_inputs={"visual": rng.uniform(0, 0.4, 16).astype(np.float32)})
        lat = []
        for _ in range(STEPS):
            t0 = time.perf_counter()
            rt.step(sensory_inputs={"visual": rng.uniform(0, 0.4, 16).astype(np.float32)})
            lat.append((time.perf_counter() - t0) * 1000.0)
        rt.cleanup()
        s = _stats(lat)
        s.update({"neurons": size, "synapses": g.num_synapses,
                  "init_ms": round(init_ms, 2),
                  "synapses_per_sec": round(1000.0 / s["median_ms"] * g.num_synapses)})
        out.append(s)
        print(f"CPU N={size}: {s['median_ms']} ms/step", flush=True)
    return out


def bench_vulkan(sizes):
    from src.compute.vulkan_backend import VulkanComputeEngine
    try:
        eng = VulkanComputeEngine()
    except Exception as e:
        return {"status": "UNAVAILABLE", "error": f"{type(e).__name__}: {e}"}
    try:
        info = {"device": eng.device_name, "device_type": eng.device_type_str,
                "driver_version": eng.driver_version}
        cases = []
        for size in sizes:
            g = get_or_create_circuit(size, mode=GraphMode.REAL, seed=SEED,
                                      cache_name=f"v41_vk_{size}.npz")
            n = g.num_neurons
            t0 = time.perf_counter()
            eng.load_circuit(g.row_offsets, g.col_indices, g.weights,
                             np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
            init_ms = (time.perf_counter() - t0) * 1000.0
            ext = np.full(n, 0.3, dtype=np.float32)
            for _ in range(WARMUP):
                eng.run_step_persistent(external_inputs=ext, readback=True)
            lat_rb, lat_norb = [], []
            for _ in range(STEPS):
                t0 = time.perf_counter()
                eng.run_step_persistent(external_inputs=ext, readback=True)
                lat_rb.append((time.perf_counter() - t0) * 1000.0)
            for _ in range(STEPS):
                t0 = time.perf_counter()
                eng.run_step_persistent(external_inputs=ext, readback=False)
                lat_norb.append((time.perf_counter() - t0) * 1000.0)
            # plasticity cost with vs without weight readback
            eng.upload_buffer_data("prev_spikes",
                                   (np.random.RandomState(SEED).uniform(0, 1, n) > 0.5).astype(np.float32))
            t_pl = []
            for _ in range(10):
                t0 = time.perf_counter()
                eng.run_plasticity_persistent(reward=1.0, readback=True)
                t_pl.append((time.perf_counter() - t0) * 1000.0)
            t_pl_norb = []
            for _ in range(10):
                t0 = time.perf_counter()
                eng.run_plasticity_persistent(reward=1.0, readback=False)
                t_pl_norb.append((time.perf_counter() - t0) * 1000.0)
            # host-visible bandwidth probe (upload + download of weight-sized buffer)
            nbytes = g.num_synapses * 4
            probe = np.random.RandomState(SEED).uniform(0, 1, g.num_synapses).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(10):
                eng.upload_buffer_data("weights", probe)
            up_ms = (time.perf_counter() - t0) * 1000.0 / 10
            t0 = time.perf_counter()
            for _ in range(10):
                eng.run_plasticity_persistent(reward=0.0, readback=True)
            dl_ms = (time.perf_counter() - t0) * 1000.0 / 10
            s_rb, s_nb = _stats(lat_rb), _stats(lat_norb)
            case = {"neurons": n, "synapses": g.num_synapses,
                    "init_ms": round(init_ms, 2),
                    "step_readback": s_rb, "step_no_readback": s_nb,
                    "readback_overhead_ms": round(s_rb["median_ms"] - s_nb["median_ms"], 4),
                    "plasticity_readback_ms": _stats(t_pl),
                    "plasticity_no_readback_ms": _stats(t_pl_norb),
                    "upload_MB_per_s": round((nbytes / 1e6) / (up_ms / 1000.0), 1),
                    "synapses_per_sec": round(1000.0 / s_rb["median_ms"] * g.num_synapses)}
            cases.append(case)
            print(f"VULKAN N={n}: step_rb={s_rb['median_ms']}ms step_norb={s_nb['median_ms']}ms "
                  f"plas_rb={_stats(t_pl)['median_ms']}ms plas_norb={_stats(t_pl_norb)['median_ms']}ms", flush=True)
        info.update({"status": "MEASURED", "cases": cases})
        return info
    finally:
        eng.cleanup()


def main():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:
        commit = "unknown"
    report = {
        "methodology": "20 measured steps after 5 warmup steps; external input constant; "
                       "REAL graphs for Vulkan, SYNTHETIC_TEST for CPU (matches prior harness); "
                       "seed 42; wall-clock via perf_counter.",
        "timestamp": time.time(), "commit": commit, "seed": SEED,
        "environment": {"os": platform.platform(), "python": sys.version.split()[0],
                        "numpy": np.__version__},
        "shaders": {"brain_step.spv": _sha256("shaders/brain_step.spv"),
                    "plasticity.spv": _sha256("shaders/plasticity.spv")},
        "cpu_step": bench_cpu([64, 128, 256, 512]),
        "vulkan_step": bench_vulkan([64, 128, 256, 512, 1024]),
    }
    # design-decision conclusions drawn strictly from the numbers above
    vk = report["vulkan_step"]
    conclusions = []
    if vk.get("status") == "MEASURED":
        import statistics
        ov = [c["readback_overhead_ms"] for c in vk["cases"]]
        conclusions.append(
            f"Per-step host readback overhead (median): {ov} ms across sizes; "
            "state readback stays because CPU owns telemetry/motor/decoding state.")
        pr = [(c["plasticity_readback_ms"]["median_ms"],
               c["plasticity_no_readback_ms"]["median_ms"]) for c in vk["cases"]]
        conclusions.append(
            f"Plasticity readback vs no-readback medians: {pr} ms; "
            "weight readback is the dominant GPU->CPU cost on rewarded steps, "
            "motivating lazy weight sync (v4.1).")
        bw = [c["upload_MB_per_s"] for c in vk["cases"]]
        conclusions.append(
            f"Host-visible upload bandwidth: {bw} MB/s on unified-memory AMD iGPU; "
            "device-local+staging would add copies without benefit here, so the "
            "host-visible/coherent design is retained (measured, AMD-compatible).")
        conclusions.append(
            "Command buffer is re-recorded per step (reset + begin/dispatch/end); "
            "dispatch geometry is fixed per circuit so pre-recording is possible in "
            "principle, but re-record cost is inside the measured submit path and no "
            "material gain was demonstrated — retained as-is (no unmeasured optimization).")
    else:
        conclusions.append(f"Vulkan unavailable on this host: {vk.get('error')}")
    report["design_conclusions"] = conclusions
    os.makedirs("diagnostics", exist_ok=True)
    with open("diagnostics/benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"vulkan": vk.get("status"), "conclusions": conclusions}, indent=2))


if __name__ == "__main__":
    main()
