#!/usr/bin/env python3
"""
FlyBrain - Autonomous Biological Connectome Framework Entrypoint.
Supports rich subcommands: run, lab, experiment (run, verify, compare),
acceptance-matrix, docs-verify, diagnostics, test, validate-vulkan, benchmark, evolve.
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.connectome.types import GraphMode
from src.connectome.loader import get_or_create_circuit, load_raw_neurons
from src.brain.runtime import BrainRuntime
from src.brain.simulation_engine import SimulationEngine
from src.memory.persistence import PersistentMemoryManager
from src.trainer.curriculum import CurriculumTrainer
from src.evolution.scheduler import EvolutionScheduler
from src.dream.engine import DreamEngine
from src.experiment.manager import ExperimentManager
from src.compute.validator import run_cpu_gpu_validation

def run_diagnostics():
    from scripts.detect_env import main as detect_main
    from scripts.generate_manifests import main as manifest_main
    from scripts.validate_malecns_data import main as malecns_main
    
    print("Collecting system environment diagnostics...")
    detect_main()
    print("Validating biological malecns connectome source...")
    malecns_main()
    print("Generating dependency and reproducibility manifests...")
    manifest_main()
    print("Running Vulkan GPU vs CPU reference validation...")
    run_cpu_gpu_validation()
    print("\nAll diagnostics and validation manifests generated.")

def run_app(host="127.0.0.1", port=8080):
    import uvicorn
    from src.ui.server import app
    print(f"Starting FlyBrain Lab Scientific Workstation at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

def run_acceptance_matrix():
    from scripts.run_acceptance_matrix import evaluate_acceptance_matrix
    rep = evaluate_acceptance_matrix()
    if rep["failed"] > 0:
        sys.exit(1)

def run_docs_verify():
    from scripts.verify_docs_consistency import verify_docs_consistency
    ok = verify_docs_consistency()
    if not ok:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="FlyBrain - Autonomous Biological Connectome Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Optional backward compatibility flag
    parser.add_argument("--mode", choices=["run", "lab", "diagnostics", "validate-vulkan", "benchmark", "evolve", "test", "acceptance-matrix", "docs-verify"], default=None,
                        help="Legacy execution mode selector")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for UI server")
    parser.add_argument("--port", type=int, default=8080, help="Port for UI server")
    parser.add_argument("--circuit-size", type=int, default=512, help="Number of neurons in connectome circuit")
    parser.add_argument("--generations", type=int, default=3, help="Evolution generations")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")

    subparsers = parser.add_subparsers(dest="subcommand", help="FlyBrain Subcommands")

    # flybrain run / lab
    sub_run = subparsers.add_parser("run", help="Start FlyBrain Lab interactive server")
    sub_run.add_argument("--host", default="127.0.0.1")
    sub_run.add_argument("--port", type=int, default=8080)

    sub_lab = subparsers.add_parser("lab", help="Launch FlyBrain Lab scientific workstation")
    sub_lab.add_argument("--host", default="127.0.0.1")
    sub_lab.add_argument("--port", type=int, default=8080)

    # flybrain experiment (run, verify, compare)
    sub_exp = subparsers.add_parser("experiment", help="Manage deterministic research experiments")
    exp_subs = sub_exp.add_subparsers(dest="exp_command", help="Experiment subcommands")

    exp_run = exp_subs.add_parser("run", help="Execute an experiment")
    exp_run.add_argument("--spec", default=None, help="Path to experiment JSON spec file")
    exp_run.add_argument("--id", default=None, help="Custom experiment identifier")
    exp_run.add_argument("--seed", type=int, default=42, help="PRNG seed")
    exp_run.add_argument("--mode", choices=["REAL", "REAL_SUBGRAPH", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"], default="REAL", help="Connectome graph mode (REAL = REAL_SUBGRAPH sampled)")
    exp_run.add_argument("--scale", type=int, default=256, help="Number of neurons")
    exp_run.add_argument("--steps", type=int, default=50, help="Number of simulation steps")
    exp_run.add_argument("--no-gpu", action="store_true", help="Force CPU reference backend")

    exp_verify = exp_subs.add_parser("verify", help="Verify bitwise replication of an experiment")
    exp_verify.add_argument("--result", required=True, help="Experiment ID or path to experiment manifest JSON")

    exp_compare = exp_subs.add_parser("compare", help="Compare two experiment results")
    exp_compare.add_argument("--a", required=True, help="First experiment ID or manifest path")
    exp_compare.add_argument("--b", required=True, help="Second experiment ID or manifest path")

    # flybrain acceptance-matrix
    subparsers.add_parser("acceptance-matrix", help="Run canonical release acceptance matrix (verification/acceptance_schema.json)")

    # flybrain docs-verify
    subparsers.add_parser("docs-verify", help="Verify consistency between code, shaders, and documentation")

    # flybrain diagnostics
    subparsers.add_parser("diagnostics", help="Run full environment & hardware diagnostic suite")

    # flybrain validate-vulkan
    subparsers.add_parser("validate-vulkan", help="Run CPU vs Vulkan GPU compute parity validation suite")

    # flybrain test
    subparsers.add_parser("test", help="Run full test suite")

    # flybrain doctor
    subparsers.add_parser("doctor", help="Verify real environment: dataset, shaders, GPU, LLM, runtime")

    # flybrain version
    subparsers.add_parser("version", help="Print FlyBrain version")

    # flybrain benchmark
    subparsers.add_parser("benchmark", help="Run curriculum performance benchmarks")

    # flybrain evolve
    sub_evolve = subparsers.add_parser("evolve", help="Run genetic evolution loop on connectome")
    sub_evolve.add_argument("--circuit-size", type=int, default=512)
    sub_evolve.add_argument("--generations", type=int, default=3)
    sub_evolve.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Determine execution target
    target = args.subcommand or args.mode or "run"

    if target in ("run", "lab"):
        h = getattr(args, "host", "127.0.0.1")
        p = getattr(args, "port", 8080)
        run_app(host=h, port=p)
    elif target == "experiment":
        exp_mgr = ExperimentManager()
        if args.exp_command == "run":
            if args.spec and os.path.exists(args.spec):
                with open(args.spec, "r", encoding="utf-8") as f:
                    spec_data = json.load(f)
                from src.connectome.types import coerce_graph_mode as _coerce
                seed = spec_data.get("seed", 42)
                mode = _coerce(spec_data.get("graph_mode", "REAL"))
                scale = spec_data.get("neuron_scale", 256)
                steps = spec_data.get("duration_steps", 50)
                use_gpu = spec_data.get("use_gpu", True)
                exp_id = spec_data.get("experiment_id", None)
            else:
                from src.connectome.types import coerce_graph_mode
                seed = args.seed
                mode = coerce_graph_mode(args.mode)
                scale = args.scale
                steps = args.steps
                use_gpu = not args.no_gpu
                exp_id = args.id

            manifest = exp_mgr.run_experiment(
                experiment_id=exp_id,
                seed=seed,
                graph_mode=mode,
                neuron_scale=scale,
                duration_steps=steps,
                use_gpu=use_gpu
            )
            print(f"Experiment completed successfully: {manifest.experiment_id}")
            print(f"Final State Hash: {manifest.final_state_hash}")
            print(f"Mean Latency: {manifest.metrics['mean_step_latency_ms']} ms ({manifest.metrics['throughput_steps_per_sec']} steps/sec)")
        elif args.exp_command == "verify":
            exp_id = args.result
            if exp_id.endswith(".json"):
                exp_id = os.path.splitext(os.path.basename(exp_id))[0]
            res = exp_mgr.verify_experiment(exp_id)
            print(json.dumps(res, indent=2))
            if res["status"] != "PASS":
                sys.exit(1)
        elif args.exp_command == "compare":
            id_a = os.path.splitext(os.path.basename(args.a))[0] if args.a.endswith(".json") else args.a
            id_b = os.path.splitext(os.path.basename(args.b))[0] if args.b.endswith(".json") else args.b
            comp = exp_mgr.compare_experiments(id_a, id_b)
            print(json.dumps(comp, indent=2))
        else:
            sub_exp.print_help()
    elif target == "acceptance-matrix":
        run_acceptance_matrix()
    elif target == "docs-verify":
        run_docs_verify()
    elif target == "diagnostics":
        run_diagnostics()
    elif target == "validate-vulkan":
        run_cpu_gpu_validation()
    elif target == "benchmark":
        import scripts.run_curriculum_benchmark as bm
        bm.run_benchmarks()
    elif target == "evolve":
        circuit = get_or_create_circuit(args.circuit_size)
        evo = EvolutionScheduler(circuit)
        for g in range(args.generations):
            evo.run_generation(num_candidates=4, seed=args.seed + g*10)
    elif target == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        if not res.wasSuccessful():
            sys.exit(1)
    elif target == "doctor":
        from src.diagnostics.doctor import main as doctor_main
        sys.exit(doctor_main())
    elif target == "version":
        from src.version import VERSION
        print(f"FlyBrain {VERSION}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
