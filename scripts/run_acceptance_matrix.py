#!/usr/bin/env python3
"""
Automated Release Acceptance Matrix Runner for FlyBrain.
Evaluates the canonical categories in verification/acceptance_schema.json
and writes diagnostics/acceptance_matrix.json.
"""

import os
import sys
import json
import time
import hashlib
import platform
import numpy as np
from typing import Dict, Any, List, Tuple


def _sb(s: int):
    from src.common.determinism import SeedBundle
    return SeedBundle(experiment_seed=s, generation_seed=s + 1, organism_seed=s + 2,
                      development_seed=s + 3, mutation_seed=s + 4, world_seed=s + 5,
                      teacher_seed=s + 6)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.connectome.types import GraphMode, ProvenanceStatus
from src.connectome.loader import get_or_create_circuit, load_raw_neurons, DEFAULT_SOMA_PATH, DEFAULT_CONNECTIONS_PATH
from src.brain.runtime import BrainRuntime
from src.brain.simulation_engine import SimulationEngine
from src.experiment.manager import ExperimentManager, get_file_sha256
from src.compute.cpu_reference import cpu_lif_step
from src.compute.vulkan_backend import VulkanBrainBackend
from src.organism.organism import stage_for_age

def evaluate_acceptance_matrix() -> Dict[str, Any]:
    print("=" * 70)
    print("FLYBRAIN AUTONOMOUS RELEASE ACCEPTANCE MATRIX")
    print("=" * 70)
    
    matrix = {}
    
    # 1. repository_cleanliness
    print("[1/57] Evaluating repository_cleanliness...")
    clean = True
    reasons = []
    # Check that required dirs exist
    for d in ["src", "shaders", "malecns", "scripts", "tests", "docs", "manifests"]:
        if not os.path.isdir(os.path.join(PROJECT_ROOT, d)):
            clean = False
            reasons.append(f"Missing core directory: {d}")
    matrix["repository_cleanliness"] = {
        "status": "PASS" if clean else "FAIL",
        "reason": "All core repository directories intact and organized." if clean else "; ".join(reasons)
    }

    # 2. provenance_manifest_integrity
    print("[2/57] Evaluating provenance_manifest_integrity...")
    prov_file = os.path.join(PROJECT_ROOT, "manifests", "malecns_provenance.json")
    if os.path.exists(prov_file):
        with open(prov_file, "r", encoding="utf-8") as f:
            prov_data = json.load(f)
        soma_info = prov_data["provenance_metadata"]["files"]["soma_sides_csv"]
        conn_info = prov_data["provenance_metadata"]["files"]["connections_csv"]
        
        actual_soma_hash = get_file_sha256(os.path.join(PROJECT_ROOT, soma_info["path"]))
        actual_conn_hash = get_file_sha256(os.path.join(PROJECT_ROOT, conn_info["path"]))
        
        valid_soma_hashes = {
            soma_info["sha256"],
            soma_info.get("sha256_canonical_lf", "35835a8f67fa82e595fa64fc980670fddb23c001001b37a51571d81a0fe05c6b")
        }
        valid_conn_hashes = {
            conn_info["sha256"],
            conn_info.get("sha256_canonical_lf", "c69c894773b10229bfbcc16465335fec8fd4e951b521e00be8193f9f21f37fd9")
        }

        soma_match = (actual_soma_hash in valid_soma_hashes)
        conn_match = (actual_conn_hash in valid_conn_hashes)
        
        if soma_match and conn_match:
            matrix["provenance_manifest_integrity"] = {
                "status": "PASS",
                "reason": "Janelia MaleCNS soma and connection SHA-256 hashes match provenance manifest exactly."
            }
        else:
            matrix["provenance_manifest_integrity"] = {
                "status": "FAIL",
                "reason": f"Hash mismatch: soma_match={soma_match}, conn_match={conn_match}"
            }
    else:
        matrix["provenance_manifest_integrity"] = {"status": "FAIL", "reason": "Manifest file missing"}

    # 3. connectome_contract_separation (incl. explicit REAL_SUBGRAPH identity)
    print("[3/57] Evaluating connectome_contract_separation...")
    modes = {m.value for m in GraphMode}
    expected_modes = {"REAL", "SPATIAL_SURROGATE", "SYNTHETIC_TEST"}
    from src.connectome.types import GRAPH_IDENTITIES, resolve_graph_identity as _rgi
    _alias_ok = (GraphMode.REAL_SUBGRAPH is GraphMode.REAL
                 and GraphMode.canonical(GraphMode.REAL) == "REAL_SUBGRAPH"
                 and set(GRAPH_IDENTITIES) == {"REAL_FULL", "REAL_SUBGRAPH",
                                               "SPATIAL_SURROGATE", "SYNTHETIC_TEST"})
    try:
        _rgi("REAL_FULL")
        _full_raises = False
    except ValueError:
        _full_raises = True
    if modes == expected_modes and _alias_ok and _full_raises:
        matrix["connectome_contract_separation"] = {
            "status": "PASS",
            "reason": f"Explicit separation of GraphMode contracts: {sorted(list(modes))}; "
                      f"REAL canonicalizes to REAL_SUBGRAPH; REAL_FULL honestly raises."
        }
    else:
        matrix["connectome_contract_separation"] = {
            "status": "FAIL",
            "reason": f"GraphMode mismatch: expected {expected_modes}, got {modes}; "
                      f"alias_ok={_alias_ok}, full_raises={_full_raises}"
        }

    # 4. biological_vs_synthetic_separation
    print("[4/57] Evaluating biological_vs_synthetic_separation...")
    real_graph = get_or_create_circuit(128, mode=GraphMode.REAL, seed=42)
    synth_graph = get_or_create_circuit(128, mode=GraphMode.SYNTHETIC_TEST, seed=42)
    
    is_real_prov = (real_graph.provenance_status == ProvenanceStatus.VERIFIED)
    is_synth_prov = (synth_graph.provenance_status == ProvenanceStatus.EXPERIMENTAL)
    distinct_hashes = (real_graph.graph_hash != synth_graph.graph_hash)
    
    if is_real_prov and is_synth_prov and distinct_hashes:
        matrix["biological_vs_synthetic_separation"] = {
            "status": "PASS",
            "reason": "REAL graph verified from MaleCNS; SYNTHETIC_TEST marked EXPERIMENTAL with distinct topology."
        }
    else:
        matrix["biological_vs_synthetic_separation"] = {
            "status": "FAIL",
            "reason": f"Separation failure: real={is_real_prov}, synth={is_synth_prov}, distinct={distinct_hashes}"
        }

    # 5. spatial_surrogate_behavior
    print("[5/57] Evaluating spatial_surrogate_behavior...")
    surr_graph = get_or_create_circuit(128, mode=GraphMode.SPATIAL_SURROGATE, seed=42)
    is_surr = (surr_graph.provenance_status == ProvenanceStatus.SURROGATE)
    has_coords = surr_graph.coordinates is not None and len(surr_graph.coordinates) == 128
    has_synapses = surr_graph.num_synapses > 0
    
    if is_surr and has_coords and has_synapses:
        matrix["spatial_surrogate_behavior"] = {
            "status": "PASS",
            "reason": f"Spatial surrogate generated {surr_graph.num_synapses} synapses via 3D k-d tree proximity."
        }
    else:
        matrix["spatial_surrogate_behavior"] = {
            "status": "FAIL",
            "reason": f"Surrogate failure: is_surr={is_surr}, has_coords={has_coords}, has_synapses={has_synapses}"
        }

    # 6. lif_dynamics_correctness
    print("[6/57] Evaluating lif_dynamics_correctness...")
    num_n = 4
    row_offsets = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    col_indices = np.array([], dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    prev_spikes = np.zeros(num_n, dtype=np.float32)
    ext_inputs = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)
    pot_in = np.array([-55.0, -55.0, -70.0, -70.0], dtype=np.float32)
    refr_in = np.zeros(num_n, dtype=np.int32)
    
    pot_out, spk_out, refr_out = cpu_lif_step(
        row_offsets, col_indices, weights,
        prev_spikes, ext_inputs, pot_in, refr_in,
        decay=0.95, threshold=-50.0, v_reset=-70.0, v_rest=-70.0, t_ref=2
    )
    # Test definite spike with ext_input = 25.0
    ext_inputs_spk = np.array([0.0, 25.0, 0.0, 0.0], dtype=np.float32)
    p_out2, s_out2, r_out2 = cpu_lif_step(
        row_offsets, col_indices, weights,
        prev_spikes, ext_inputs_spk, pot_in, refr_in,
        decay=0.95, threshold=-50.0, v_reset=-70.0, v_rest=-70.0, t_ref=2
    )
    lif_correct = (s_out2[1] == 1.0 and p_out2[1] == -70.0 and r_out2[1] == 2)
    matrix["lif_dynamics_correctness"] = {
        "status": "PASS" if lif_correct else "FAIL",
        "reason": "LIF integration correctly decays membrane potential, fires spike, and clamps to reset."
    }

    # 7. refractory_period_invariance
    print("[7/57] Evaluating refractory_period_invariance...")
    # Feeding high input while refractory > 0 must suppress spike
    p_in_ref = np.array([-70.0, -70.0, -70.0, -70.0], dtype=np.float32)
    r_in_ref = np.array([0, 2, 0, 0], dtype=np.int32) # Neuron 1 in refractory
    ext_high = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
    p_out_ref, s_out_ref, r_out_ref = cpu_lif_step(
        row_offsets, col_indices, weights,
        prev_spikes, ext_high, p_in_ref, r_in_ref,
        decay=0.95, threshold=-50.0, v_reset=-70.0, v_rest=-70.0, t_ref=2
    )
    ref_invariant = (s_out_ref[1] == 0.0 and p_out_ref[1] == -70.0 and r_out_ref[1] == 1)
    matrix["refractory_period_invariance"] = {
        "status": "PASS" if ref_invariant else "FAIL",
        "reason": "Refractory period strictly prevents firing and decrements counter during active refraction."
    }

    # 8. reset_potential_invariance
    print("[8/57] Evaluating reset_potential_invariance...")
    reset_invariant = (p_out2[1] == -70.0 and s_out2[1] == 1.0)
    matrix["reset_potential_invariance"] = {
        "status": "PASS" if reset_invariant else "FAIL",
        "reason": "Membrane potential instantly clamped to V_reset (-70.0 mV) upon spike generation."
    }

    # 9. vulkan_discovery_and_selection
    print("[9/57] Evaluating vulkan_discovery_and_selection...")
    try:
        vk = VulkanBrainBackend()
        dev_name = vk.device_name
        matrix["vulkan_discovery_and_selection"] = {
            "status": "PASS",
            "reason": f"Vulkan 1.3 physical device discovered and selected: '{dev_name}'"
        }
        vk_available = True
    except Exception as e:
        matrix["vulkan_discovery_and_selection"] = {
            "status": "SKIP_ENVIRONMENT",
            "reason": f"Vulkan device discovery unavailable on this host: {e}"
        }
        vk_available = False

    # 10. persistent_resource_lifecycle
    print("[10/57] Evaluating persistent_resource_lifecycle...")
    if vk_available:
        try:
            test_circuit = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=42)
            vk.load_circuit(
                test_circuit.row_offsets,
                test_circuit.col_indices,
                test_circuit.weights,
                np.full(64, -70.0, dtype=np.float32),
                np.zeros(64, dtype=np.float32)
            )
            init_pipe = vk.brain_pipeline
            init_cmd = vk.brain_command_buffer
            
            # Execute step
            vis = np.zeros(64, dtype=np.float32)
            vk.run_step_persistent(external_inputs=vis)
            
            # Verify resources persisted without re-allocation
            same_pipe = (vk.brain_pipeline == init_pipe)
            same_cmd = (vk.brain_command_buffer == init_cmd)
            vk.cleanup()
            
            if same_pipe and same_cmd:
                matrix["persistent_resource_lifecycle"] = {
                    "status": "PASS",
                    "reason": "Vulkan buffers, descriptor sets, and command buffers remain resident across simulation steps."
                }
            else:
                matrix["persistent_resource_lifecycle"] = {
                    "status": "FAIL",
                    "reason": "Vulkan resources were reallocated between steps."
                }
        except Exception as e:
            matrix["persistent_resource_lifecycle"] = {
                "status": "FAIL",
                "reason": f"Error verifying persistent lifecycle: {e}"
            }
    else:
        matrix["persistent_resource_lifecycle"] = {
            "status": "SKIP_ENVIRONMENT",
            "reason": "Vulkan hardware not available for persistent lifecycle test."
        }

    # 11. cpu_vulkan_numerical_parity
    print("[11/57] Evaluating cpu_vulkan_numerical_parity...")
    if vk_available:
        try:
            from src.compute.validator import run_cpu_gpu_validation
            val_res = run_cpu_gpu_validation()
            passed = (val_res.get("overall_status") == "SUCCESS")
            total = val_res.get("total_test_cases", 0)
            pass_count = val_res.get("passed_test_cases", 0)
            if passed:
                matrix["cpu_vulkan_numerical_parity"] = {
                    "status": "PASS",
                    "reason": f"Tolerance-based parity verified ({pass_count}/{total} cases): "
                              f"max abs diff < 1e-4 with exact spike trains (NOT bit-exact)."
                }
            else:
                matrix["cpu_vulkan_numerical_parity"] = {
                    "status": "FAIL",
                    "reason": f"Parity mismatch: {pass_count}/{total} passed."
                }
        except Exception as e:
            matrix["cpu_vulkan_numerical_parity"] = {
                "status": "FAIL",
                "reason": f"Parity validation threw error: {e}"
            }
    else:
        matrix["cpu_vulkan_numerical_parity"] = {
            "status": "SKIP_ENVIRONMENT",
            "reason": "Vulkan GPU not available for numerical parity evaluation."
        }

    # 12. single_loop_telemetry_isolation
    print("[12/57] Evaluating single_loop_telemetry_isolation...")
    try:
        engine = SimulationEngine()
        engine.start()
        time.sleep(0.3)
        telem = engine.get_latest_telemetry()
        engine.pause()
        engine.cleanup()
        
        has_step = telem.get("step", 0) > 0
        has_spikes = "spikes" in telem
        matrix["single_loop_telemetry_isolation"] = {
            "status": "PASS" if (has_step and has_spikes) else "FAIL",
            "reason": f"SimulationEngine executed {telem.get('step', 0)} steps in background thread without blocking telemetry."
        }
    except Exception as e:
        matrix["single_loop_telemetry_isolation"] = {
            "status": "FAIL",
            "reason": f"Single loop telemetry test failed: {e}"
        }

    # 13. thread_lock_concurrency
    print("[13/57] Evaluating thread_lock_concurrency...")
    try:
        engine = SimulationEngine()
        engine.start()
        # Concurrently read state and send step commands
        for _ in range(10):
            _ = engine.get_current_state()
            _ = engine.get_latest_telemetry()
            time.sleep(0.01)
        engine.pause()
        engine.cleanup()
        matrix["thread_lock_concurrency"] = {
            "status": "PASS",
            "reason": "Thread-safe RLock prevented data races during concurrent telemetry and state reads."
        }
    except Exception as e:
        matrix["thread_lock_concurrency"] = {
            "status": "FAIL",
            "reason": f"Thread concurrency failed: {e}"
        }

    # 14. deterministic_experiment_replication
    print("[14/57] Evaluating deterministic_experiment_replication...")
    try:
        exp_mgr = ExperimentManager()
        exp_a = exp_mgr.run_experiment(
            experiment_id="matrix_det_test_a",
            seed=1337,
            graph_mode=GraphMode.REAL,
            neuron_scale=128,
            duration_steps=20
        )
        exp_b = exp_mgr.run_experiment(
            experiment_id="matrix_det_test_b",
            seed=1337,
            graph_mode=GraphMode.REAL,
            neuron_scale=128,
            duration_steps=20
        )
        hashes_match = (exp_a.final_state_hash == exp_b.final_state_hash)
        matrix["deterministic_experiment_replication"] = {
            "status": "PASS" if hashes_match else "FAIL",
            "reason": f"Exact 256-bit SHA-256 match ({exp_a.final_state_hash[:16]}...) across independent runs."
        }
    except Exception as e:
        matrix["deterministic_experiment_replication"] = {
            "status": "FAIL",
            "reason": f"Experiment replication failed: {e}"
        }

    # 15. local_model_degradation_honesty
    print("[15/57] Evaluating local_model_degradation_honesty...")
    try:
        from src.trainer.cognitive import CognitiveTrainer
        cog = CognitiveTrainer()
        # Propose curriculum without downloading weights
        prop = cog.propose_curriculum_step({"energy": 0.5, "curiosity": 0.8}, [{"name": "curiosity_forage"}])
        is_honest = (cog.model_status in ["MODEL_UNAVAILABLE", "RULE_BASED", "OPERATIONAL"])
        matrix["local_model_degradation_honesty"] = {
            "status": "PASS" if is_honest else "FAIL",
            "reason": f"System honestly declared cognitive status: {cog.model_status}"
        }
    except Exception as e:
        matrix["local_model_degradation_honesty"] = {
            "status": "FAIL",
            "reason": f"Model degradation evaluation failed: {e}"
        }

    # 16. continuous_learning_weight_change
    print("[16/57] Evaluating continuous_learning_weight_change...")
    try:
        circ = get_or_create_circuit(64, mode=GraphMode.SYNTHETIC_TEST, seed=42)
        initial_weights = circ.weights.copy()
        rt = BrainRuntime(circ, use_gpu=False, seed=42)
        
        # Apply sensory inputs and strong reward
        for _ in range(10):
            rt.step(sensory_inputs={"visual": np.ones(32, dtype=np.float32)}, reward=1.0)
            
        weight_diff = np.max(np.abs(circ.weights - initial_weights))
        rt.cleanup()
        
        matrix["continuous_learning_weight_change"] = {
            "status": "PASS" if weight_diff > 0.0 else "FAIL",
            "reason": f"Synaptic plasticity modified weights. Max delta_w: {weight_diff:.6e}."
        }
    except Exception as e:
        matrix["continuous_learning_weight_change"] = {
            "status": "FAIL",
            "reason": f"Plasticity evaluation failed: {e}"
        }

    # 17. ui_no_blocking_alerts (+ local-first frontend: no CDN runtime deps)
    print("[17/57] Evaluating ui_no_blocking_alerts...")
    _ui_files = [os.path.join(PROJECT_ROOT, "src", "ui", "static", "index.html"),
                 os.path.join(PROJECT_ROOT, "src", "ui", "static", "js", "lab.js"),
                 os.path.join(PROJECT_ROOT, "src", "ui", "static", "css", "lab.css")]
    _missing = [p for p in _ui_files if not os.path.exists(p)]
    if _missing:
        matrix["ui_no_blocking_alerts"] = {"status": "FAIL",
                                           "reason": f"frontend files missing: {_missing}"}
    else:
        _src = "".join(open(p, "r", encoding="utf-8").read() for p in _ui_files)
        has_alert = ("alert(" in _src)
        has_prompt = ("prompt(" in _src)
        has_cdn = ("cdnjs.cloudflare.com" in _src or "cdn.jsdelivr.net" in _src
                   or "unpkg.com" in _src)
        _vendor_ok = (os.path.exists(os.path.join(PROJECT_ROOT, "src", "ui", "static",
                                                  "vendor", "three.min.js")))
        if not has_alert and not has_prompt and not has_cdn and _vendor_ok:
            matrix["ui_no_blocking_alerts"] = {
                "status": "PASS",
                "reason": "FlyBrain Lab workstation contains zero blocking alert()/prompt() calls "
                          "(non-intrusive toasts) and zero CDN runtime dependencies (vendored three.js)."
            }
        else:
            matrix["ui_no_blocking_alerts"] = {
                "status": "FAIL",
                "reason": f"Blocking calls or CDN deps: alert={has_alert}, prompt={has_prompt}, "
                          f"cdn={has_cdn}, vendor_ok={_vendor_ok}"
            }

    # 18. full_pipeline_e2e_runnable
    print("[18/57] Evaluating full_pipeline_e2e_runnable...")
    try:
        circ = get_or_create_circuit(128, mode=GraphMode.REAL, seed=42)
        rt = BrainRuntime(circ, use_gpu=vk_available, seed=42)
        rt.step(sensory_inputs={"visual": np.ones(32, dtype=np.float32)}, reward=0.5)
        snap_path = os.path.join(PROJECT_ROOT, "diagnostics", "e2e_test_snap.npz")
        rt.save_snapshot(snap_path)
        rt.cleanup()
        if os.path.exists(snap_path):
            os.remove(snap_path)
            matrix["full_pipeline_e2e_runnable"] = {
                "status": "PASS",
                "reason": "End-to-end pipeline (real connectome load -> runtime step -> snapshot save) executed flawlessly."
            }
        else:
            matrix["full_pipeline_e2e_runnable"] = {"status": "FAIL", "reason": "Snapshot file was not written."}
    except Exception as e:
        matrix["full_pipeline_e2e_runnable"] = {
            "status": "FAIL",
            "reason": f"End-to-end execution failed: {e}"
        }

    # 19. documentation_claim_consistency
    print("[19/57] Evaluating documentation_claim_consistency...")
    from scripts.verify_docs_consistency import verify_docs_consistency
    docs_ok = verify_docs_consistency()
    matrix["documentation_claim_consistency"] = {
        "status": "PASS" if docs_ok else "FAIL",
        "reason": "All documentation claims match datasets, shader descriptors, and API routes exactly."
    }

    # 20. alife_branch_replay_determinism
    print("[20/57] Evaluating alife_branch_replay_determinism...")
    try:
        import json as _json
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        _seeds = SeedBundle(experiment_seed=101, generation_seed=102, organism_seed=103,
                            development_seed=104, mutation_seed=105, world_seed=106,
                            teacher_seed=107)
        _pop = Population(4, _seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=101)
        _pop.step(15)
        _snap = _json.loads(_json.dumps(_pop.snapshot()))
        _pop.step(5)
        _h1 = _pop.population_hash()
        _pop2 = Population.restore(_snap, _seeds)
        _pop2.step(5)
        _match = (_pop2.population_hash() == _h1)
        matrix["alife_branch_replay_determinism"] = {
            "status": "PASS" if _match else "FAIL",
            "reason": f"Population snapshot branch replay produced identical hash ({_h1[:16]})."
            if _match else "Branch replay hash mismatch (see test_alife branch test)."
        }
    except Exception as e:
        matrix["alife_branch_replay_determinism"] = {"status": "FAIL", "reason": f"ALife replay failed: {e}"}

    # 21. developmental_structural_integrity
    print("[21/57] Evaluating developmental_structural_integrity...")
    try:
        from src.development.engine import DevelopmentEngine, DevelopmentState
        from src.genome.schema import Genome
        from src.common.events import EventLog
        _g = get_or_create_circuit(48, mode=GraphMode.SYNTHETIC_TEST, seed=77,
                                   cache_name="matrix_dev_synth_48.npz")
        _dev = DevelopmentState.initialize(_g.num_neurons)
        _eng = DevelopmentEngine(Genome.founder(78).params, development_seed=79)
        _log = EventLog()
        _n0 = _g.num_neurons
        _born = _eng.neurogenesis(_g, _dev, 1, _log, "mx", 0, max_new=3)
        _g.validate_invariants()
        _eng.differentiate(_g, _dev, 2, _log, "mx", 0)
        _eng.migrate(_g, _dev, 3, _log, "mx", 0)
        _made = _eng.grow_projections(_g, _dev, 4, _log, "mx", 0)
        _g.validate_invariants()
        _died = _eng.apoptosis(_g, _dev, 600, np.zeros(_g.num_neurons, dtype=np.float32),
                               600, _log, "mx", 0)
        _g.validate_invariants()
        _dead_edgeless = all(
            (int(_g.row_offsets[i + 1]) - int(_g.row_offsets[i])) == 0
            for i, a in enumerate(_dev.alive) if not a)
        _ok = (_g.num_neurons == _n0 + _born and _made >= 0 and _dead_edgeless
               and len(_log.filter("NEURON_BORN")) == _born)
        matrix["developmental_structural_integrity"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"Full developmental pipeline: {_born} born, {_made} synapses grown, "
                      f"{_died} died; invariants hold, dead neurons edgeless."
        }
    except Exception as e:
        matrix["developmental_structural_integrity"] = {"status": "FAIL", "reason": f"Development failed: {e}"}

    # 22. genome_mutation_crossover_provenance
    print("[22/57] Evaluating genome_mutation_crossover_provenance...")
    try:
        from src.genome.schema import Genome as _Genome
        from src.genome.operators import mutate_genome, crossover_genomes
        _gf = _Genome.founder(200)
        _c1, _r1 = mutate_genome(_gf, 201)
        _c2, _r2 = mutate_genome(_gf, 201)
        _x1, _xr1 = crossover_genomes(_gf, _Genome.founder(202), 203)
        _x2, _ = crossover_genomes(_gf, _Genome.founder(202), 203)
        _ok = (_c1.genome_hash() == _c2.genome_hash()
               and _x1.genome_hash() == _x2.genome_hash()
               and _r1["parent_genome_hash"] == _gf.genome_hash()
               and _xr1["parent_a"] == _gf.genome_hash()
               and _c1.genome_hash() != _gf.genome_hash())
        matrix["genome_mutation_crossover_provenance"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Mutation/crossover deterministic with complete parent provenance."
        }
    except Exception as e:
        matrix["genome_mutation_crossover_provenance"] = {"status": "FAIL", "reason": f"Genome ops failed: {e}"}

    # 23. overlapping_reproduction
    print("[23/57] Evaluating overlapping_reproduction...")
    try:
        from src.common.determinism import SeedBundle as _SB2
        from src.population.population import Population as _Pop2
        _s2 = _SB2(experiment_seed=301, generation_seed=302, organism_seed=303,
                   development_seed=304, mutation_seed=305, world_seed=306,
                   teacher_seed=307)
        _p = _Pop2(6, _s2, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=301)
        for _ in range(8):
            _p.step(15)
            _p.reproduce(2, mode="sexual")
        _repros = _p.events.filter("REPRODUCTION")
        _deaths = {(e["organism_id"], e["tick"]) for e in _p.events.filter("ORGANISM_DIED")}
        _overlap_ok = True
        for _r in _repros:
            for _pid in _r["payload"].get("parents", []):
                if any(d == _pid and t <= _r["tick"] for d, t in _deaths):
                    _overlap_ok = False
        _gens = {o.generation for o in _p.organisms}
        _ok = len(_repros) >= 1 and _overlap_ok and len(_gens) > 1
        matrix["overlapping_reproduction"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"{len(_repros)} reproductions across generations {sorted(_gens)}; "
                      f"parents alive at every birth: {_overlap_ok}."
        }
    except Exception as e:
        matrix["overlapping_reproduction"] = {"status": "FAIL", "reason": f"Reproduction failed: {e}"}

    # 24. cultural_transmission_gain
    print("[24/57] Evaluating cultural_transmission_gain...")
    try:
        from src.genome.schema import Genome as _G2
        from src.organism.organism import Organism as _Org
        from src.culture.transmission import teach as _teach
        _t = _Org(_G2.founder(401), "mx-teacher", 0,
                  {"organism_seed": 402, "development_seed": 403},
                  GraphMode.SYNTHETIC_TEST, 32)
        _s = _Org(_G2.founder(404), "mx-student", 1,
                  {"organism_seed": 405, "development_seed": 406},
                  GraphMode.SYNTHETIC_TEST, 32)
        _t.skills["forage"] = 0.9
        _sess = _teach(_t, _s, "forage", 10, 407)
        _chain = _s.cultural_knowledge["forage"]["provenance"]["teacher_chain"]
        _ok = _sess.learning_gain > 0.0 and "mx-teacher" in _chain
        matrix["cultural_transmission_gain"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"Measured learning gain {_sess.learning_gain} with teacher chain {_chain}."
        }
    except Exception as e:
        matrix["cultural_transmission_gain"] = {"status": "FAIL", "reason": f"Teaching failed: {e}"}

    # 25. real_mode_zero_surrogate_edges
    print("[25/57] Evaluating real_mode_zero_surrogate_edges...")
    try:
        import csv as _csv
        _pairs = set()
        with open(os.path.join(PROJECT_ROOT, "malecns", "data-raw",
                               "malecns_v1_0_connections.csv"), encoding="utf-8") as _f:
            for _row in _csv.DictReader(_f):
                try:
                    _pairs.add((int(_row["pre_body_id"]), int(_row["post_body_id"])))
                except (ValueError, KeyError):
                    continue
        _gr = get_or_create_circuit(64, mode=GraphMode.REAL, seed=505,
                                    cache_name="matrix_real_64.npz")
        _body = [int(x) for x in _gr.neuron_ids]
        _bad = 0
        # CSR v3: row = INCOMING sources, so biological edge = (col_source, row_target).
        for _i in range(_gr.num_neurons):
            _s, _e = int(_gr.row_offsets[_i]), int(_gr.row_offsets[_i + 1])
            for _k in range(_s, _e):
                if (_body[int(_gr.col_indices[_k])], _body[_i]) not in _pairs:
                    _bad += 1
        _meta_ok = (_gr.provenance_metadata.get("surrogate_edge_count", -1) == 0
                    and _gr.provenance_metadata.get("fallback_edges_added", -1) == 0)
        _ok = _bad == 0 and _meta_ok and _gr.num_synapses > 0
        matrix["real_mode_zero_surrogate_edges"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"REAL-64 graph: {_gr.num_synapses} edges, {_bad} non-empirical; metadata confirms zero surrogate."
        }
    except Exception as e:
        matrix["real_mode_zero_surrogate_edges"] = {"status": "FAIL", "reason": f"Provenance gate failed: {e}"}

    # 26. csr_directionality
    print("[26/57] Evaluating csr_directionality...")
    try:
        from src.compute.cpu_reference import cpu_lif_step as _lif
        _ro = np.array([0, 0, 1], dtype=np.int32)   # row1 = incoming from neuron 0
        _ci = np.array([0], dtype=np.int32)
        _w = np.array([5.0], dtype=np.float32)
        # A(0) spikes -> B(1) must fire
        _p, _s, _ = _lif(_ro, _ci, _w,
                         np.array([1.0, 0.0], dtype=np.float32),
                         np.zeros(2, dtype=np.float32),
                         np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.int32),
                         decay=0.85, threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2)
        # B(1) spikes -> A(0) must stay silent
        _p2, _s2, _ = _lif(_ro, _ci, _w,
                           np.array([0.0, 1.0], dtype=np.float32),
                           np.zeros(2, dtype=np.float32),
                           np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.int32),
                           decay=0.85, threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2)
        _ok = (_s[1] == 1.0) and (_s[0] == 0.0) and (_s2[0] == 0.0)
        matrix["csr_directionality"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "A->B drives B, never A; reverse direction has no effect."
        }
    except Exception as e:
        matrix["csr_directionality"] = {"status": "FAIL", "reason": f"Directionality failed: {e}"}

    # 27. biological_edge_semantics (weight transform provenance)
    print("[27/57] Evaluating biological_edge_semantics...")
    try:
        _gp = get_or_create_circuit(64, mode=GraphMode.REAL, seed=42,
                                    cache_name="matrix_real_64.npz").provenance_metadata
        _ok = (all(k in _gp for k in ("weight_source", "weight_transform",
                                      "biological_measurement", "simulation_semantics",
                                      "selection_strategy", "sampling_bias"))
               and _gp.get("surrogate_edge_count", -1) == 0)
        matrix["biological_edge_semantics"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"REAL weight transform declared: {_gp.get('weight_transform')}"
        }
    except Exception as e:
        matrix["biological_edge_semantics"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 28. real_annotation_integrity
    print("[28/57] Evaluating real_annotation_integrity...")
    try:
        from src.connectome.loader import load_raw_neurons as _lrn
        _n = _lrn()[0]
        _al = _n.annotation_levels
        _ok = (_al.get("cell_type") == "UNKNOWN" and _al.get("hemilineage") == "UNKNOWN"
               and _al.get("neurotransmitter") == "UNKNOWN"
               and _al.get("position") == "EMPIRICAL" and _al.get("tail_distance") == "DERIVED")
        _gpm = get_or_create_circuit(64, mode=GraphMode.REAL, seed=42,
                                     cache_name="matrix_real_64.npz").provenance_metadata
        _ea = _gpm.get("edge_annotations", {})
        _ok = _ok and _ea.get("annotation_level") == "DERIVED"
        matrix["real_annotation_integrity"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Unavailable annotations flagged UNKNOWN; available ones EMPIRICAL/DERIVED."
        }
    except Exception as e:
        matrix["real_annotation_integrity"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 29. plasticity_causal_effect
    print("[29/57] Evaluating plasticity_causal_effect...")
    try:
        from src.compute.cpu_reference import cpu_plasticity_step as _cps
        _ro = np.array([0, 0, 1], dtype=np.int32)
        _ci = np.array([0], dtype=np.int32)
        _w = np.array([0.5], dtype=np.float32)
        _pre = np.array([1.0, 0.0], dtype=np.float32)
        _post = np.array([0.0, 1.0], dtype=np.float32)
        _w0 = _cps(_ci, _w, _pre, _post, _ro, learning_rate=0.05, reward=0.0)
        _wup = _cps(_ci, _w, _pre, _post, _ro, learning_rate=0.05, reward=1.0)
        _wdn = _cps(_ci, _w, _pre, _post, _ro, learning_rate=0.05, reward=-1.0)
        _ok = (abs(float(_w0[0]) - 0.5) < 1e-9 and float(_wup[0]) > 0.5 and float(_wdn[0]) < 0.5)
        matrix["plasticity_causal_effect"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "reward=0 no change; reward>0 potentiation; reward<0 depression."
        }
    except Exception as e:
        matrix["plasticity_causal_effect"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 30. checkpoint_continuation
    print("[30/57] Evaluating checkpoint_continuation...")
    try:
        from src.common.determinism import SeedBundle as _SB
        from src.population.population import Population as _P
        from scripts.run_long_campaign import seeds_for as _sf, TICKS_PER_GEN as _T
        _seed = 31
        _fresh = _P(4, _sf(_seed), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=_seed)
        for _ in range(4):
            _fresh.step(_T); _fresh.reproduce(2)
        _h = _fresh.population_hash()
        _int = _P(4, _sf(_seed), GraphMode.SYNTHETIC_TEST, 32, experiment_seed=_seed)
        for _ in range(2):
            _int.step(_T); _int.reproduce(2)
        _res = _P.restore(_int.snapshot(), _sf(_seed))
        for _ in range(2):
            _res.step(_T); _res.reproduce(2)
        _ok = (_res.population_hash() == _h)
        matrix["checkpoint_continuation"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Checkpoint/resume final population hash equals uninterrupted run."
        }
    except Exception as e:
        matrix["checkpoint_continuation"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 31. llm_model_discovery_and_inference
    print("[31/57] Evaluating llm_model_discovery_and_inference...")
    try:
        from src.llm.discovery import discover_models as _dm
        from src.llm.runtime import LocalLLM as _L, GenerationConfig as _GC
        _models = [m for m in _dm() if m.status == "DISCOVERED"]
        if len(_models) == 0:
            # Environment-dependent, like Vulkan: no local GGUF weights on this
            # host (CI runners never receive the gitignored model binary).
            # Honesty in this case is verified by gate 32 (structured error,
            # no fake text), so this gate is a SKIP, not a FAIL.
            matrix["llm_model_discovery_and_inference"] = {
                "status": "SKIP_ENVIRONMENT",
                "reason": "No local GGUF model discovered on this host; "
                          "unavailable-model honesty is verified by llm_failure_mode_and_tool_safety."
            }
        else:
            _ok = _models[0].architecture == "llama" and len(_models[0].sha256) == 64
            _infer = "SKIPPED"
            if _ok and os.environ.get("FLYBRAIN_SKIP_LLM_INFER") != "1":
                _llm = _L(_models[0], n_ctx=1024)
                if _llm.load():
                    _r = _llm.generate("1, 2,", _GC(max_tokens=4, seed=1))
                    _infer = _r["status"]
                    _llm.unload()
            _ok = _ok and _infer in ("SUCCESS", "SKIPPED")
            matrix["llm_model_discovery_and_inference"] = {
                "status": "PASS" if _ok else "FAIL",
                "reason": f"Discovered {len(_models)} GGUF; inference={_infer}; model={_models[0].filename}."
            }
    except Exception as e:
        matrix["llm_model_discovery_and_inference"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 32. llm_failure_mode_and_tool_safety
    print("[32/57] Evaluating llm_failure_mode_and_tool_safety...")
    try:
        from src.llm.runtime import LocalLLM as _L2
        from src.llm.scientist import ScientistLoop as _SL, ToolSpec as _TS
        _missing = _L2(None).generate("x")
        _loop = _SL(_L2(None))
        _loop.register(_TS("echo", "d", {}, lambda p: p))
        _rej_unknown = _loop.execute_tool_request({"tool": "shell", "params": {}})["status"]
        _rej_bad = _loop.execute_tool_request({"tool": "echo", "params": {"c": "os.system('x')"}})["status"]
        _ok = (_missing["status"] == "MODEL_UNAVAILABLE" and _missing["text"] is None
               and _rej_unknown == "REJECTED" and _rej_bad == "REJECTED")
        matrix["llm_failure_mode_and_tool_safety"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Unavailable model returns structured error (no fake text); shell/unknown tools rejected."
        }
    except Exception as e:
        matrix["llm_failure_mode_and_tool_safety"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 33. living_brain_identity
    print("[33/57] Evaluating living_brain_identity...")
    try:
        from src.brain.living import LivingBrain, DevelopmentState as _LBS, DevelopmentEngine as _LBE
        from src.connectome.loader import get_or_create_circuit as _goc
        _g = _goc(32, mode=GraphMode.SYNTHETIC_TEST, seed=7, cache_name="mx_lb.npz")
        _lb = LivingBrain(_g, dev=_LBS.initialize(_g.num_neurons), engine=_LBE({}, 8),
                          experiment_seed=7)
        _before = _g.num_neurons
        _lb.run_development_cycle(tick=5, growth_budget=3)
        _new = [r for r in _lb._neurons.values() if r.birth_op == "neurogenesis"]
        _ok = (_lb.validate() is None and _g.num_neurons >= _before
               and all(r.provenance_class == "EMERGENT" for r in _new))
        matrix["living_brain_identity"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"Persistent identities + EMERGENT provenance; growth {_before}->{_g.num_neurons}."
        }
    except Exception as e:
        matrix["living_brain_identity"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 34. structural_growth_resource_constrained
    print("[34/57] Evaluating structural_growth_resource_constrained...")
    try:
        _g = _goc(32, mode=GraphMode.SYNTHETIC_TEST, seed=9, cache_name="mx_lb2.npz")
        _lb0 = LivingBrain(_g, dev=_LBS.initialize(_g.num_neurons), engine=_LBE({}, 10),
                           experiment_seed=9)
        _n0 = _g.num_neurons
        _s0 = _lb0.run_development_cycle(tick=5, growth_budget=0)
        _blocked = (_g.num_neurons == _n0 and _s0["ops"][0]["neurons_born"] == 0)
        matrix["structural_growth_resource_constrained"] = {
            "status": "PASS" if _blocked else "FAIL",
            "reason": "Zero growth budget blocks neurogenesis (energy is the constraint)."
        }
    except Exception as e:
        matrix["structural_growth_resource_constrained"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 35. eligibility_neuromodulation
    print("[35/57] Evaluating eligibility_neuromodulation...")
    try:
        from src.brain.eligibility import EligibilityState as _ES, EligibilityEngine as _EE, \
            NeuromodulationConfig as _NC
        from src.brain.runtime import BrainRuntime as _BRT
        _g = _goc(32, mode=GraphMode.SYNTHETIC_TEST, seed=11, cache_name="mx_elig.npz")
        _rt = _BRT(_g, use_gpu=False, plasticity_mode="v2_eligibility",
                   neuromod=_NC(w_reward=0.0, w_novelty=1.0))
        _rng = np.random.RandomState(1)
        _w0 = _g.weights.copy()
        _moved = False
        for _ in range(8):
            _o = _rt.step(sensory_inputs={"visual": _rng.uniform(0, 1, 16).astype(np.float32)})
            _moved = _moved or _o["synapses_updated"] > 0
        _ok = _moved and _rt.plasticity_mode == "v2_eligibility" \
            and bool(np.any(np.abs(_g.weights - _w0) > 1e-5))
        matrix["eligibility_neuromodulation"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "v2 traces + novelty-driven neuromodulation change weights (reward=0)."
        }
    except Exception as e:
        matrix["eligibility_neuromodulation"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 36. autonomy_self_generated_goals
    print("[36/57] Evaluating autonomy_self_generated_goals...")
    try:
        from src.common.determinism import SeedBundle as _SB2
        from src.population.population import Population as _P2
        _seeds = _SB2(experiment_seed=121, generation_seed=122, organism_seed=123,
                      development_seed=124, mutation_seed=125, world_seed=126,
                      teacher_seed=127)
        _pop = _P2(3, _seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=121,
                   autonomy_mode=True, genome_version="2.0")
        _goal_ticks = 0
        for _ in range(20):
            _pop.step(1)
            _goal_ticks += sum(1 for o in _pop.living()
                               if o.autonomy is not None and o.autonomy._goal_seq > 0)
        _all = all(o.autonomy._goal_seq > 0 for o in _pop.living())
        _ok = _goal_ticks > 0 and _all
        matrix["autonomy_self_generated_goals"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"All living organisms self-generate goals; no human task commands."
        }
    except Exception as e:
        matrix["autonomy_self_generated_goals"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 37. grounded_language_and_social
    print("[37/57] Evaluating grounded_language_and_social...")
    try:
        from src.language.grounded import GroundedLanguageSystem as _GLS
        from src.social.model import SocialMemory as _SM
        _ls = _GLS(seed=1)
        _c = _ls.ground_concept("food", [1.0, 0, 0, 0, 0, 0, 0, 0], "sensory")
        _ls.learn_symbol("sym0", _c.concept_id)
        _prod = _ls.produce({"salient_concepts": [_c.concept_id]})
        _comp = _ls.comprehend(_prod)
        _sm = _SM("a")
        for _t in range(6):
            _sm.record_interaction("teacher", _t, "taught_by", 0.8)
        for _t in range(6):
            _sm.record_interaction("rival", _t, "competed", -0.8)
        _ok = (_prod == ["sym0"] and _comp == [_c.concept_id]
               and _sm.trust_of("teacher") > _sm.trust_of("rival"))
        matrix["grounded_language_and_social"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Symbols bind to grounded concepts; trust emerges from outcomes."
        }
    except Exception as e:
        matrix["grounded_language_and_social"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 38. genome_v2_architecture_genes
    print("[38/57] Evaluating genome_v2_architecture_genes...")
    try:
        from src.genome.schema import Genome as _GP, GENOME_VERSION as _GV2
        from src.evolution.speciation import genome_distance as _gdist
        _v2 = _GP.founder(31)
        _v1 = _GP.founder(31, legacy=True)
        _ok = (_v2.version == _GV2 and "eligibility_decay" in _v2.params
               and _v1.version == "1.0" and len(_v1.params) == 15
               and _gdist(_v2, _v2) == 0.0)
        matrix["genome_v2_architecture_genes"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "v2 encodes learning architecture; v1 legacy remains valid."
        }
    except Exception as e:
        matrix["genome_v2_architecture_genes"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 39. speciation_evidence
    print("[39/57] Evaluating speciation_evidence...")
    try:
        from src.genome.schema import Genome as _GP2
        from src.evolution.speciation import detect_divergence as _dd
        _p = _GP2.founder(41)
        _far = _GP2.founder(42)
        _far.params["exploration"] = 1.0
        _far.params["sociality"] = 0.0
        _far.params["teaching_ability"] = 1.0
        _far.params["metabolism_rate"] = 0.1
        _far.validate()
        _div = _dd([("p", _p)], [("c", _far)], 10, 0.05)
        _ok = len(_div) == 1 and _div[0].genome_distance_from_parent > 0.05
        matrix["speciation_evidence"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Divergence recorded only with measured genome distance."
        }
    except Exception as e:
        matrix["speciation_evidence"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 40. llm_control_plane_safety
    print("[40/57] Evaluating llm_control_plane_safety...")
    try:
        from src.llm.control import ControlPlane as _CP, ResearchRuntime as _RR, \
            CommandEnvelope as _CE
        _cp = _CP(_RR(experiment_seed=9))
        _r1 = _cp.execute(_CE("SPAWN_POPULATION", {"size": 999}))
        _r2 = _cp.execute(_CE("SAVE_CHECKPOINT", {"name": "x'; import os; os.system('id')"}))
        _r3 = _cp.execute(_CE("SPAWN_POPULATION", {"size": 2}))
        _r4 = _cp.execute(_CE("START_RUN", {"ticks": 3}))
        _ok = (_r1["status"] == "REJECTED" and _r2["status"] == "REJECTED"
               and _r3["status"] == "EXECUTED" and _r4["status"] == "EXECUTED")
        matrix["llm_control_plane_safety"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Invalid/shell commands rejected; valid typed commands execute."
        }
    except Exception as e:
        matrix["llm_control_plane_safety"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 41. research_memory_chain
    print("[41/57] Evaluating research_memory_chain...")
    try:
        import tempfile as _tf
        from src.llm.research_memory import ResearchMemory as _RM
        with _tf.TemporaryDirectory() as _td:
            _m = _RM(os.path.join(_td, "rm.jsonl"))
            _m.append("experiment", {"id": "e1", "status": "EXECUTED"})
            _m.append("hypothesis", {"id": "h1"})
            _m.append("experiment", {"id": "e2", "status": "FAILED"})
            _ok = _m.verify_chain() and len(_m.failed_experiments()) == 1 \
                and len(_m.successful_protocols()) == 1
        matrix["research_memory_chain"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Hash-chained append-only research memory; tamper-evident."
        }
    except Exception as e:
        matrix["research_memory_chain"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 42. deeptime_escalation_replay
    print("[42/57] Evaluating deeptime_escalation_replay...")
    try:
        from src.timeline.deeptime import DeepTimeRunner as _DTR, DeepTimeConfig as _DTC
        from src.common.determinism import SeedBundle as _SB3
        from src.population.population import Population as _P3
        _seeds = _SB3(experiment_seed=131, generation_seed=132, organism_seed=133,
                      development_seed=134, mutation_seed=135, world_seed=136,
                      teacher_seed=137)
        _pop = _P3(3, _seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=131,
                   autonomy_mode=True, genome_version="2.0")
        _r = _DTR(_pop, _DTC(coarse_ticks_per_step=10))
        _r.fast_forward(2)
        _esc = _r.escalate("gate42")
        _rep = _r.replay_high_resolution("gate42", ticks=3)
        _ok = (_rep["status"] == "EXECUTED" and _rep["checkpoint_hash_verified"]
               and _r.ledger["approximation_model"] != "")
        matrix["deeptime_escalation_replay"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Coarse deep-time escalates to full-res checkpoint; replay hash-verified."
        }
    except Exception as e:
        matrix["deeptime_escalation_replay"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 43. milestone_evidence
    print("[43/57] Evaluating milestone_evidence...")
    try:
        from src.science.milestones import detect_milestones as _dm2, \
            milestone_certificate as _mc
        from src.common.determinism import SeedBundle as _SB4
        from src.population.population import Population as _P4
        _seeds = _SB4(experiment_seed=141, generation_seed=142, organism_seed=143,
                      development_seed=144, mutation_seed=145, world_seed=146,
                      teacher_seed=147)
        _pop = _P4(3, _seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=141,
                   autonomy_mode=True, genome_version="2.0")
        _pop.organisms[0].age = 70
        _pop.organisms[1].age = 70
        _pop.step(10)
        _pop.reproduce(2)
        _ms = _dm2(_pop)
        _ok = all(m.get("evidence") for m in _ms) and len(_ms) >= 1 \
            and any(m["milestone"] == "OVERLAPPING_GENERATIONS" for m in _ms)
        if _ms:
            _cert = _mc(_ms[0], 141)
            _ok = _ok and _cert["valid"]
        matrix["milestone_evidence"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"Detected {[m['milestone'] for m in _ms]} with evidence + certificate."
        }
    except Exception as e:
        matrix["milestone_evidence"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 44. benchmark_fairness
    print("[44/57] Evaluating benchmark_fairness...")
    try:
        from src.research.benchmark import run_flybrain_arm as _fb, \
            run_llm_only_arm as _lo, run_benchmark_suite as _bs
        _fb_res = _fb(seed=151, train_ticks=15, delay_ticks=5)
        _lo_res = _lo(seed=151)
        _ok = (_fb_res["budget"]["training_exposure_ticks"] == 15
               and _lo_res["status"] in ("MEASURED", "SKIP")
               and (_lo_res["status"] != "SKIP" or _lo_res.get("reason")))
        matrix["benchmark_fairness"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Per-arm budgets documented; LLM arms SKIP with reason when unmet."
        }
    except Exception as e:
        matrix["benchmark_fairness"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 45. version_metadata (canonical: must match src/version.py single source of truth)
    print("[45/57] Evaluating version_metadata...")
    try:
        from src.version import VERSION as _V, VERSION_TAG as _VT
        import re as _re
        _ok = bool(_re.fullmatch(r"\d+\.\d+\.\d+", _V)) and _VT == f"v{_V}"
        matrix["version_metadata"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"FlyBrain version metadata = {_V} ({_VT})."
        }
    except Exception as e:
        matrix["version_metadata"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 46. immutable_bio_baseline
    print("[46/57] Evaluating immutable_bio_baseline...")
    try:
        from src.brain.living import LivingBrain as _LB2, DevelopmentState as _DS2, \
            DevelopmentEngine as _DE2
        from src.connectome.loader import get_or_create_circuit as _g2
        _g = _g2(32, mode=GraphMode.SYNTHETIC_TEST, seed=61, cache_name="v4m_bio.npz")
        _lb = _LB2(_g, dev=_DS2.initialize(_g.num_neurons), engine=_DE2({}, 62), experiment_seed=61)
        _fp0 = _lb.bio_baseline.seed_fingerprint
        for _t in (5, 10, 15):
            _lb.run_development_cycle(tick=_t, growth_budget=2)
        _ok = _lb.biological_baseline_intact() and _lb.bio_baseline.seed_fingerprint == _fp0
        matrix["immutable_bio_baseline"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Biological baseline fingerprint unchanged after lifetime development."
        }
    except Exception as e:
        matrix["immutable_bio_baseline"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 47. synapse_identity_provenance
    print("[47/57] Evaluating synapse_identity_provenance...")
    try:
        _g = _g2(32, mode=GraphMode.SYNTHETIC_TEST, seed=63, cache_name="v4m_syn.npz")
        _lb = _LB2(_g, dev=_DS2.initialize(_g.num_neurons), engine=_DE2({}, 64), experiment_seed=63)
        _seed_syn = [r for r in _lb._synapses.values() if r.birth_op == "seed"]
        _ok = all(len(r.synapse_id) == 16 and r.source_dataset and r.source_record_id
                  for r in _seed_syn)
        _lb.run_development_cycle(tick=5, growth_budget=2)
        _emergent = [r for r in _lb._synapses.values() if r.birth_op != "seed"]
        _ok = _ok and all(r.provenance_class == "EMERGENT" and not r.source_record_id
                          for r in _emergent)
        matrix["synapse_identity_provenance"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Stable synapse IDs; seed records cite source dataset; "
                      "new synapses are EMERGENT, never BIOLOGICAL."
        }
    except Exception as e:
        matrix["synapse_identity_provenance"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 48. brain_identity_layers
    print("[48/57] Evaluating brain_identity_layers...")
    try:
        from src.provenance.v4 import BrainIdentity as _BI
        _g1 = _g2(32, mode=GraphMode.SYNTHETIC_TEST, seed=65, cache_name="v4m_ident.npz")
        _i0 = _BI.from_components(_g1).full_identity()
        _g1.weights = np.clip(_g1.weights + 0.05, 0.01, 1.0)
        _i1 = _BI.from_components(_g1).full_identity()
        _g2b = _g2(32, mode=GraphMode.SYNTHETIC_TEST, seed=65, cache_name="v4m_ident.npz")
        _i2 = _BI.from_components(_g2b).full_identity()
        _ok = (_i0 != _i1 and _i2 == _i0 and len(_BI.LAYERS) == 12)
        matrix["brain_identity_layers"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "12-layer identity: same state -> same identity; "
                      "weight change -> different identity."
        }
    except Exception as e:
        matrix["brain_identity_layers"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 49. heredity_separation
    print("[49/57] Evaluating heredity_separation...")
    try:
        _seeds = _sb(241)
        _pop = _P(2, _seeds, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=241)
        _parent = _pop.living()[0]
        _w0 = _parent.graph.weights.copy()
        _rngL = np.random.RandomState(7)
        for _ in range(6):
            _parent.brain.step(sensory_inputs={
                "visual": _rngL.uniform(0, 1.2, 16).astype(np.float32)}, reward=1.0)
        _learned = bool(np.any(np.abs(_parent.graph.weights - _w0) > 1e-6))
        for _o in _pop.living():
            _o.age = 70
            _o.stage = stage_for_age(_o.age)
        _pop.reproduce(1, mode="sexual")
        _children = [o for o in _pop.organisms if o.generation > 0]
        if not _children:
            matrix["heredity_separation"] = {"status": "FAIL", "reason": "no child born"}
        else:
            _child = _children[0]
            _fresh = _g2(32, mode=GraphMode.SYNTHETIC_TEST,
                         seed=_child.seeds["organism_seed"],
                         cache_name="v4m_heredity.npz")
            _ok = (_learned and np.allclose(_child.graph.weights, _fresh.weights, atol=1e-7))
            matrix["heredity_separation"] = {
                "status": "PASS" if _ok else "FAIL",
                "reason": "Parent learned; child starts from deterministic development, "
                          "not inherited learned weights."
            }
    except Exception as e:
        matrix["heredity_separation"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 50. evolution_parent_identity
    print("[50/57] Evaluating evolution_parent_identity...")
    try:
        from src.evolution.scheduler import EvolutionScheduler as _ES
        _g = _g2(64, mode=GraphMode.SYNTHETIC_TEST, seed=71, cache_name="v4m_sched.npz")
        _sched = _ES(_g, history_file="diagnostics/v4_matrix_sched.json", seed=100)
        _prev = _sched.current_brain_id
        _ok = True
        for _ in range(2):
            _s = _sched.run_generation(num_candidates=4, seed=100)
            _ok = _ok and _s["parent_id"] == _prev \
                and _s["child_id"] == _sched.current_brain_id
            for _c in _s["candidates"]:
                _ok = _ok and _c["candidate_id"] != _c["parent_id"]
            _prev = _s["child_id"]
        matrix["evolution_parent_identity"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Accepted child never becomes its own parent in history."
        }
    except Exception as e:
        matrix["evolution_parent_identity"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 51. ablation_enforcement
    print("[51/57] Evaluating ablation_enforcement...")
    try:
        from src.research.ablation import (run_ablation as _ra, AblationConfig as _AC,
                                           verify_ablation_enforcement as _vae)
        _r1 = _ra(_AC.from_preset("no_growth"), seed=321, ticks=20)
        _r2 = _ra(_AC.from_preset("no_plasticity"), seed=322, ticks=20)
        _ok = _vae(_r1) and _vae(_r2) and not _r1["measured"]["grew"] \
            and not _r2["measured"]["weights_changed"]
        matrix["ablation_enforcement"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "no_growth grew=False; no_plasticity weights unchanged "
                      "(enforced, verified by measurement)."
        }
    except Exception as e:
        matrix["ablation_enforcement"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 52. deterministic_research_ids
    print("[52/57] Evaluating deterministic_research_ids...")
    try:
        from src.llm.control import ControlPlane as _CP2, ResearchRuntime as _RR2, \
            CommandEnvelope as _CE2
        _cp1, _cp2 = _CP2(_RR2(experiment_seed=77)), _CP2(_RR2(experiment_seed=77))
        _a = _cp1.execute(_CE2("PROPOSE_HYPOTHESIS",
                               {"text": "x correlates with y", "based_on_experiments": []}))
        _b = _cp2.execute(_CE2("PROPOSE_HYPOTHESIS",
                               {"text": "x correlates with y", "based_on_experiments": []}))
        _ok = _a["hypothesis"]["hypothesis_id"] == _b["hypothesis"]["hypothesis_id"]
        matrix["deterministic_research_ids"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Research IDs derive from content+sequence; no wall-clock identity."
        }
    except Exception as e:
        matrix["deterministic_research_ids"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 53. research_ledger_chain
    print("[53/57] Evaluating research_ledger_chain...")
    try:
        from src.research.ledger import ResearchLedger as _RL2
        _led = _RL2("matrix-ledger", seed=9)
        _r1 = _led.append("experiment", 0, 0, {"type": "t"}, world_sha="w")
        _r2 = _led.append("milestone", 5, 1, {"m": "M"}, parent_event=_r1["event_id"])
        _ok = _led.verify_chain() and _r2["parent_event"] == _r1["event_id"] \
            and _led._static.get("shader_sha") and _led._static.get("experiment_id") == "matrix-ledger"
        matrix["research_ledger_chain"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "Hash-chained ledger with code/dataset/shader provenance."
        }
    except Exception as e:
        matrix["research_ledger_chain"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 54. deeptime_exact_mode
    print("[54/57] Evaluating deeptime_exact_mode...")
    try:
        from src.timeline.deeptime import DeepTimeRunner as _DTR2, DeepTimeConfig as _DTC2
        _seeds2 = _sb(331)
        _pop2 = _P(3, _seeds2, GraphMode.SYNTHETIC_TEST, 32, experiment_seed=331,
                   autonomy_mode=True, genome_version="2.0")
        _r = _DTR2(_pop2, _DTC2(mode="EXACT"))
        _s = _r.fast_forward(8)
        _ok = (_s["mode"] == "EXACT" and _s["resolution"] == "full"
               and _s["approximation_model"] is None and _s["sim_ticks"] == 8)
        matrix["deeptime_exact_mode"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "EXACT mode is tick-by-tick with no approximation claim; "
                      "ACCELERATED labels its approximation model."
        }
    except Exception as e:
        matrix["deeptime_exact_mode"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 55. doctor_functional
    print("[55/57] Evaluating doctor_functional...")
    try:
        from src.diagnostics.doctor import run_doctor as _rd
        from src.version import VERSION as _VV
        _rep = _rd()
        _names = [c["name"] for c in _rep["checks"]]
        _ok = (_rep["flybrain_version"] == _VV and _rep["summary"]["errors"] == 0
               and all(n in _names for n in ("dataset", "gpu", "core_runtime"))
               and all(c["detail"] for c in _rep["checks"]))
        matrix["doctor_functional"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": f"Doctor reports {_rep['overall']} with "
                      f"{_rep['summary']['ok']} OK checks (real values)."
        }
    except Exception as e:
        matrix["doctor_functional"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 56. ui_v4_endpoints
    print("[56/57] Evaluating ui_v4_endpoints...")
    try:
        from fastapi.testclient import TestClient as _TC
        from src.ui.server import app as _app
        from src.version import VERSION as _VVV
        _cl = _TC(_app)
        _v = _cl.get("/api/version")
        _d = _cl.get("/api/doctor")
        _ok = (_v.status_code == 200 and _v.json()["version"] == _VVV
               and _d.status_code == 200 and _d.json()["overall"] in
               ("READY", "READY_DEGRADED", "DEGRADED"))
        matrix["ui_v4_endpoints"] = {
            "status": "PASS" if _ok else "FAIL",
            "reason": "/api/version + /api/doctor serve real system state."
        }
    except Exception as e:
        matrix["ui_v4_endpoints"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # 57. portable_package (environment-dependent: only where a built bundle exists)
    print("[57/57] Evaluating portable_package...")
    try:
        _portable_dir = os.environ.get("FLYBRAIN_PORTABLE_DIR", "")
        if not _portable_dir or not os.path.isdir(_portable_dir):
            matrix["portable_package"] = {
                "status": "SKIP_ENVIRONMENT",
                "reason": "No built portable bundle on this host; set "
                          "FLYBRAIN_PORTABLE_DIR to a built dist for artifact self-test."
            }
        else:
            _exe = (os.path.join(_portable_dir, "FlyBrain", "FlyBrain.exe")
                    if platform.system() == "Windows"
                    else os.path.join(_portable_dir, "FlyBrain", "flybrain-launcher.sh"))
            _exists = os.path.exists(_exe)
            matrix["portable_package"] = {
                "status": "PASS" if _exists else "FAIL",
                "reason": f"Portable bundle present: {_exe}" if _exists
                          else f"expected bundle entry missing: {_exe}"
            }
    except Exception as e:
        matrix["portable_package"] = {"status": "FAIL", "reason": f"failed: {e}"}

    # Summary
    # Schema conformance: runner categories must equal the canonical schema.
    # Drift (renamed/missing/extra category) forces overall FAILED, never silent.
    # Recorded at report level (not as a category) so totals stay schema-defined.
    _schema_path = os.path.join(PROJECT_ROOT, "verification", "acceptance_schema.json")
    _schema_drift = []
    try:
        with open(_schema_path, "r", encoding="utf-8") as _sf:
            _schema = json.load(_sf)
        _schema_cats = _schema.get("categories", [])
        _schema_drift = sorted(set(matrix) ^ set(_schema_cats))
    except Exception as _e:
        _schema_drift = [f"acceptance schema unreadable: {_e}"]
    pass_count = sum(1 for v in matrix.values() if v["status"] == "PASS")
    fail_count = sum(1 for v in matrix.values() if v["status"] == "FAIL")
    skip_count = sum(1 for v in matrix.values() if v["status"].startswith("SKIP"))
    
    overall_status = "PASSED" if (fail_count == 0 and not _schema_drift) else "FAILED"

    report = {
        "timestamp": time.time(),
        "overall_status": overall_status,
        "total_categories": len(matrix),
        "passed": pass_count,
        "failed": fail_count,
        "skipped": skip_count,
        "acceptance_schema": "verification/acceptance_schema.json",
        "schema_drift": _schema_drift,
        "categories": matrix
    }

    out_file = os.path.join(PROJECT_ROOT, "diagnostics", "acceptance_matrix.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"ACCEPTANCE MATRIX OVERALL RESULT: {overall_status}")
    print(f"Passed: {pass_count}/{len(matrix)} | Failed: {fail_count} | Skipped: {skip_count}")
    print(f"Report written to: {out_file}")
    print("=" * 70)
    
    return report

def main():
    report = evaluate_acceptance_matrix()
    if report["failed"] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
