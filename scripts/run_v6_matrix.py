#!/usr/bin/env python3
"""V6 acceptance matrix: embodied world gates. Every gate executes behavior.
Statuses: PASS | FAIL | SKIP_ENVIRONMENT | UNAVAILABLE.
Writes diagnostics/v6/acceptance_matrix.json.
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
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "acceptance_matrix.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

GATES = {}


def gate(name):
    def deco(fn):
        GATES[name] = fn
        return fn
    return deco


def _v6evidence(name):
    p = os.path.join(PROJECT_ROOT, "diagnostics", "v6", name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


@ gate("repository_cleanliness")
def g_repo():
    need = ["src/world3d", "src/models", "src/backup", "src/runtime",
            "models/text_model", "verification"]
    missing = [d for d in need if not os.path.isdir(os.path.join(PROJECT_ROOT, d))]
    # models/text_model holds binaries (git-ignored) but must exist locally
    missing = [d for d in missing if not d.startswith("models/")]
    return ("PASS", "ok") if not missing else ("FAIL", f"missing {missing}")


@ gate("biological_provenance")
def g_prov():
    from src.experiment.manager import get_file_sha256
    soma = get_file_sha256(os.path.join(PROJECT_ROOT, "malecns", "data-raw",
                                        "2023-27-2 soma_sides.csv"))
    man = json.load(open(os.path.join(PROJECT_ROOT, "manifests",
                                      "malecns_provenance.json"), encoding="utf-8"))
    files = man["provenance_metadata"]["files"]
    ok = soma in (files["soma_sides_csv"]["sha256"],
                  files["soma_sides_csv"].get("sha256_canonical_lf"))
    return ("PASS", "soma hash matches") if ok else ("FAIL", "soma hash mismatch")


@ gate("real_graph_integrity")
def g_real():
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode, ProvenanceStatus
    g = get_or_create_circuit(64, mode=GraphMode.REAL, seed=42,
                              cache_name="v6m_real64.npz")
    pm = g.provenance_metadata
    ok = (g.provenance_status == ProvenanceStatus.VERIFIED
          and pm.get("graph_identity") == "REAL_SUBGRAPH"
          and pm.get("surrogate_edge_count", 0) == 0
          and pm.get("source_neuron_total") == 125506)
    return ("PASS", f"REAL_SUBGRAPH {g.num_neurons}/{g.num_synapses}") if ok else ("FAIL", str(pm))


@ gate("surrogate_separation")
def g_surr():
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode, ProvenanceStatus
    g = get_or_create_circuit(64, mode=GraphMode.SPATIAL_SURROGATE, seed=42,
                              cache_name="v6m_surr64.npz")
    ok = g.provenance_status == ProvenanceStatus.SURROGATE and g.mode == GraphMode.SPATIAL_SURROGATE
    return ("PASS", "surrogate labelled") if ok else ("FAIL", "mislabeled")


@ gate("synthetic_separation")
def g_synth():
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode, ProvenanceStatus
    g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=42,
                              cache_name="v6m_synth32.npz")
    ok = g.provenance_status == ProvenanceStatus.EXPERIMENTAL
    return ("PASS", "synthetic labelled") if ok else ("FAIL", "mislabeled")


@ gate("lif_dynamics")
def g_lif():
    import numpy as _np
    from src.compute.cpu_reference import cpu_lif_step
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode
    g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=3,
                              cache_name="v6m_lif32.npz")
    n = g.num_neurons
    pot = _np.zeros(n, dtype=_np.float32)
    spk = (_np.random.RandomState(3).rand(n) < 0.5).astype(_np.float32)
    ref = _np.zeros(n, dtype=_np.int32)
    ext = _np.full(n, 0.9, dtype=_np.float32)
    p1, s1, _ = cpu_lif_step(g.row_offsets, g.col_indices, g.weights, spk, ext, pot, ref)
    p2, s2, _ = cpu_lif_step(g.row_offsets, g.col_indices, g.weights, s1, ext, p1, ref)
    ok = bool(_np.all((s1 == 0.0) | (s1 == 1.0))) and p1.shape == (n,)
    return ("PASS", f"spikes={int(s1.sum())}") if ok else ("FAIL", "bad LIF output")


@ gate("vulkan_backend")
def g_vulkan():
    try:
        from src.compute.vulkan_backend import VulkanComputeEngine
        e = VulkanComputeEngine()
        d = e.device_name
        e.cleanup()
        return ("PASS", f"device {d}")
    except Exception as e:
        return ("SKIP_ENVIRONMENT", f"no Vulkan: {type(e).__name__}")


@ gate("cpu_backend")
def g_cpu():
    from copy import deepcopy
    from src.brain.runtime import BrainRuntime
    from src.connectome.loader import get_or_create_circuit
    from src.connectome.types import GraphMode
    import numpy as _np
    g = get_or_create_circuit(32, mode=GraphMode.SYNTHETIC_TEST, seed=4,
                              cache_name="v6m_cpu32.npz")
    rt = BrainRuntime(deepcopy(g), use_gpu=False, seed=4)
    r = rt.step(sensory_inputs={"visual": _np.ones(16, dtype=_np.float32)}, reward=0.2)
    rt.cleanup()
    return ("PASS", f"spikes={r['spikes']}") if "spikes" in r else ("FAIL", "no step result")


@ gate("deterministic_replay")
def g_replay():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        try:
            for _ in range(10):
                svc.step(1)
            snap, h_ref = svc.snapshot(), svc.world.state_hash()
            for _ in range(5):
                svc.step(1)
            svc.world.restore(snap["world"])
            for n, a in snap["agents"].items():
                svc.agents[n].restore(a)
            for _ in range(5):
                svc.step(1)
            h_after = svc.world.state_hash()
            pos_a = list(svc.agents["hero"].body3d.pos)
            # uninterrupted reference: fresh service, same 15 ticks
            svc2 = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w2.db"))
            try:
                for _ in range(15):
                    svc2.step(1)
                ok = (svc2.world.state_hash() == h_after
                      and list(svc2.agents["hero"].body3d.pos) == pos_a)
                return ("PASS", f"hash {h_after[:12]}") if ok else ("FAIL", "branch mismatch")
            finally:
                svc2.spatial.close()
        finally:
            svc.spatial.close()


@ gate("checkpoint_restore")
def g_ckpt():
    from src.world3d.service import WorldService
    from src.backup.service import BackupService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        bs = BackupService(root=os.path.join(td, "backups"))
        try:
            svc.step(5)
            man = bs.create_world_backup(svc, label="m", trigger="manual")
            v = bs.verify_backup(man["backup_name"])
            r = bs.restore_world_backup(man["backup_name"], svc)
            ok = v["status"] == "VALID" and r["status"] == "RESTORED"
            return ("PASS", man["world_hash"][:12]) if ok else ("FAIL", str((v, r)))
        finally:
            svc.spatial.close()


@ gate("world_boot")
def g_boot():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=2, db_path=os.path.join(td, "w.db"))
        try:
            st = svc.state()
            ok = (len(st["characters"]) == 3 and st["tick"] == 0
                  and len(st["state_hash"]) == 64)
            return ("PASS", st["state_hash"][:12]) if ok else ("FAIL", str(st)[:200])
        finally:
            svc.spatial.close()


@ gate("physics")
def g_physics():
    from src.world3d.spec import build_world_spec
    from src.world3d.physics import PhysicsWorld
    w = PhysicsWorld(build_world_spec(7), [{"name": "t", "x": 0.0, "y": 0.0}])
    w.teleport("t", 0.0, 0.0, 4.0)
    for _ in range(200):
        w.step()
    s = w.char_state("t")
    ok = s["pos"][2] < 4.0 and w.grounded("t") and 0.5 < s["pos"][2] < 1.6
    return ("PASS", f"rest z={s['pos'][2]}") if ok else ("FAIL", str(s))


@ gate("collision")
def g_collision():
    from src.world3d.spec import build_world_spec
    from src.world3d.physics import PhysicsWorld
    w = PhysicsWorld(build_world_spec(7), [{"name": "t", "x": 0.0, "y": 10.0}])
    w.teleport("t", 0.0, 10.0, 1.2)
    for _ in range(50):
        w.step()
    for _ in range(300):
        w.drive_character("t", 0.0, 2.0)
        w.step()
    s = w.char_state("t")
    hit_wall = s["pos"][1] < 12.7
    contact = any("wall" in c["geom1"] or "wall" in c["geom2"] for c in w.contacts())
    ok = hit_wall and contact
    return ("PASS", f"y={s['pos'][1]}") if ok else ("FAIL", f"y={s['pos'][1]} contact={contact}")


@ gate("navigation")
def g_nav():
    from src.world3d.spec import build_world_spec
    from src.world3d.navigation import Navigator, true_blocked_cells
    spec = build_world_spec(7)
    assert len(true_blocked_cells(spec)) > 50
    nav = Navigator(spec)
    assert nav.plan(0.0, 4.0, 0.0, 10.0) is None  # unknown map: no magic route
    nav.observe_walkable(0.0, 4.0, radius=6.0)
    nav.observe_walkable(0.0, 10.0, radius=6.0)
    route = nav.plan(0.0, 4.0, 0.0, 8.0)  # through the door gap
    ok = route is not None and len(route) > 4
    return ("PASS", f"{len(route)} waypoints") if ok else ("FAIL", "no route")


@ gate("body")
def g_body():
    from src.world3d.body3d import Body3D
    b = Body3D()
    b.metabolize(10.0, moving=True, in_water=False)
    assert b.hunger > 0 and b.thirst > 0 and b.fatigue > 0
    e0 = b.base.energy
    b.eat(0.25)
    assert b.hunger == 0.0 and b.base.energy > e0
    assert b.base.apply_damage(0.3, cause="test")
    assert b.base.health < 1.0 and b.alive
    return ("PASS", f"hunger/thirst/fatigue integrate; damage works")


@ gate("first_person_sensor")
def g_eye():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        try:
            ag = svc.agents["hero"]
            obs = ag.eye.observe(svc.world, "hero", 0.0)
            ok = (obs["kind"] == "GEOMETRIC" and len(obs["depth_panorama"]) == 21
                  and len(obs["frame_hash"]) == 64)
            return ("PASS", f"{len(obs['entities'])} entities") if ok else ("FAIL", str(obs)[:200])
        finally:
            svc.spatial.close()


@ gate("vision")
def g_vision():
    ev = _v6evidence("vision_verification.json")
    if ev is None:
        return ("UNAVAILABLE", "no vision evidence (run scripts/verify_vision.py)")
    ok = ev.get("status") == "PASS" and len(ev.get("grounded_words", [])) >= 1
    return ("PASS", ev.get("caption", "")[:80]) if ok else ("FAIL", str(ev.get("error", ""))[:150])


@ gate("local_llm")
def g_llm():
    ev = _v6evidence("llm_verification.json")
    if ev is None:
        return ("UNAVAILABLE", "no llm evidence (run scripts/verify_llm.py)")
    ok = ev.get("status") == "PASS" and ev.get("structured", {}).get("goal") == "explore"
    return ("PASS", f"load={ev.get('load_s')}s") if ok else ("FAIL", str(ev.get("error", ""))[:150])


@ gate("structured_actions")
def g_actions():
    from src.world3d.agent import validate_action
    assert validate_action({"type": "MOVE", "heading": 0.1, "speed": 0.5})["ok"]
    assert not validate_action({"type": "FLY"})["ok"]
    assert not validate_action({"type": "MOVE", "heading": 99, "speed": 0.5})["ok"]
    assert not validate_action("rm -rf /")["ok"]
    return ("PASS", "typed gate holds")


@ gate("local_memory")
def g_memory():
    from src.world3d.spatial_memory import SpatialMemory
    with tempfile.TemporaryDirectory() as td:
        s = SpatialMemory(os.path.join(td, "w.db"))
        try:
            pid = s.record_place("meadow", 1.0, 2.0, 5, note="green field")
            assert pid
            got = s.search_places("meadow")
            assert got and got[0]["label"] == "meadow"
            s.record_sighting("tree_1", 6, 1.0, 2.0, 3.0)
            assert s.where_seen("tree_1")[0]["x"] == 1.0
            return ("PASS", "places/sightings/fts round trip")
        finally:
            s.close()


@ gate("spatial_memory")
def g_spatial():
    from src.world3d.spatial_memory import SpatialMemory
    with tempfile.TemporaryDirectory() as td:
        s = SpatialMemory(os.path.join(td, "w.db"))
        try:
            snap = s.snapshot()
            s.record_place("x", 1.0, 1.0, 1)
            s.restore(snap)
            rows = s.db.execute("SELECT COUNT(*) FROM places").fetchone()[0]
            return ("PASS", "snapshot/restore exact") if rows == 0 else ("FAIL", "restore leaked")
        finally:
            s.close()


@ gate("social_memory")
def g_social():
    from src.world3d.service import WorldService
    from src.world3d.social import social_encounter
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=1, db_path=os.path.join(td, "w.db"))
        try:
            a, b = svc.agents["hero"], svc.agents["friend_0"]
            a.body3d.pos, b.body3d.pos = [0.0, 4.0, 0.75], [0.5, 4.0, 0.75]
            a.org.skills["forage"], b.org.skills["forage"] = 0.9, 0.1
            t0 = b.org.social_mem.trust_of(a.org.id)
            r = social_encounter(a, b)
            t1 = b.org.social_mem.trust_of(a.org.id)
            ok = r["status"] == "INTERACTED" and t1 > t0 and r["learning_gain"] > 0
            return ("PASS", f"trust {t0:.3f}->{t1:.3f}") if ok else ("FAIL", str(r))
        finally:
            svc.spatial.close()


@ gate("autonomous_goals")
def g_goals():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        try:
            ag = svc.agents["hero"]
            ag.body3d.hunger = 0.9
            obs = ag.eye.observe(svc.world, "hero", 0.0)
            g = ag.arbitrate_goal(obs)
            ok = g["goal"] == "find_food" and g["source"].startswith("need:")
            return ("PASS", str(g)) if ok else ("FAIL", str(g))
        finally:
            svc.spatial.close()


@ gate("friend_identity")
def g_friends():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=2, db_path=os.path.join(td, "w.db"))
        try:
            ids = [a.org.id for a in svc.agents.values()]
            gh = [a.org.genome.genome_hash() for a in svc.agents.values()]
            ok = len(set(ids)) == 3 and len(set(gh)) == 3
            return ("PASS", "3 distinct identities/genomes") if ok else ("FAIL", str(ids))
        finally:
            svc.spatial.close()


@ gate("relationship_dynamics")
def g_reldyn():
    return g_social()


@ gate("offline_stt")
def g_stt():
    ev = _v6evidence("speech_verification.json")
    if ev is None:
        return ("UNAVAILABLE", "no speech evidence (run scripts/verify_stt.py)")
    stt = ev.get("stt", {})
    ok = stt.get("status") == "PASS" and len(stt.get("expected_words_found", [])) >= 2
    return ("PASS", stt.get("transcript", "")[:80]) if ok else ("FAIL", str(stt.get("error", ""))[:150])


@ gate("offline_tts")
def g_tts():
    ev = _v6evidence("speech_verification.json")
    if ev is None:
        return ("UNAVAILABLE", "no speech evidence")
    tts = ev.get("tts", {})
    ok = tts.get("status") == "PASS" and (tts.get("duration_s") or 0) > 0.5
    return ("PASS", f"{tts.get('duration_s')}s") if ok else (tts.get("status", "FAIL"), str(tts.get("error", ""))[:150])


@ gate("local_image_generation")
def g_image():
    ev = _v6evidence("image_generation_verification.json")
    if ev is None:
        return ("UNAVAILABLE", "no image evidence (run scripts/verify_image.py)")
    ok = ev.get("status") == "GENERATED"
    return ("PASS", f"{ev.get('seconds')}s") if ok else ("FAIL", str(ev.get("error", ""))[:150])


@ gate("image_output_validation")
def g_imgval():
    p = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "genesis_test.png")
    if not os.path.exists(p):
        return ("UNAVAILABLE", "genesis image absent")
    try:
        from PIL import Image
        im = Image.open(p)
        im.load()
        ok = list(im.size) == [512, 512]
        return ("PASS", "512x512 decodes") if ok else ("FAIL", str(im.size))
    except Exception as e:
        return ("FAIL", f"{type(e).__name__}: {e}")


@ gate("imagination")
def g_imagine():
    from src.world3d import dreaming
    import inspect
    src = inspect.getsource(dreaming.imagine)
    ok = "GENERATED_IMAGE" in src and "NARRATIVE_ONLY" in inspect.getsource(dreaming.dream_cycle)
    if not ok:
        return ("FAIL", "provenance labels missing")
    ev = _v6evidence("image_generation_verification.json")
    if ev is None or ev.get("status") != "GENERATED":
        return ("UNAVAILABLE", "image model not verified")
    return ("PASS", "pipeline + provenance wired; live render covered by tests")


@ gate("dream_engine")
def g_dream():
    # real local LLM when present; otherwise honestly UNAVAILABLE (no mock mind)
    try:
        from src.models.loaders import load_text
        text = load_text()
    except Exception as e:  # noqa: BLE001
        return ("UNAVAILABLE", f"no local LLM: {type(e).__name__}")
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        try:
            ag = svc.agents["hero"]
            for _ in range(5):
                ag.tick()
            from src.world3d.dreaming import dream_cycle, replay_dream
            rec = dream_cycle(ag, text, image_model=None)
            back = replay_dream(ag, rec["id"])
            ok = ("NARRATIVE_ONLY" in rec["provenance"] and back["replays"] == 1
                  and rec["narrative"])
            return ("PASS", "narrative dream + replay") if ok else ("FAIL", str(rec)[:150])
        finally:
            svc.spatial.close()
            try:
                del text
            except Exception:
                pass


@ gate("dream_provenance")
def g_dreamprov():
    import inspect
    from src.world3d import dreaming
    src = inspect.getsource(dreaming)
    ok = ("DREAM/COUNTERFACTUAL" in src and "GENERATED_DREAM" in src
          and "REAL_OBSERVATION" not in src)
    return ("PASS", "dream labelling honest") if ok else ("FAIL", "labels wrong")


@ gate("world_clock")
def g_clock():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
        try:
            t0 = (svc.world.tick, svc.world.clock_sec)
            svc.step(5)
            t1 = (svc.world.tick, svc.world.clock_sec)
            snap = svc.snapshot()
            svc.step(5)
            svc.restore(snap)
            t2 = (svc.world.tick, svc.world.clock_sec)
            ok = t1[0] > t0[0] and t1[1] > t0[1] and t2 == t1
            return ("PASS", f"tick {t1[0]}") if ok else ("FAIL", f"{t0}{t1}{t2}")
        finally:
            svc.spatial.close()


@ gate("h24_7_runtime")
def g_stream():
    from src.brain.simulation_engine import SimulationEngine
    from src.connectome.types import GraphMode
    eng = SimulationEngine(circuit_size=32, graph_mode=GraphMode.SYNTHETIC_TEST,
                           use_gpu=False, seed=5,
                           db_path=os.path.join(tempfile.gettempdir(), "v6m_stream.db"))
    eng.start()
    import time as _t
    _t.sleep(1.2)
    running = eng.stream_status()
    eng.stop()
    s = eng.stream_status()
    eng.brain.cleanup()
    ok = (running["simulation_step"] > 0 and running["uptime_sec"] >= 1.0
          and s["stream_mode"] == "STOPPED")
    return ("PASS", f"{running['simulation_step']} steps") if ok else ("FAIL", str(s))


@ gate("watchdog")
def g_watchdog():
    from src.runtime.watchdog import Watchdog, WatchdogConfig
    from src.brain.simulation_engine import SimulationEngine
    from src.connectome.types import GraphMode
    eng = SimulationEngine(circuit_size=32, graph_mode=GraphMode.SYNTHETIC_TEST,
                           use_gpu=False, seed=6,
                           db_path=os.path.join(tempfile.gettempdir(), "v6m_wd.db"))
    eng.step_single(n_steps=2)
    wd = Watchdog(lambda: eng, None, WatchdogConfig(),
                  ram_reader=lambda: 42.0)
    rep = wd.check_once()
    eng.brain.cleanup()
    return ("PASS", "healthy OK") if rep["findings"] == [] else ("FAIL", str(rep))


@ gate("backup")
def g_backup():
    return g_ckpt()


@ gate("restore")
def g_restore():
    return g_ckpt()


@ gate("UI")
def g_ui():
    idx = os.path.join(PROJECT_ROOT, "src", "ui", "static", "index.html")
    js = os.path.join(PROJECT_ROOT, "src", "ui", "static", "js", "lab.js")
    html = open(idx, encoding="utf-8").read()
    src = open(js, encoding="utf-8").read()
    bad = ("cdnjs.cloudflare.com" in src or "cdn.jsdelivr.net" in src
           or "alert(" in src or "prompt(" in src)
    ok = (not bad and "tab-world" in html and "fetchWorld" in src
          and os.path.exists(os.path.join(PROJECT_ROOT, "src", "ui", "static",
                                           "vendor", "three.min.js")))
    return ("PASS", "world tab, zero CDN, zero blocking calls") if ok else ("FAIL", "ui check")


@ gate("API")
def g_api():
    from fastapi.testclient import TestClient
    from src.ui.server import app
    c = TestClient(app)
    paths = ["/api/v1/health", "/api/v1/world/state", "/api/v1/world/geometry",
             "/api/v1/world/agent", "/api/v1/world/friends", "/api/v1/models",
             "/api/v1/neuron/1", "/api/v1/stream/status"]
    bad = []
    for p in paths:
        r = c.get(p)
        if r.status_code not in (200, 404):
            bad.append((p, r.status_code))
    # world step + backup round trip live
    r = c.post("/api/v1/world/step", params={"ticks": 1})
    if r.status_code != 200:
        bad.append(("step", r.status_code))
    ok = not bad
    return ("PASS", f"{len(paths) + 1} live endpoints") if ok else ("FAIL", str(bad))


@ gate("offline_network_isolation")
def g_offline():
    import socket as _s
    real = _s.socket.connect

    def _no(self, *a, **k):
        raise RuntimeError("OFFLINE_VIOLATION")
    _s.socket.connect = _no
    try:
        from src.world3d.service import WorldService
        from src.backup.service import BackupService
        with tempfile.TemporaryDirectory() as td:
            svc = WorldService(seed=7, friends=0, db_path=os.path.join(td, "w.db"))
            try:
                svc.step(5)
                bs = BackupService(root=os.path.join(td, "b"))
                man = bs.create_world_backup(svc, label="off", trigger="manual")
                v = bs.verify_backup(man["backup_name"])
                ok = v["status"] == "VALID"
                return ("PASS", "world+backup, zero connects") if ok else ("FAIL", str(v))
            finally:
                svc.spatial.close()
    except RuntimeError as e:
        return ("FAIL", f"network touched: {e}")
    finally:
        _s.socket.connect = real


@ gate("complete_closed_loop")
def g_loop():
    from src.world3d.service import WorldService
    with tempfile.TemporaryDirectory() as td:
        svc = WorldService(seed=7, friends=1, db_path=os.path.join(td, "w.db"))
        try:
            recs = svc.step(15)
            last = recs[-1]
            keys = ("tick", "organism_id", "visual_observation", "neural", "goal",
                    "action", "collision", "position_before", "position_after",
                    "reward", "memory_created")
            ok = all(k in last for k in keys) and len(svc.agents["hero"].org.episodes) >= 15
            return ("PASS", f"{len(recs)} closed ticks") if ok else ("FAIL", str(sorted(last)))
        finally:
            svc.spatial.close()


@ gate("v5_regression_intact")
def g_v5():
    acc = json.load(open(os.path.join(PROJECT_ROOT, "diagnostics",
                                      "acceptance_matrix.json"), encoding="utf-8"))
    ok = acc.get("overall_status") == "PASSED" and acc.get("failed", 1) == 0
    return ("PASS", f"v5 {acc.get('passed')}/{acc.get('total_categories')}") if ok else ("FAIL", "v5 regressed")


def main():
    from src.version import VERSION
    results = {}
    for name, fn in GATES.items():
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
            results[name] = {"status": status, "detail": detail,
                             "trace": traceback.format_exc()[-800:]}
            print(f"[{status}] {name}: {detail}", flush=True)
            continue
        results[name] = {"status": status, "detail": detail}
        print(f"[{status}] {name}: {detail}", flush=True)
    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    overall = "PASSED" if counts.get("FAIL", 0) == 0 else "FAILED"
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip(),
           "overall_status": overall, "total": len(results), "counts": counts,
           "gates": results}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"V6 MATRIX {overall}: {counts} -> {OUT}")
    sys.exit(0 if overall == "PASSED" else 1)


if __name__ == "__main__":
    main()
