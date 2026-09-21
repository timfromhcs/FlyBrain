"""First-class backup infrastructure (V5 phase_12).

Local-disk backup target with:
  - manifest schema flybrain_backup_v1 (hashes, identity, provenance)
  - triggers: manual / interval / before_experiment / before_restore /
    before_restart / shutdown (callers pass the trigger name; enforced log)
  - multiple generations with retention (never deletes the only valid
    recovery point: prune verifies the survivor first)
  - strict path validation (no traversal, no overwrite of live sources)
  - restore tested against a clean runtime (tests/test_backup_roundtrip.py)

Google Drive is a SEPARATE optional provider (src/backup/gdrive.py) and is
never silently mixed into local backups.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from src.version import VERSION

BACKUP_SCHEMA_VERSION = "flybrain_backup_v1"
DEFAULT_ROOT = os.environ.get("FLYBRAIN_BACKUP_DIR", "backups")
MAX_GENERATIONS = 5
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


class BackupService:
    def __init__(self, root: str = DEFAULT_ROOT, max_generations: int = MAX_GENERATIONS):
        self.root = os.path.abspath(root)
        self.max_generations = max(1, int(max_generations))
        os.makedirs(self.root, exist_ok=True)
        self.log_path = os.path.join(self.root, "backup_log.jsonl")

    # ---- validation ----
    def _dir_for(self, name: str) -> str:
        if _NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid backup name {name!r}")
        path = os.path.abspath(os.path.join(self.root, name))
        if path != os.path.join(self.root, name) or not path.startswith(self.root + os.sep):
            raise ValueError(f"backup path escapes root: {name!r}")
        return path

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "event": event,
                                    **payload}, sort_keys=True) + "\n")
        except Exception:
            pass

    # ---- create ----
    def create_backup(self, engine, colony=None, label: str = "manual",
                      trigger: str = "manual") -> Dict[str, Any]:
        """Snapshot engine (+ optional colony) into a new generation directory."""
        from src.connectome.types import GraphMode as _GM
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        safe_label = _NAME_RE.fullmatch(label) and label or "backup"
        name = f"{stamp}_{safe_label}"
        dest = self._dir_for(name)
        if os.path.exists(dest):
            raise ValueError(f"backup {name!r} already exists (never overwrite)")
        os.makedirs(dest)

        brain = engine.brain
        brain.sync_gpu_weights() if hasattr(brain, "sync_gpu_weights") else None
        snap_path = os.path.join(dest, "brain_snapshot.npz")
        brain.save_snapshot(snap_path)

        state_hash = hashlib.sha256()
        for arr in (brain.state.membrane_potentials, brain.state.spikes,
                    brain.state.refractory_steps, brain.graph.weights):
            state_hash.update(arr.tobytes())
        files = {"brain_snapshot.npz": _sha256_file(snap_path)}

        colony_hash = ""
        if colony is not None:
            cpath = os.path.join(dest, "colony.json")
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(colony.snapshot(), f)
            files["colony.json"] = _sha256_file(cpath)
            colony_hash = colony.population_hash()

        research_src = os.path.join("diagnostics", "research_memory.jsonl")
        if os.path.exists(research_src):
            rpath = os.path.join(dest, "research_memory.jsonl")
            shutil.copyfile(research_src, rpath)
            files["research_memory.jsonl"] = _sha256_file(rpath)

        config = {
            "project_version": VERSION, "git_commit": _git_commit(),
            "simulation_step": brain.state.step_count,
            "experiment_seed": engine.seed,
            "graph_mode": engine.circuit.mode.value,
            "graph_identity": _GM.canonical(engine.circuit.mode),
            "graph_hash": engine.circuit.graph_hash,
            "circuit_size": engine.circuit.num_neurons,
            "backend": ("vulkan_gpu" if (brain.gpu_engine and brain.use_gpu)
                        else "cpu_reference"),
        }
        cpath = os.path.join(dest, "config.json")
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        files["config.json"] = _sha256_file(cpath)

        manifest = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "backup_name": name, "trigger": trigger, "label": safe_label,
            "created_at": time.time(), "target": "local_disk",
            "config": config,
            "state_hash": state_hash.hexdigest(),
            "population_hash": colony_hash,
            "files": files,
        }
        mpath = os.path.join(dest, "manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        self._log("created", {"backup": name, "trigger": trigger,
                              "state_hash": manifest["state_hash"]})
        self._prune_generations()
        return manifest

    # ---- list / verify ----
    def list_backups(self) -> List[Dict[str, Any]]:
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root), reverse=True):
            dest = os.path.join(self.root, name)
            mpath = os.path.join(dest, "manifest.json")
            if not os.path.isdir(dest) or not os.path.exists(mpath):
                continue
            try:
                m = json.load(open(mpath, encoding="utf-8"))
                out.append({"backup": name, "trigger": m.get("trigger"),
                            "created_at": m.get("created_at"),
                            "state_hash": m.get("state_hash"),
                            "step": m.get("config", {}).get("simulation_step")})
            except Exception:
                out.append({"backup": name, "error": "unreadable manifest"})
        return out

    def verify_backup(self, name: str) -> Dict[str, Any]:
        """Recompute every file hash; NEVER report valid without checking."""
        dest = self._dir_for(name)
        mpath = os.path.join(dest, "manifest.json")
        if not os.path.exists(mpath):
            return {"backup": name, "status": "ERROR", "reason": "missing manifest"}
        try:
            manifest = json.load(open(mpath, encoding="utf-8"))
        except Exception as e:
            return {"backup": name, "status": "ERROR", "reason": f"manifest unreadable: {e}"}
        if manifest.get("schema_version") != BACKUP_SCHEMA_VERSION:
            return {"backup": name, "status": "ERROR",
                    "reason": f"unsupported schema {manifest.get('schema_version')!r}"}
        per_file = {}
        ok = True
        for fname, expected in manifest.get("files", {}).items():
            fpath = os.path.join(dest, fname)
            if not os.path.exists(fpath):
                per_file[fname] = "MISSING"
                ok = False
            elif _sha256_file(fpath) != expected:
                per_file[fname] = "HASH_MISMATCH"
                ok = False
            else:
                per_file[fname] = "OK"
        # state hash recomputation from the snapshot (independent check)
        state_ok = "SKIPPED"
        try:
            import numpy as _np
            if manifest.get("kind") == "world":
                ws = json.load(open(os.path.join(dest, "world_snapshot.json"),
                                    encoding="utf-8"))
                ph = ws["world"]["physics"]
                h = hashlib.sha256()
                h.update(_np.ascontiguousarray(ph["qpos"], dtype=_np.float64).tobytes())
                h.update(_np.ascontiguousarray(ph["qvel"], dtype=_np.float64).tobytes())
                h.update(json.dumps(
                    {"tick": ws["world"].get("tick"),
                     "clock": round(float(ws["world"].get("clock_sec", 0.0)), 4),
                     "consumed": sorted(ws["world"].get("consumed", [])),
                     "doors": ws["world"].get("door_state", {})},
                    sort_keys=True).encode())
                # world_snapshot stores live door_state under world_state? use manifest
                state_ok = "OK" if h.hexdigest() == manifest.get("world_hash") else "MISMATCH"
                ok = ok and state_ok == "OK"
            else:
                d = _np.load(os.path.join(dest, "brain_snapshot.npz"), allow_pickle=True)
                h = hashlib.sha256()
                for k in ("membrane_potentials", "spikes", "refractory_steps", "weights"):
                    h.update(_np.ascontiguousarray(d[k]).tobytes())
                state_ok = "OK" if h.hexdigest() == manifest.get("state_hash") else "MISMATCH"
                ok = ok and state_ok == "OK"
        except Exception as e:
            state_ok = f"ERROR: {e}"
            ok = False
        result = {"backup": name, "status": "VALID" if ok else "CORRUPT",
                  "files": per_file, "state_hash_check": state_ok}
        self._log("verified", {"backup": name, **{k: v for k, v in result.items()
                                                  if k != "files"}})
        return result

    # ---- restore ----
    def restore_backup(self, name: str, engine, colony=None) -> Dict[str, Any]:
        """Restore into engine (must be paused). Auto-backup first
        (before_restore trigger); post-restore state must equal manifest."""
        if getattr(engine, "is_running", False):
            raise RuntimeError("engine must be paused before restore (refusing live restore)")
        pre = self.create_backup(engine, colony, label="pre_restore", trigger="before_restore")
        dest = self._dir_for(name)
        verdict = self.verify_backup(name)
        if verdict["status"] != "VALID":
            raise ValueError(f"refusing to restore {name!r}: {verdict['status']}")
        engine.brain.load_snapshot(os.path.join(dest, "brain_snapshot.npz"))
        restored = None
        if colony is not None and os.path.exists(os.path.join(dest, "colony.json")):
            from src.common.determinism import SeedBundle
            from src.population.population import Population
            snap = json.load(open(os.path.join(dest, "colony.json"), encoding="utf-8"))
            seed = int(snap.get("experiment_seed", 7))
            seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                               organism_seed=seed + 2, development_seed=seed + 3,
                               mutation_seed=seed + 4, world_seed=seed + 5,
                               teacher_seed=seed + 6)
            restored = Population.restore(snap, seeds)
        # post-restore verification
        h = hashlib.sha256()
        for arr in (engine.brain.state.membrane_potentials, engine.brain.state.spikes,
                    engine.brain.state.refractory_steps, engine.brain.graph.weights):
            h.update(arr.tobytes())
        manifest = json.load(open(os.path.join(dest, "manifest.json"), encoding="utf-8"))
        match = h.hexdigest() == manifest.get("state_hash")
        self._log("restored", {"backup": name, "pre_restore": pre["backup_name"],
                               "state_match": match})
        if not match:
            raise ValueError("post-restore state hash mismatch (restore rejected)")
        return {"status": "RESTORED", "backup": name, "state_hash": h.hexdigest(),
                "pre_restore_backup": pre["backup_name"],
                "colony_restored": restored is not None}

    # ---- world backup (V6: world + agents + spatial memory) ----
    def create_world_backup(self, world_service, label: str = "world",
                            trigger: str = "manual") -> Dict[str, Any]:
        """Snapshot a WorldService (world+agents+spatial DB file). Model
        binaries are NEVER duplicated: only metadata (registry snapshot)."""
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        safe_label = _NAME_RE.fullmatch(label) and label or "world"
        name = f"{stamp}_{safe_label}"
        dest = self._dir_for(name)
        if os.path.exists(dest):
            raise ValueError(f"backup {name!r} already exists (never overwrite)")
        os.makedirs(dest)
        snap = world_service.snapshot()
        spath = os.path.join(dest, "world_snapshot.json")
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(snap, f)
        files = {"world_snapshot.json": _sha256_file(spath)}
        try:
            world_service.spatial.db.commit()
            dpath = os.path.join(dest, "spatial.db")
            shutil.copyfile(world_service.spatial.path, dpath)
            files["spatial.db"] = _sha256_file(dpath)
        except Exception as e:  # noqa: BLE001
            files["spatial.db"] = f"SKIPPED:{e}"
        try:
            from src.models.registry import scan_local
            reg = os.path.join(dest, "model_registry.json")
            with open(reg, "w", encoding="utf-8") as f:
                json.dump(scan_local(), f, indent=2)
            files["model_registry.json"] = _sha256_file(reg)
        except Exception:
            pass
        from src.version import VERSION as _V
        manifest = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "backup_name": name, "trigger": trigger, "label": safe_label,
            "created_at": time.time(), "target": "local_disk",
            "kind": "world",
            "project_version": _V, "git_commit": _git_commit(),
            "world_tick": world_service.world.tick,
            "world_hash": world_service.world.state_hash(),
            "agents": sorted(world_service.agents),
            "state_hash": hashlib.sha256(
                world_service.world.state_hash().encode()).hexdigest(),
            "files": files,
        }
        mpath = os.path.join(dest, "manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        self._log("created", {"backup": name, "trigger": trigger, "kind": "world",
                              "world_hash": manifest["world_hash"]})
        self._prune_generations()
        return manifest

    def restore_world_backup(self, name: str, world_service) -> Dict[str, Any]:
        verdict = self.verify_backup(name)
        if verdict["status"] != "VALID":
            raise ValueError(f"refusing to restore {name!r}: {verdict['status']}")
        dest = self._dir_for(name)
        snap = json.load(open(os.path.join(dest, "world_snapshot.json"), encoding="utf-8"))
        world_service.restore(snap)
        h = world_service.world.state_hash()
        manifest = json.load(open(os.path.join(dest, "manifest.json"), encoding="utf-8"))
        match = h == manifest.get("world_hash")
        self._log("restored", {"backup": name, "kind": "world", "hash_match": match})
        if not match:
            raise ValueError("post-restore world hash mismatch")
        return {"status": "RESTORED", "backup": name, "world_hash": h}

    # ---- retention ----
    def _prune_generations(self) -> None:
        backups = [b["backup"] for b in self.list_backups() if "error" not in b]
        while len(backups) > self.max_generations:
            victim = backups.pop()  # oldest last in reverse-sorted list
            survivors = [b for b in backups]
            if not survivors:
                return  # never delete the only recovery point
            if self.verify_backup(survivors[-1])["status"] != "VALID":
                return  # survivor unverified: keep everything, log
            shutil.rmtree(self._dir_for(victim), ignore_errors=True)
            self._log("pruned", {"backup": victim})
