"""Central Autonomous Resource Manager for FlyBrain V8/V9.

Implements Sections 11–22:
- Dynamic empirical system probing
- Subsystem budget allocations
- Model lifecycle & admission control
- Priority-driven onload/offload (P0 to P5)
- Hysteresis pressure tracking (GREEN to CRITICAL)
- Bounded OOM recovery & quarantine
"""
import os
import gc
import time
import logging
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable, List

from src.runtime.resource_manager.prober import probe_system, probe_memory
from src.runtime.resource_manager.profile import derive_profile, compute_budgets, ProfileTier, SubsystemBudgets
from src.runtime.resource_manager.pressure import PressureEvaluator, PressureState

logger = logging.getLogger("FlyBrain.ResourceManager")


class ModelLifecycle(str, Enum):
    DISCOVERED = "DISCOVERED"
    INDEXED = "INDEXED"
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    LOADED = "LOADED"
    WARM = "WARM"
    BUSY = "BUSY"
    IDLE = "IDLE"
    DRAINING = "DRAINING"
    OFFLOADING = "OFFLOADING"
    OFFLOADED = "OFFLOADED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


class ModelPriority(int, Enum):
    P0_CORE = 0         # Brain controller, life simulation (NEVER EVICTED)
    P1_INTERACTIVE = 1  # Navigation, direct dialogue, active VLM
    P2_SIMULATION = 2   # Embeddings, spatial memory search
    P3_RESEARCH = 3     # Image generation, counterfactual replay
    P4_BACKGROUND = 4   # Dream synthesis, image->3D
    P5_SPECULATIVE = 5  # Pre-baked assets, speculative variation (FIRST EVICTED)


class ResourceManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, override_tier: Optional[ProfileTier] = None):
        if self._initialized:
            return
        self._lock = threading.RLock()
        self.probe = probe_system()
        self.tier = override_tier or derive_profile(self.probe)
        self.budgets = compute_budgets(self.probe, self.tier)
        self.pressure_eval = PressureEvaluator()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.oom_retries: Dict[str, int] = {}
        self.events: List[Dict[str, Any]] = []
        self._initialized = True
        self.log_event("INITIALIZED", f"Tier={self.tier.value}, SafeRAM={self.budgets.safe_ram_target_mb}MB")

    def log_event(self, kind: str, detail: str) -> None:
        event = {"ts": time.time(), "kind": kind, "detail": detail}
        self.events.append(event)
        if len(self.events) > 500:
            self.events.pop(0)
        logger.info(f"[{kind}] {detail}")

    def refresh_telemetry(self) -> Dict[str, Any]:
        """Refreshes live memory and updates pressure state."""
        mem = probe_memory()
        state = self.pressure_eval.evaluate(mem["ram_used_percent"])
        
        if state in (PressureState.RED, PressureState.CRITICAL):
            self._handle_high_pressure(state)
            
        return {
            "tier": self.tier.value,
            "pressure": self.pressure_eval.status(),
            "memory": mem,
            "budgets": self.budgets.to_dict(),
            "loaded_models": [k for k, v in self.models.items() if v["state"] in (ModelLifecycle.LOADED, ModelLifecycle.WARM, ModelLifecycle.BUSY)]
        }

    def register_model(self, model_id: str, priority: ModelPriority,
                       ram_estimate_mb: float, vram_estimate_mb: float,
                       loader: Callable[[], Any], unloader: Callable[[Any], None]) -> None:
        """Registers a model into managed lifecycle."""
        with self._lock:
            self.models[model_id] = {
                "id": model_id,
                "priority": priority,
                "ram_mb": ram_estimate_mb,
                "vram_mb": vram_estimate_mb,
                "state": ModelLifecycle.AVAILABLE,
                "instance": None,
                "loader": loader,
                "unloader": unloader,
                "last_used": 0.0,
                "use_count": 0,
                "quarantined": False
            }

    def request_model(self, model_id: str) -> Optional[Any]:
        """Admission-controlled model acquisition."""
        with self._lock:
            spec = self.models.get(model_id)
            if not spec:
                raise ValueError(f"Unknown model {model_id}")

            if spec["state"] == ModelLifecycle.QUARANTINED:
                raise RuntimeError(f"Model {model_id} is quarantined due to repeated failures.")

            if spec["state"] in (ModelLifecycle.LOADED, ModelLifecycle.WARM):
                spec["last_used"] = time.time()
                spec["use_count"] += 1
                return spec["instance"]

            # Admission check
            if not self._check_admission(spec):
                self._evict_for_admission(spec["ram_mb"])
                if not self._check_admission(spec):
                    raise MemoryError(f"Admission denied for {model_id}: Insufficient resource budget.")

            # Load model with OOM guard
            return self._load_model_safe(model_id)

    def _check_admission(self, spec: Dict[str, Any]) -> bool:
        mem = probe_memory()
        available_mb = mem["ram_available_mb"] if "ram_available_mb" in mem else (mem["ram_available_bytes"] / (1024 ** 2))
        safety_margin_mb = self.budgets.os_reserve_mb * 0.5
        return (available_mb - spec["ram_mb"]) >= safety_margin_mb

    def _load_model_safe(self, model_id: str) -> Any:
        spec = self.models[model_id]
        spec["state"] = ModelLifecycle.LOADING
        self.log_event("MODEL_LOADING", f"Loading {model_id} (Est: {spec['ram_mb']}MB)")
        
        try:
            instance = spec["loader"]()
            spec["instance"] = instance
            spec["state"] = ModelLifecycle.LOADED
            spec["last_used"] = time.time()
            spec["use_count"] += 1
            self.log_event("MODEL_LOADED", f"Successfully loaded {model_id}")
            return instance
        except Exception as e:
            self.log_event("MODEL_FAILED", f"Error loading {model_id}: {e}")
            return self._recover_oom(model_id, e)

    def release_model(self, model_id: str) -> None:
        """Unloads an active model releasing host and device memory."""
        with self._lock:
            spec = self.models.get(model_id)
            if not spec or spec["state"] not in (ModelLifecycle.LOADED, ModelLifecycle.WARM, ModelLifecycle.BUSY):
                return
            
            spec["state"] = ModelLifecycle.OFFLOADING
            self.log_event("MODEL_OFFLOADING", f"Evicting {model_id}")
            try:
                if spec["instance"] is not None and spec["unloader"]:
                    spec["unloader"](spec["instance"])
            except Exception as e:
                logger.warning(f"Error during unload of {model_id}: {e}")
            finally:
                spec["instance"] = None
                spec["state"] = ModelLifecycle.OFFLOADED
                gc.collect()
                self.log_event("MODEL_OFFLOADED", f"Freed memory for {model_id}")

    def _evict_for_admission(self, required_mb: float) -> None:
        """Evicts lower priority models (P5 down to P1) until required_mb is freed."""
        candidates = [m for m in self.models.values() if m["state"] in (ModelLifecycle.LOADED, ModelLifecycle.WARM)]
        # Sort by priority DESC (P5 first, P0 never), then by last_used ASC
        candidates.sort(key=lambda x: (x["priority"].value, -x["last_used"]), reverse=True)

        for cand in candidates:
            if cand["priority"] == ModelPriority.P0_CORE:
                continue # Never evict P0
            self.release_model(cand["id"])
            mem = probe_memory()
            avail_mb = mem["ram_available_bytes"] / (1024 ** 2)
            if avail_mb >= required_mb + self.budgets.os_reserve_mb * 0.5:
                break

    def _handle_high_pressure(self, state: PressureState) -> None:
        """Automated response to resource pressure."""
        self.log_event("PRESSURE_RESPONSE", f"Triggered response for {state.value}")
        # Evict P5 speculative and P4 background models
        for m in list(self.models.values()):
            if m["state"] in (ModelLifecycle.LOADED, ModelLifecycle.WARM):
                if state == PressureState.CRITICAL and m["priority"].value >= ModelPriority.P2_SIMULATION.value:
                    self.release_model(m["id"])
                elif state == PressureState.RED and m["priority"].value >= ModelPriority.P3_RESEARCH.value:
                    self.release_model(m["id"])
                elif state == PressureState.ORANGE and m["priority"].value >= ModelPriority.P4_BACKGROUND.value:
                    self.release_model(m["id"])

    def _recover_oom(self, model_id: str, error: Exception) -> Optional[Any]:
        """Bounded OOM recovery policy (Section 20)."""
        retries = self.oom_retries.get(model_id, 0)
        self.log_event("OOM_RECOVERY_ATTEMPT", f"Model {model_id} retry #{retries + 1}")
        
        if retries >= 2:
            spec = self.models[model_id]
            spec["state"] = ModelLifecycle.QUARANTINED
            self.log_event("MODEL_QUARANTINED", f"Quarantined {model_id} after {retries} failed attempts")
            raise RuntimeError(f"OOM recovery exhausted for {model_id}: {error}")

        self.oom_retries[model_id] = retries + 1
        gc.collect()
        
        # Evict all non-essential models
        for m in list(self.models.values()):
            if m["id"] != model_id and m["priority"].value >= ModelPriority.P2_SIMULATION.value:
                self.release_model(m["id"])
                
        # Re-attempt load
        return self._load_model_safe(model_id)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tier": self.tier.value,
                "pressure": self.pressure_eval.status(),
                "budgets": self.budgets.to_dict(),
                "registered_models": {k: {"state": v["state"].value, "priority": v["priority"].name, "ram_mb": v["ram_mb"]} for k, v in self.models.items()},
                "recent_events": self.events[-15:]
            }
