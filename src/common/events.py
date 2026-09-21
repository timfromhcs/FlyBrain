"""Typed Runtime Event Bus for FlyBrain V8/V9.

Implements Section 84:
Publishes and dispatches typed simulation, cognitive, asset, and resource events.
Thread-safe, non-blocking listener invocation with bounded history.
"""
import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger("FlyBrain.EventBus")


class EventType(str, Enum):
    # Simulation & Physics
    WORLD_TICK = "WORLD_TICK"
    CHUNK_LOADED = "CHUNK_LOADED"
    CHUNK_UNLOADED = "CHUNK_UNLOADED"
    ORGANISM_MOVED = "ORGANISM_MOVED"
    COLLISION = "COLLISION"
    CONTACT = "CONTACT"
    
    # Cognition & Biology
    NEURON_SPIKE = "NEURON_SPIKE"
    GOAL_CHANGED = "GOAL_CHANGED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    SOCIAL_INTERACTION = "SOCIAL_INTERACTION"
    DREAM_STARTED = "DREAM_STARTED"
    DREAM_COMPLETED = "DREAM_COMPLETED"
    
    # Generative AI & Assets
    IMAGE_GENERATION_STARTED = "IMAGE_GENERATION_STARTED"
    IMAGE_GENERATED = "IMAGE_GENERATED"
    ASSET_GENERATION_STARTED = "ASSET_GENERATION_STARTED"
    ASSET_GENERATED = "ASSET_GENERATED"
    ASSET_VALIDATED = "ASSET_VALIDATED"
    RIG_CREATED = "RIG_CREATED"
    MOTION_STARTED = "MOTION_STARTED"
    MOTION_COMPLETED = "MOTION_COMPLETED"
    
    # Model & Resource Management
    MODEL_LOADING = "MODEL_LOADING"
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_OFFLOADED = "MODEL_OFFLOADED"
    MODEL_FAILED = "MODEL_FAILED"
    RESOURCE_PRESSURE_CHANGED = "RESOURCE_PRESSURE_CHANGED"
    
    # Persistence & Resilience
    BACKUP_CREATED = "BACKUP_CREATED"
    BACKUP_VERIFIED = "BACKUP_VERIFIED"
    BACKUP_RESTORED = "BACKUP_RESTORED"
    WATCHDOG_RECOVERY = "WATCHDOG_RECOVERY"


@dataclass
class Event:
    type: EventType
    timestamp: float = field(default_factory=time.time)
    source: str = "core"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": self.payload
        }


class EventBus:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._lock = threading.RLock()
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            et: [] for et in EventType
        }
        self._all_subscribers: List[Callable[[Event], None]] = []
        self._history: List[Event] = []
        self._max_history = 1000
        self._initialized = True

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribes a callback to a specific typed event."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """Subscribes a callback to all events (e.g. for WebSocket telemetry)."""
        with self._lock:
            self._all_subscribers.append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        """Publishes an event to all subscribers and records to bounded history."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Copy listener lists for safe iteration outside lock
            listeners = list(self._subscribers.get(event.type, []))
            wildcard = list(self._all_subscribers)

        for cb in listeners:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in listener for {event.type.value}: {e}")

        for cb in wildcard:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in wildcard listener for {event.type.value}: {e}")

    def get_recent(self, count: int = 50, event_type: Optional[EventType] = None) -> List[Dict[str, Any]]:
        """Returns recent events in chronological order."""
        with self._lock:
            if event_type is not None:
                filtered = [e for e in self._history if e.type == event_type]
            else:
                filtered = self._history
            return [e.to_dict() for e in filtered[-count:]]


def get_event_bus() -> EventBus:
    return EventBus()


# ---- Backward Compatibility Biological Event Stream (V4/V5/V7) ----
import hashlib
import os

VALID_EVENTS = {
    "NEURON_BORN", "NEURON_DIFFERENTIATED", "NEURON_MOVED",
    "AXON_GROWN", "DENDRITE_GROWN", "SYNAPSE_CREATED",
    "SYNAPSE_STRENGTHENED", "SYNAPSE_WEAKENED", "SYNAPSE_PRUNED",
    "NEURON_DIED", "MEMORY_CREATED", "MEMORY_RECALLED",
    "MEMORY_CONSOLIDATED", "SLEEP_STARTED", "DREAM_STARTED",
    "DREAM_ENDED", "TEACHING_STARTED", "TEACHING_ENDED",
    "LEARNING_EVENT", "REPRODUCTION", "MUTATION", "CROSSOVER",
    "ORGANISM_BORN", "ORGANISM_DIED", "GENERATION_STARTED",
    "GENERATION_ENDED", "WORLD_STEP", "SNAPSHOT_SAVED",
}


class EventLog:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._tick_counter = 0

    def log(self, event_type: str, tick: int, organism_id: str = "",
            generation: int = 0, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if event_type not in VALID_EVENTS:
            raise ValueError(f"Unknown event type: {event_type}")
        ev = {
            "seq": len(self.events),
            "type": event_type,
            "tick": int(tick),
            "organism_id": organism_id,
            "generation": int(generation),
            "payload": payload or {},
        }
        self.events.append(ev)
        return ev

    def __len__(self):
        return len(self.events)

    def filter(self, event_type: str = "", organism_id: str = "") -> List[Dict[str, Any]]:
        out = self.events
        if event_type:
            out = [e for e in out if e["type"] == event_type]
        if organism_id:
            out = [e for e in out if e["organism_id"] == organism_id]
        return out

    def compute_hash(self) -> str:
        import json as _json
        h = hashlib.sha256()
        for e in self.events:
            h.update(_json.dumps(e, sort_keys=True).encode())
        return h.hexdigest()

    def to_jsonl(self, path: str):
        import json as _json
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(_json.dumps(e, sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "EventLog":
        import json as _json
        log = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    log.events.append(_json.loads(line))
        return log

