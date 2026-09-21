"""Eligibility-trace plasticity + neuromodulation (STAGE D, versioned mode).

v1_hebbian (compat baseline, unchanged):
    dW = lr * reward * (a_pre*a_post - beta*W)

v2_eligibility (new mode):
    elig(t) = decay * elig(t-1) + a_pre(t) * a_post(t)
    signal  = w_r*reward + w_n*novelty + w_p*prediction_error
              + w_s*social + w_g*goal_success
    dW      = lr * elig * signal            (clipped to [min_w, max_w])

The neuromodulatory signal is an explicit, versioned combination; the default
config reduces exactly to reward-only. Eligibility traces persist across steps
and are resized when the living brain's synapse set changes.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

PLASTICITY_MODES = ("v1_hebbian", "v2_eligibility")


@dataclass
class NeuromodulationConfig:
    """Versioned neuromodulatory signal weights. Defaults = reward-only."""
    version: str = "neuromod_v1"
    w_reward: float = 1.0
    w_novelty: float = 0.0
    w_prediction_error: float = 0.0
    w_social: float = 0.0
    w_goal_success: float = 0.0
    signal_clip: float = 4.0

    def validate(self) -> None:
        for name in ("w_reward", "w_novelty", "w_prediction_error",
                     "w_social", "w_goal_success"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0.0 or v > 4.0:
                raise ValueError(f"neuromod weight {name}={v} outside [0, 4]")
        if not np.isfinite(self.signal_clip) or self.signal_clip <= 0:
            raise ValueError("signal_clip must be positive finite")

    def signal(self, reward: float, novelty: float = 0.0,
               prediction_error: float = 0.0, social: float = 0.0,
               goal_success: float = 0.0) -> float:
        s = (self.w_reward * float(reward)
             + self.w_novelty * float(novelty)
             + self.w_prediction_error * float(prediction_error)
             + self.w_social * float(social)
             + self.w_goal_success * float(goal_success))
        return float(np.clip(s, -self.signal_clip, self.signal_clip))

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NeuromodulationConfig":
        kwargs = {}
        for k, v in d.items():
            if k not in cls.__dataclass_fields__:
                continue
            kwargs[k] = v if k == "version" else float(v)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg


class EligibilityState:
    """Persistent per-synapse eligibility traces aligned with graph CSR order."""

    def __init__(self, num_synapses: int, seed: int = 42):
        self.traces = np.zeros(max(0, int(num_synapses)), dtype=np.float32)
        self.updates = 0
        self.seed = int(seed)

    def sync_size(self, num_synapses: int) -> bool:
        """Resize on structural change. New synapses start at zero eligibility.
        Returns True if a resize happened."""
        n = max(0, int(num_synapses))
        if len(self.traces) == n:
            return False
        if n < len(self.traces):
            self.traces = self.traces[:n].copy()
        else:
            self.traces = np.concatenate(
                [self.traces, np.zeros(n - len(self.traces), dtype=np.float32)])
        return True

    def update(self, row_offsets: np.ndarray, col_indices: np.ndarray,
               pre_spikes: np.ndarray, post_spikes: np.ndarray,
               decay: float = 0.9) -> float:
        """elig = decay*elig + pre*post per synapse (CSR: row=post, col=pre)."""
        self.sync_size(len(col_indices))
        if len(col_indices) == 0:
            return 0.0
        pre = np.asarray(pre_spikes, dtype=np.float32)
        post = np.asarray(post_spikes, dtype=np.float32)
        rows = np.repeat(np.arange(len(row_offsets) - 1),
                         np.diff(row_offsets).astype(np.int64))
        cols = np.asarray(col_indices, dtype=np.int64)
        contrib = post[rows] * pre[cols]
        self.traces = np.float32(decay) * self.traces + contrib
        self.updates += 1
        return float(np.mean(self.traces))

    def mean(self) -> float:
        return float(np.mean(self.traces)) if len(self.traces) else 0.0

    def snapshot(self) -> Dict[str, object]:
        return {"traces": self.traces.tolist(), "updates": self.updates, "seed": self.seed}

    @classmethod
    def restore(cls, payload: Dict[str, object]) -> "EligibilityState":
        st = cls(0)
        st.traces = np.asarray(payload["traces"], dtype=np.float32)
        st.updates = int(payload["updates"])
        st.seed = int(payload["seed"])
        return st


@dataclass
class EligibilityEngine:
    """v2 eligibility plasticity engine (pure functions over graph arrays)."""
    learning_rate: float = 0.05
    trace_decay: float = 0.9
    min_weight: float = 0.01
    max_weight: float = 1.0
    neuromod: NeuromodulationConfig = field(default_factory=NeuromodulationConfig)

    def __post_init__(self) -> None:
        if not (0.0 <= self.trace_decay < 1.0):
            raise ValueError(f"trace_decay {self.trace_decay} outside [0, 1)")
        self.neuromod.validate()

    def apply(self, graph, eligibility: EligibilityState, signal: float) -> int:
        """dW = lr * elig * signal, clipped. Returns number of updated synapses."""
        if len(graph.weights) == 0:
            return 0
        if len(eligibility.traces) != len(graph.weights):
            eligibility.sync_size(len(graph.weights))
        delta = np.float32(self.learning_rate * signal) * eligibility.traces
        new_w = np.clip(graph.weights + delta, self.min_weight, self.max_weight)
        graph.weights = new_w.astype(np.float32)
        return int(len(graph.weights))

    def step(self, graph, eligibility: EligibilityState, pre_spikes: np.ndarray,
             post_spikes: np.ndarray, reward: float = 0.0, novelty: float = 0.0,
             prediction_error: float = 0.0, social: float = 0.0,
             goal_success: float = 0.0) -> Dict[str, float]:
        eligibility.update(graph.row_offsets, graph.col_indices,
                           pre_spikes, post_spikes, decay=self.trace_decay)
        signal = self.neuromod.signal(reward, novelty, prediction_error, social, goal_success)
        n = self.apply(graph, eligibility, signal)
        return {"signal": round(signal, 6), "eligibility_mean": round(eligibility.mean(), 6),
                "synapses_updated": n}
