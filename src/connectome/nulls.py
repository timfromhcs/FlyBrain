"""Deterministic topology null models (V5 phase_05).

Controls for testing whether observed effects depend on the empirical
topology rather than on size/density/weight statistics:

  EDGE_SHUFFLED            - same N/M, targets rewired uniformly (seeded)
  DEGREE_PRESERVING_RANDOM - in+out degree sequences preserved via
                             stub-matching (seeded, no self-loops)
  WEIGHT_SHUFFLED          - topology fixed, weights permuted (seeded)

Every null graph carries provenance_metadata["null_model"] naming the
transform + the source graph hash, and empirical=False: a null derived
from REAL topology is NEVER labelled VERIFIED biology.
"""
import hashlib
from typing import Dict, List

import numpy as np

from src.connectome.loader import build_population_registry
from src.connectome.types import ConnectomeGraph, GraphMode, ProvenanceStatus

NULL_MODELS = ("EDGE_SHUFFLED", "DEGREE_PRESERVING_RANDOM", "WEIGHT_SHUFFLED")


def _edges_incoming(graph: ConnectomeGraph) -> List[List[int]]:
    N = graph.num_neurons
    out: List[List[int]] = [[] for _ in range(N)]
    for post in range(N):
        s, e = int(graph.row_offsets[post]), int(graph.row_offsets[post + 1])
        out[post] = [int(c) for c in graph.col_indices[s:e]]
    return out


def _build_csr(N: int, incoming: List[List[int]], weights: List[List[float]]):
    row = [0]
    cols: List[int] = []
    wts: List[float] = []
    for i in range(N):
        order = sorted(range(len(incoming[i])), key=lambda k: incoming[i][k])
        for k in order:
            cols.append(incoming[i][k])
            wts.append(weights[i][k])
        row.append(len(cols))
    return (np.array(row, dtype=np.int32), np.array(cols, dtype=np.int32),
            np.array(wts, dtype=np.float32))


def build_null_graph(graph: ConnectomeGraph, null_model: str, seed: int = 42
                     ) -> ConnectomeGraph:
    if null_model not in NULL_MODELS:
        raise ValueError(f"unknown null model {null_model!r}; choices {NULL_MODELS}")
    rng = np.random.RandomState(seed)
    N = graph.num_neurons
    incoming = _edges_incoming(graph)
    M = sum(len(r) for r in incoming)
    w_in = [[float(graph.weights[k]) for k in
             range(int(graph.row_offsets[i]), int(graph.row_offsets[i + 1]))]
            for i in range(N)]

    if null_model == "EDGE_SHUFFLED":
        all_w = [w for row in w_in for w in row]
        perm = rng.permutation(len(all_w))
        flat_w = [all_w[p] for p in perm]
        new_in = [[int(rng.randint(N)) for _ in row] for row in incoming]
        # drop accidental self-loops deterministically (shift target)
        new_in = [[(t + 1) % N if t == i else t for t in row]
                  for i, row in enumerate(new_in)]
        pos = 0
        new_w = []
        for row in new_in:
            new_w.append(flat_w[pos:pos + len(row)])
            pos += len(row)
    elif null_model == "WEIGHT_SHUFFLED":
        all_w = [w for row in w_in for w in row]
        perm = rng.permutation(len(all_w))
        flat_w = [all_w[p] for p in perm]
        new_in = [list(row) for row in incoming]
        pos = 0
        new_w = []
        for row in new_in:
            new_w.append(flat_w[pos:pos + len(row)])
            pos += len(row)
    else:  # DEGREE_PRESERVING_RANDOM via directed stub matching
        out_stubs: List[int] = []
        in_stubs: List[int] = []
        out_deg = [0] * N
        in_deg = [len(r) for r in incoming]
        for post, row in enumerate(incoming):
            for pre in row:
                out_stubs.append(pre)
                in_stubs.append(post)
                out_deg[pre] += 1
        rng.shuffle(out_stubs)
        rng.shuffle(in_stubs)
        pairs = []
        used = set()
        left_o, left_i = list(out_stubs), list(in_stubs)
        attempts = 0
        while left_o and attempts < 20 * max(1, M):
            attempts += 1
            o = left_o.pop(0)
            # find a non-self, non-duplicate target
            pick = None
            for j, t in enumerate(left_i):
                if t != o and (o, t) not in used:
                    pick = j
                    break
            if pick is None:
                left_o.append(o)
                continue
            t = left_i.pop(pick)
            used.add((o, t))
            pairs.append((o, t))
        # any leftovers (rare, dense graphs): deterministic fallback edges
        for o in left_o:
            t = next((c for c in range(N) if c != o and (o, c) not in used), None)
            if t is not None:
                used.add((o, t))
                pairs.append((o, t))
        new_in = [[] for _ in range(N)]
        new_w_dict: Dict[int, List[float]] = {i: [] for i in range(N)}
        # weights: permuted source weights keep the marginal distribution
        all_w = [w for row in w_in for w in row]
        rng.shuffle(all_w)
        wi = 0
        for (o, t) in sorted(pairs):
            new_in[t].append(o)
            new_w_dict[t].append(all_w[wi % len(all_w)] if all_w else 0.05)
            wi += 1
        new_w = [new_w_dict[i] for i in range(N)]
        # verify degree preservation (raises loudly if violated)
        got_in = [len(r) for r in new_in]
        got_out = [0] * N
        for row in new_in:
            for pre in row:
                got_out[pre] += 1
        if got_in != in_deg or got_out != out_deg:
            raise ValueError("degree preservation failed; refusing null graph")

    row_offsets, col_indices, weights = _build_csr(N, new_in, new_w)
    populations = build_population_registry(graph.neuron_ids, graph.coordinates,
                                            graph.tbars, list(graph.sides))
    meta = dict(getattr(graph, "provenance_metadata", {}) or {})
    meta.update({
        "null_model": null_model,
        "null_seed": int(seed),
        "derived_from_graph_hash": graph.graph_hash,
        "empirical": False,
        "note": f"{null_model} control derived from {meta.get('graph_identity', graph.mode.value)}; "
                "NOT empirical biology regardless of source.",
    })
    return ConnectomeGraph(
        neuron_ids=np.copy(graph.neuron_ids),
        coordinates=np.copy(graph.coordinates),
        tbars=np.copy(graph.tbars),
        sides=list(graph.sides),
        row_offsets=row_offsets,
        col_indices=col_indices,
        weights=weights,
        mode=graph.mode,
        provenance_status=(ProvenanceStatus.EXPERIMENTAL
                           if graph.mode == GraphMode.REAL else graph.provenance_status),
        populations=populations,
        provenance_metadata=meta,
    )
