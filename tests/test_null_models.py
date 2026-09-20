"""V5 phase_05: topology null controls are deterministic, honest, exact."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode, ProvenanceStatus
from src.connectome.nulls import build_null_graph, NULL_MODELS


class TestNullModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = get_or_create_circuit(64, mode=GraphMode.REAL, seed=42,
                                      cache_name="null_src_64.npz")

    def test_all_models_deterministic(self):
        for model in NULL_MODELS:
            a = build_null_graph(self.g, model, seed=7)
            b = build_null_graph(self.g, model, seed=7)
            self.assertEqual(a.graph_hash, b.graph_hash)
            c = build_null_graph(self.g, model, seed=8)
            self.assertNotEqual(a.graph_hash, c.graph_hash)

    def test_edge_counts_and_honesty(self):
        for model in NULL_MODELS:
            n = build_null_graph(self.g, model, seed=7)
            self.assertEqual(n.num_neurons, self.g.num_neurons)
            self.assertEqual(n.num_synapses, self.g.num_synapses)
            self.assertNotEqual(n.graph_hash, self.g.graph_hash)
            pm = n.provenance_metadata
            self.assertEqual(pm["null_model"], model)
            self.assertFalse(pm["empirical"])
            self.assertEqual(pm["derived_from_graph_hash"], self.g.graph_hash)
            if self.g.mode == GraphMode.REAL:
                self.assertEqual(n.provenance_status, ProvenanceStatus.EXPERIMENTAL)

    def test_degree_preserving_exact(self):
        n = build_null_graph(self.g, "DEGREE_PRESERVING_RANDOM", seed=7)
        src_in = [int(self.g.row_offsets[i + 1]) - int(self.g.row_offsets[i])
                  for i in range(self.g.num_neurons)]
        got_in = [int(n.row_offsets[i + 1]) - int(n.row_offsets[i])
                  for i in range(n.num_neurons)]
        self.assertEqual(src_in, got_in)
        # out-degree via column histogram
        src_out = np.bincount(self.g.col_indices, minlength=self.g.num_neurons)
        got_out = np.bincount(n.col_indices, minlength=n.num_neurons)
        np.testing.assert_array_equal(src_out, got_out)
        # no self loops
        for i in range(n.num_neurons):
            s, e = int(n.row_offsets[i]), int(n.row_offsets[i + 1])
            self.assertNotIn(i, list(n.col_indices[s:e]))

    def test_weight_shuffled_topology_identical(self):
        n = build_null_graph(self.g, "WEIGHT_SHUFFLED", seed=7)
        np.testing.assert_array_equal(n.row_offsets, self.g.row_offsets)
        np.testing.assert_array_equal(n.col_indices, self.g.col_indices)
        self.assertFalse(np.array_equal(n.weights, self.g.weights))
        self.assertAlmostEqual(float(np.sum(n.weights)), float(np.sum(self.g.weights)),
                               places=4)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            build_null_graph(self.g, "FAKE_MODEL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
