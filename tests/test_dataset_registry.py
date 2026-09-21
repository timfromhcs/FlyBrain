import unittest
from src.connectome.dataset_registry import DatasetRegistry, DatasetCategory


class TestDatasetRegistry(unittest.TestCase):
    def test_registered_datasets_exist(self):
        datasets = DatasetRegistry.list_all()
        self.assertGreaterEqual(len(datasets), 4)
        ids = [d.dataset_id for d in datasets]
        self.assertIn("MALECNS_V1_FULL", ids)
        self.assertIn("MALECNS_V1_DERIVED_SUBGRAPH", ids)
        self.assertIn("MALECNS_SPATIAL_SURROGATE", ids)
        self.assertIn("SYNTHETIC_TEST_V1", ids)

    def test_category_separation(self):
        full = DatasetRegistry.get("MALECNS_V1_FULL")
        derived = DatasetRegistry.get("MALECNS_V1_DERIVED_SUBGRAPH")
        surrogate = DatasetRegistry.get("MALECNS_SPATIAL_SURROGATE")
        synthetic = DatasetRegistry.get("SYNTHETIC_TEST_V1")

        self.assertEqual(full.category, DatasetCategory.BIOLOGICAL)
        self.assertEqual(derived.category, DatasetCategory.DERIVED)
        self.assertEqual(surrogate.category, DatasetCategory.SURROGATE)
        self.assertEqual(synthetic.category, DatasetCategory.SYNTHETIC)

    def test_full_vs_subgraph_distinction(self):
        full = DatasetRegistry.get("MALECNS_V1_FULL")
        derived = DatasetRegistry.get("MALECNS_V1_DERIVED_SUBGRAPH")

        self.assertTrue(full.is_full_connectome)
        self.assertFalse(derived.is_full_connectome)
        self.assertIn("DERIVED_FROM", derived.derivation_relation)

    def test_license_separation(self):
        full = DatasetRegistry.get("MALECNS_V1_FULL")
        self.assertEqual(full.dataset_license, "CC-BY-4.0")
        self.assertEqual(full.software_license, "GPL-3.0")

        derived = DatasetRegistry.get("MALECNS_V1_DERIVED_SUBGRAPH")
        self.assertEqual(derived.dataset_license, "CC-BY-4.0")
        self.assertEqual(derived.software_license, "Apache-2.0")

    def test_provenance_identity_changes_with_dataset(self):
        hash1 = DatasetRegistry.compute_experiment_provenance("MALECNS_V1_FULL", 512, 42)
        hash2 = DatasetRegistry.compute_experiment_provenance("MALECNS_V1_DERIVED_SUBGRAPH", 512, 42)
        hash3 = DatasetRegistry.compute_experiment_provenance("MALECNS_SPATIAL_SURROGATE", 512, 42)
        hash4 = DatasetRegistry.compute_experiment_provenance("SYNTHETIC_TEST_V1", 512, 42)

        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(hash2, hash3)
        self.assertNotEqual(hash3, hash4)


if __name__ == "__main__":
    unittest.main()
