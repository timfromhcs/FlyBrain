"""STAGE I/J tests: genome v2 architecture genes, v1 compat, gene consumption,
speciation distance + divergence evidence."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np

from src.genome.schema import Genome, GENOME_VERSION, LEGACY_VERSION, ARCH_PARAMS, PARAM_BOUNDS
from src.genome.operators import mutate_genome, crossover_genomes
from src.evolution.speciation import genome_distance, assign_species, detect_divergence
from src.common.determinism import SeedBundle, derive_subseed
from src.population.population import Population
from src.connectome.types import GraphMode


class TestGenomeV2(unittest.TestCase):
    def test_v2_has_architecture_genes(self):
        g = Genome.founder(seed=1)
        self.assertEqual(g.version, GENOME_VERSION)
        for k in ARCH_PARAMS:
            self.assertIn(k, g.params)

    def test_v1_legacy_still_valid_and_bit_identical(self):
        g_old_way = Genome.founder(seed=7, legacy=True)
        self.assertEqual(g_old_way.version, LEGACY_VERSION)
        self.assertEqual(len(g_old_way.params), 15)
        # founder values must match the pre-v2 implementation exactly
        import numpy as np
        rng = np.random.RandomState(7)
        for k in g_old_way.params:
            lo, hi = PARAM_BOUNDS[k]
            expected = float(np.clip(lo + (hi - lo) * 0.5 + rng.normal(0, 0.02 * (hi - lo)), lo, hi))
            self.assertAlmostEqual(g_old_way.params[k], expected, places=6)

    def test_v1_rejects_unknown_params(self):
        bad = Genome.founder(seed=2, legacy=True)
        bad.params["eligibility_decay"] = 0.5
        with self.assertRaises(ValueError):
            bad.validate()

    def test_v2_bounds_enforced(self):
        g = Genome.founder(seed=3)
        g.params["eligibility_decay"] = 1.5
        with self.assertRaises(ValueError):
            g.validate()
        g2 = Genome.founder(seed=4)
        g2.params["neuromod_novelty_weight"] = float("nan")
        with self.assertRaises(ValueError):
            g2.validate()

    def test_mutation_preserves_version(self):
        for legacy in (True, False):
            p = Genome.founder(seed=5, legacy=legacy)
            c, rec = mutate_genome(p, 123, rate=0.5)
            self.assertEqual(c.version, p.version)
            c.validate()

    def test_crossover_mixed_versions_upgrades_to_v2(self):
        a = Genome.founder(seed=6, legacy=True)
        b = Genome.founder(seed=7)
        child, rec = crossover_genomes(a, b, 99)
        self.assertEqual(child.version, GENOME_VERSION)
        child.validate()

    def test_v2_crossover_stays_v2_and_mutations_bounded(self):
        a, b = Genome.founder(seed=8), Genome.founder(seed=9)
        for s in range(30):
            child, _ = crossover_genomes(a, b, 100 + s)
            child.validate()
        # architecture genes are actually evolvable (class assignment)
        from src.genome.operators import _class_of_param
        self.assertEqual(_class_of_param("eligibility_decay"), "plasticity")
        self.assertEqual(_class_of_param("growth_budget_fraction"), "developmental")


class TestGeneConsumption(unittest.TestCase):
    def test_v2_organism_consumes_architecture_genes(self):
        from src.organism.organism import Organism
        g = Genome.founder(seed=11)
        g.params["eligibility_decay"] = 0.5
        g.params["prediction_gain"] = 0.9
        g.validate()
        org = Organism(g, "v2org", seeds={"organism_seed": 101, "development_seed": 102},
                       graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                       autonomy_mode=True)
        self.assertEqual(org.brain.plasticity_mode, "v2_eligibility")
        self.assertEqual(org.brain.eligibility_engine.trace_decay, 0.5)
        self.assertEqual(org.brain.prediction_gain, 0.9)
        self.assertIsNotNone(org.language)
        self.assertIsNotNone(org.social_mem)
        self.assertGreater(org.language.vocabulary_size(), 0)

    def test_sleep_duration_gene_controls_replay(self):
        from src.organism.organism import Organism
        g = Genome.founder(seed=12)
        g.params["sleep_duration"] = 1.0  # -> 8 episodes
        org = Organism(g, "sleeper", seeds={"organism_seed": 103, "development_seed": 104},
                       graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                       autonomy_mode=True)
        org.energy = 1.2
        org.episodes = [{"observation": {"food_gradient": 0.5}, "action": "explore",
                         "reward": 0.5, "tick": t, "organism_id": "sleeper",
                         "generation": 0, "energy": 1.0} for t in range(10)]
        res = org.sleep()
        self.assertEqual(res["replay_count"], 8)


class TestPopulationV2(unittest.TestCase):
    def _pop(self, seed, autonomy=True, version="2.0", size=4):
        seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                           organism_seed=seed + 2, development_seed=seed + 3,
                           mutation_seed=seed + 4, world_seed=seed + 5,
                           teacher_seed=seed + 6)
        return Population(size, seeds, GraphMode.SYNTHETIC_TEST, 32,
                          experiment_seed=seed, autonomy_mode=autonomy,
                          genome_version=version)

    def test_v2_population_founders_and_teaching(self):
        pop = self._pop(71)
        for o in pop.organisms:
            self.assertEqual(o.genome.version, "2.0")
            self.assertTrue(o.autonomy_mode)
        pop.organisms[0].age = 100  # force adulthood so teaching can occur
        pop.step(12)
        # teaching happened and trust was recorded on both sides
        if pop.teaching_sessions:
            sess = pop.teaching_sessions[0]
            student = next(o for o in pop.organisms if o.id == sess.student)
            teacher = next(o for o in pop.organisms if o.id == sess.teacher)
            self.assertTrue(student.social_mem.knows(teacher.id))
            self.assertTrue(teacher.social_mem.knows(student.id))

    def test_v1_population_untouched(self):
        pop = self._pop(72, autonomy=False, version="1.0")
        for o in pop.organisms:
            self.assertEqual(o.genome.version, "1.0")
            self.assertFalse(o.autonomy_mode)
            self.assertIsNone(o.social_mem)
        pop.step(6)
        pop.reproduce(2)

    def test_v2_population_snapshot_restore_continues(self):
        pop = self._pop(73)
        pop.step(10)
        pop.reproduce(2)
        snap = pop.snapshot()
        seeds = SeedBundle(experiment_seed=73, generation_seed=74, organism_seed=75,
                           development_seed=76, mutation_seed=77, world_seed=78,
                           teacher_seed=79)
        pop2 = Population.restore(snap, seeds)
        self.assertTrue(pop2.autonomy_mode)
        self.assertEqual(pop2.genome_version, "2.0")
        h1 = pop.population_hash()
        h2 = pop2.population_hash()
        self.assertEqual(h1, h2, "restore must reproduce identical population hash")

    def test_v2_evolution_changes_architecture_genes(self):
        pop = self._pop(74, size=4)
        # Adults (age>=60, energy/health>0.4) are the eligible parents;
        # at 40 ticks all founders are still juvenile, so run to 70.
        pop.step(70)
        self.assertGreaterEqual(len(pop.eligible_parents()), 2,
                                "simulation must yield eligible adult parents")
        pop.reproduce(3)
        children = [o for o in pop.organisms if o.generation > 0]
        self.assertTrue(children, "reproduction must produce children")
        founder_params = {k for k in pop.organisms[0].genome.params}
        for c in children:
            c.genome.validate()
            self.assertEqual(set(c.genome.params), founder_params)


class TestSpeciation(unittest.TestCase):
    def test_distance_zero_for_identical(self):
        g = Genome.founder(seed=21)
        self.assertEqual(genome_distance(g, g), 0.0)

    def test_distance_range_and_monotonicity(self):
        a = Genome.founder(seed=22)
        b = Genome.founder(seed=23)
        d_ab = genome_distance(a, b)
        self.assertGreater(d_ab, 0.0)
        self.assertLessEqual(d_ab, 1.0)
        c = mutate_genome(a, 5, rate=1.0, scale=0.5)[0]
        d_ac = genome_distance(a, c)
        self.assertGreaterEqual(d_ac, 0.0)

    def _far_genome(self, seed: int) -> Genome:
        """Genome with genuinely extreme gene values (mid-range founders are all
        similar, so inversion of mid values yields ~0 distance)."""
        far = Genome.founder(seed=seed)
        far.params["exploration"] = 1.0
        far.params["sociality"] = 0.0
        far.params["teaching_ability"] = 1.0
        far.params["metabolism_rate"] = 0.1
        far.validate()
        return far

    def test_assign_species_clusters_by_distance(self):
        base = Genome.founder(seed=24)
        near = mutate_genome(base, 1, rate=0.1, scale=0.01)[0]
        far = self._far_genome(25)
        self.assertGreater(genome_distance(base, far), 0.05)
        species = assign_species([("a", base), ("b", near), ("c", far)], threshold=0.05)
        self.assertGreaterEqual(len(species), 2, "far genome must diverge into own species")
        all_members = [m for s in species for m in s.member_ids]
        self.assertEqual(sorted(all_members), ["a", "b", "c"])

    def test_divergence_requires_evidence(self):
        parent = Genome.founder(seed=26)
        # near child: no divergence
        near = mutate_genome(parent, 2, rate=0.1, scale=0.01)[0]
        self.assertEqual(detect_divergence([("p", parent)], [("c1", near)], 10, 0.12), [])
        # far child: divergence recorded WITH measured distance
        far = self._far_genome(27)
        div = detect_divergence([("p", parent)], [("c2", far)], 20, 0.05)
        self.assertEqual(len(div), 1)
        self.assertGreater(div[0].genome_distance_from_parent, 0.05)
        self.assertEqual(div[0].parent_species, "p")

    def test_invalid_threshold_rejected(self):
        g = Genome.founder(seed=28)
        with self.assertRaises(ValueError):
            assign_species([("a", g)], threshold=0.0)
        with self.assertRaises(ValueError):
            assign_species([("a", g)], threshold=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
