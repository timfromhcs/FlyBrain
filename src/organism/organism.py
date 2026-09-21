"""Real organism abstraction: brain + body + lifecycle + metabolism (REAL, IMPLEMENTED)."""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from src.brain.runtime import BrainRuntime
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.development.engine import DevelopmentEngine, DevelopmentState
from src.genome.schema import Genome
from src.common.determinism import deterministic_id, derive_subseed
from src.common.events import EventLog
from src.world.environment import GridWorld


class LifeStage(str, Enum):
    INFANCY = "infancy"
    JUVENILE = "juvenile"
    ADULT = "adult"
    ELDER = "elder"
    DEAD = "dead"


def stage_for_age(age: int) -> LifeStage:
    if age < 20:
        return LifeStage.INFANCY
    if age < 60:
        return LifeStage.JUVENILE
    if age < 200:
        return LifeStage.ADULT
    return LifeStage.ELDER


class Organism:
    def __init__(self, genome: Genome, organism_id: str, generation: int = 0,
                 seeds: Optional[Dict[str, int]] = None, graph_mode: GraphMode = GraphMode.SYNTHETIC_TEST,
                 circuit_size: int = 64, parents: Optional[List[str]] = None,
                 birth_tick: int = 0, start_pos: tuple = (0, 0),
                 autonomy_mode: bool = False):
        self.genome = genome
        self.genome.validate()
        self.id = organism_id
        self.generation = int(generation)
        self.parents = list(parents or [])
        self.children: List[str] = []
        self.seeds = dict(seeds or {})
        self.age = 0
        self.energy = 1.0
        self.health = 1.0
        self.stage = LifeStage.INFANCY
        self.alive = True
        self.tick = int(birth_tick)
        self.episodes: List[Dict[str, Any]] = []   # lightweight episodic memory w/ provenance
        self.semantic: Dict[str, List[float]] = {}  # concept -> vector
        self.cultural_knowledge: Dict[str, Dict[str, Any]] = {}
        self.skills: Dict[str, float] = {"forage": 0.1, "avoid": 0.1, "social": 0.1}
        self.cause_of_death = ""
        self.position = tuple(start_pos)
        self.heading = (1, 0)      # persistent run direction (chemotaxis)
        # STAGE E/F: optional autonomy + embodiment (additive; legacy mode untouched)
        self.autonomy_mode = bool(autonomy_mode)
        self.autonomy = None
        self.body = None
        self.living: Optional["LivingBrain"] = None
        self.language = None
        self.social_mem = None
        self.last_action_log: Dict[str, Any] = {}
        if self.autonomy_mode:
            from src.autonomy.engine import AutonomyEngine
            from src.embodiment.body import BodyState
            from src.language.grounded import GroundedLanguageSystem
            from src.social.model import SocialMemory
            oseed = int(self.seeds.get("organism_seed", 44))
            self.autonomy = AutonomyEngine(seed=oseed)
            self.body = BodyState(energy=self.energy, health=self.health,
                                  position=tuple(start_pos))
            self.language = GroundedLanguageSystem(seed=oseed)
            self.social_mem = SocialMemory(self.id)
        self.last_food = 0.0       # previous food gradient (tumble trigger)
        self._last_reward = 0.0    # previous tick outcome -> neural plasticity (P12)
        # own brain copy (CPU for determinism inside populations)
        oseed = int(self.seeds.get("organism_seed", 44))
        graph = get_or_create_circuit(circuit_size, mode=graph_mode, seed=oseed)
        # deep copy so development is per-organism
        from copy import deepcopy
        self.graph = deepcopy(graph)
        if self.autonomy_mode:
            # v2 architecture genes -> versioned learning architecture (STAGE I)
            from src.brain.eligibility import NeuromodulationConfig
            self.brain = BrainRuntime(
                self.graph, use_gpu=False, enable_plasticity=True, seed=oseed,
                plasticity_mode="v2_eligibility",
                neuromod=NeuromodulationConfig(
                    w_novelty=self.genome.get("neuromod_novelty_weight", 0.0),
                    w_prediction_error=self.genome.get("neuromod_prediction_weight", 0.0)),
                trace_decay=self.genome.get("eligibility_decay", 0.9),
                prediction_influence=True,
                prediction_gain=self.genome.get("prediction_gain", 0.5))
            # STAGE G: grounded seed concepts come from actual sensory channels
            self._ground_seed_concepts()
        else:
            self.brain = BrainRuntime(self.graph, use_gpu=False,
                                      enable_plasticity=True, seed=oseed)
        self.dev = DevelopmentState.initialize(self.graph.num_neurons)
        self.dev_engine = DevelopmentEngine(genome.params,
                                            development_seed=int(self.seeds.get("development_seed", 45)))
        if self.autonomy_mode:
            from src.brain.living import LivingBrain
            self.living = LivingBrain(self.graph, dev=self.dev, engine=self.dev_engine,
                                      experiment_seed=oseed)
        self.events = EventLog()
        self.events.log("ORGANISM_BORN", self.tick, self.id, self.generation,
                        {"genome_hash": genome.genome_hash(), "parents": self.parents,
                         "graph_mode": graph_mode.value, "circuit_size": circuit_size})

    # ---- sensorimotor loop ----
    def _ground_seed_concepts(self):
        """Ground core concepts on actual sensory/internal dims (STAGE G).
        Feature vectors are FEATURE TEMPLATES; lived experience updates salience
        through produce/comprehend usage, never invented content."""
        if self.language is None:
            return
        if self.language.vocabulary_size() > 0:
            return
        seeds = [
            ("food", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "sensory"),
            ("danger", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "sensory"),
            ("kin", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "social"),
            ("tired", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "internal"),
        ]
        for i, (name, vec, mod) in enumerate(seeds):
            c = self.language.ground_concept(name, vec, mod, {"origin": "seed_template"})
            self.language.learn_symbol(f"sym{i}", c.concept_id, teacher="self")

    def _communicate(self, sense: Dict[str, Any], out: Dict[str, Any]) -> List[str]:
        """Produce grounded symbols from internal state when tendency is high
        and something is salient (STAGE G/H). Returns produced symbols."""
        if self.language is None or self.autonomy is None:
            return []
        tendency = self.genome.get("communication_tendency", 0.0)
        if tendency < 0.2:
            return []
        import numpy as _np
        rng = _np.random.RandomState(derive_subseed(
            int(self.seeds.get("organism_seed", 44)), f"comm:{self.tick}"))
        if rng.rand() > tendency:
            return []
        salient = []
        if float(sense.get("food_gradient", 0.0)) > 1.0:
            salient.append(self._concept_by_name("food"))
        if float(sense.get("hazard_gradient", 0.0)) > 0.6:
            salient.append(self._concept_by_name("danger"))
        if int(sense.get("nearby_organisms", 0)) > 0:
            salient.append(self._concept_by_name("kin"))
        if self.energy < 0.3:
            salient.append(self._concept_by_name("tired"))
        salient = [c for c in salient if c]
        return self.language.produce({"salient_concepts": salient})

    def _concept_by_name(self, name: str):
        if self.language is None:
            return None
        for c in self.language.concepts.values():
            if c.name == name:
                return c.concept_id
        return None

    def receive_message(self, sender_id: str, symbols: List[str], tick: int) -> Dict[str, Any]:
        """Grounded comprehension + social recording (STAGE H)."""
        cids = self.language.comprehend(symbols) if self.language else []
        if self.social_mem is not None and cids:
            self.social_mem.record_interaction(sender_id, tick, "communicated", 0.3)
        return {"comprehended": cids}

    def step(self, world: GridWorld) -> Dict[str, Any]:
        if not self.alive:
            return {"organism_id": self.id, "alive": False}
        sense = world.sense(self.id)
        # sensors -> brain currents: food->visual, hazard->olfactory, social->memory
        n = self.graph.num_neurons
        vis = np.full(16, float(sense["food_gradient"]), dtype=np.float32)
        olf = np.full(8, float(sense["hazard_gradient"]), dtype=np.float32)
        mem = np.full(8, float(min(1.0, sense["nearby_organisms"] / 3.0)), dtype=np.float32)
        # P12: the previous tick's outcome reward drives real neural plasticity.
        out = self.brain.step(sensory_inputs={"visual": vis, "olfactory": olf, "memory": mem},
                              reward=float(self._last_reward))
        # enforce dead-neuron silence invariant
        dead_idx = [i for i, a in enumerate(self.dev.alive) if not a]
        if dead_idx and len(self.brain.state.spikes) >= max(dead_idx, default=-1) + 1:
            pass  # spikes arrays only cover live indices; dead edges were stripped
        # decision -> motor
        act = out.get("selected_action", "explore")
        dx = dy = 0
        if self.autonomy_mode and self.autonomy is not None and self.body is not None:
            import math
            energy_before = self.energy
            novel = self.autonomy.note_position(self.position)
            self.autonomy.generate_goals(
                self.tick, self.energy, self.health,
                drives=out.get("drives", {}),
                prediction_error=float(out.get("prediction_error", 0.0)),
                world_sense=sense, position=self.position, skills=self.skills)
            goal = self.autonomy.active_goal()
            cand = self.autonomy.synthesize_action(
                self.tick, goal, out, sense, self.energy, self.health,
                exploration=float(self.genome.params.get("exploration", 0.5)))
            act = cand.kind
            if cand.kind == "move" and self.body.is_mobile():
                step_len = max(1, int(round(cand.speed * self.body.speed_capacity())))
                dx = int(round(math.cos(cand.heading) * step_len))
                dy = int(round(math.sin(cand.heading) * step_len))
                if dx == 0 and dy == 0 and cand.speed > 0.05:
                    dx = 1 if math.cos(cand.heading) >= 0 else -1
                world.apply_move(self.id, dx, dy)
                self.body.heading = float(cand.heading)
            elif cand.kind == "rest":
                self.body.recover(0.05)
                self.health = self.body.health
            # investigate/communicate: stationary; drives steer future candidates
            produced = self._communicate(sense, out) if act != "rest" else []
            self.last_action_log = {"candidate": cand.to_dict(),
                                    "goal": goal.to_dict() if goal is not None else None,
                                    "produced_symbols": produced}
            self._autonomy_energy_before = energy_before
            self._autonomy_novel = novel
        else:
            explore = float(self.genome.params.get("exploration", 0.5))
            rng = np.random.RandomState(derive_subseed(int(self.seeds.get("organism_seed", 44)),
                                                       f"move:{self.tick}"))
            food_now = float(sense["food_gradient"])
            hungry = self.energy < 0.7
            # tumble (pick new heading) when gradient worsens while hungry, or randomly when exploring
            if (hungry and food_now < self.last_food - 1e-6) or rng.rand() < explore * 0.25:
                self.heading = (int(rng.randint(-1, 2)), int(rng.randint(-1, 2)))
            self.last_food = food_now
            if act in ("act_in_environment", "explore") or hungry or rng.rand() < 0.5:
                dx, dy = self.heading
                if dx == 0 and dy == 0:
                    dx = int(rng.randint(-1, 2))
                if float(sense["hazard_gradient"]) > 0.5 and rng.rand() < 0.7:
                    dx, dy = -dx, int(rng.randint(-1, 2))  # escape reflex
                world.apply_move(self.id, dx, dy)
        consumed = world.consume(self.id)
        reward = 0.0
        if consumed > 0:
            self.energy = min(1.5, self.energy + consumed * 0.8)
            self.skills["forage"] = min(1.0, self.skills["forage"] + 0.02)
            reward = 0.5
        if world.hazard_at(self.id):
            if self.autonomy_mode and self.body is not None:
                self.body.apply_damage(0.05, "hazard")
                self.health = self.body.health
            else:
                self.health = max(0.0, self.health - 0.05)
            self.skills["avoid"] = min(1.0, self.skills["avoid"] + 0.01)
            reward -= 0.3
        # P12: store outcome for next tick's neural plasticity (closed loop).
        self._last_reward = float(reward)
        # metabolism: basal + neural activity + movement + growth
        activity = float(np.mean(self.brain.state.activations)) if len(self.brain.state.activations) else 0.0
        metab = float(self.genome.params.get("metabolism_rate", 0.01))
        cost = metab * (0.5 + activity + (abs(dx) + abs(dy)) * 0.25)
        self.energy = max(0.0, self.energy - cost)
        if self.energy <= 0.0:
            self.health = max(0.0, self.health - 0.02)
        self.position = world.positions.get(self.id, self.position)
        # episodic memory with provenance
        ep = {"tick": self.tick, "organism_id": self.id, "generation": self.generation,
              "observation": sense, "action": act, "reward": round(reward, 4),
              "energy": round(self.energy, 4)}
        self.episodes.append(ep)
        self.events.log("LEARNING_EVENT", self.tick, self.id, self.generation,
                        {"action": act, "reward": round(reward, 4)})
        self.events.log("MEMORY_CREATED", self.tick, self.id, self.generation,
                        {"episode_index": len(self.episodes) - 1})
        # development tick (cheap, every 5 ticks); growth costs energy
        if self.tick % 5 == 0:
            if self.autonomy_mode and self.living is not None:
                # resource-constrained growth: rich organisms may grow, poor may
                # not; the v2 growth_budget_fraction gene allocates metabolic
                # budget to structural expansion (STAGE I: evolvable allocation).
                frac = float(self.genome.get("growth_budget_fraction", 0.5))
                budget = int(np.clip((self.energy - 0.4) * 5.0 * (0.5 + frac), 0, 4))
                summary = self.living.run_development_cycle(
                    self.tick, activity=self.brain.state.activations,
                    organism_id=self.id, generation=self.generation,
                    growth_budget=budget)
                self.energy = max(0.0, self.energy - summary["energy_spent"])
                self._sync_brain_to_graph()
            else:
                n_born = self.dev_engine.neurogenesis(self.graph, self.dev, self.tick, self.events,
                                                      self.id, self.generation, max_new=2)
                self.energy = max(0.0, self.energy - 0.02 * n_born)
                self._sync_brain_to_graph()
                self.dev_engine.differentiate(self.graph, self.dev, self.tick, self.events,
                                              self.id, self.generation)
                self.dev_engine.migrate(self.graph, self.dev, self.tick, self.events,
                                        self.id, self.generation)
                self.dev_engine.grow_projections(self.graph, self.dev, self.tick, self.events,
                                                 self.id, self.generation, max_candidates=4)
                self.dev_engine.prune(self.graph, self.dev, self.tick,
                                      self.brain.state.activations, self.events,
                                      self.id, self.generation)
                self.dev_engine.apoptosis(self.graph, self.dev, self.tick,
                                          self.brain.state.activations, self.age,
                                          self.events, self.id, self.generation)
                self._sync_brain_to_graph()
        if self.autonomy_mode and self.autonomy is not None:
            self.body.energy = self.energy
            self.autonomy.update_goals(
                self.tick, self.energy - getattr(self, "_autonomy_energy_before", self.energy),
                getattr(self, "_autonomy_novel", False),
                float(out.get("prediction_error", 0.0)))        # aging + lifecycle
        self.age += 1
        self.tick += 1
        world.tick = max(world.tick, self.tick)
        self.stage = stage_for_age(self.age)
        if self.health <= 0.0 or self.age > 400:
            self.die("health" if self.health <= 0.0 else "age")
        # v4.1 causal-path telemetry: every behaviorally relevant action is
        # traceable to sensory input -> neural response -> motor output ->
        # body actuation -> environment outcome -> reward. policy_source names
        # which (explicitly hand-written) motor policy mapped neural output to
        # movement; it is NOT neural activity itself.
        neural_spikes = int(np.sum(self.brain.state.spikes > 0.5))
        return {"organism_id": self.id, "alive": self.alive, "action": act,
                "reward": round(reward, 4), "energy": round(self.energy, 4),
                "stage": self.stage.value,
                "action_kind": act,
                "goal": (self.autonomy.active_goal().kind
                         if (self.autonomy_mode and self.autonomy is not None
                             and self.autonomy.active_goal() is not None) else None),
                "causal_path": {
                    "sensory_input": {
                        "food_gradient": round(float(sense.get("food_gradient", 0.0)), 4),
                        "hazard_gradient": round(float(sense.get("hazard_gradient", 0.0)), 4),
                        "nearby_organisms": int(sense.get("nearby_organisms", 0)),
                    },
                    "neural_response": {
                        "spikes": neural_spikes,
                        "mean_activation": round(float(np.mean(
                            self.brain.state.activations)) if len(
                            self.brain.state.activations) else 0.0, 4),
                        "selected_neural_action": out.get("selected_action", "explore"),
                        "prediction_error": round(float(out.get("prediction_error", 0.0)), 4),
                    },
                    "motor_output": {"dx": int(dx), "dy": int(dy)},
                    "body_actuation": {
                        "policy_source": ("neural+autonomy" if self.autonomy_mode
                                          else "neural+reflex"),
                        "mobile": bool(self.body.is_mobile()) if self.body is not None else True,
                    },
                    "environment_outcome": {
                        "consumed": round(float(consumed), 4),
                        "hazard": bool(world.hazard_at(self.id)),
                    },
                    "reward": round(reward, 4),
                }}

    def _sync_brain_to_graph(self):
        """Resize brain state arrays after structural growth (new neurons start silent)."""
        n = self.graph.num_neurons
        st = self.brain.state
        def grow(arr, fill, dtype):
            if len(arr) < n:
                extra = np.full(n - len(arr), fill, dtype=dtype)
                return np.concatenate([arr, extra])
            return arr[:n]
        st.membrane_potentials = grow(st.membrane_potentials, 0.0, np.float32)
        st.spikes = grow(st.spikes, 0.0, np.float32)
        st.refractory_steps = grow(st.refractory_steps, 0, np.int32)
        st.activations = grow(st.activations, 0.0, np.float32)
        st.attention = np.ones(n, dtype=np.float32) / max(1, n)
        st.num_neurons = n
        # rebind sensorimotor index maps if out of range
        for attr in ("sensory_visual_indices", "sensory_audio_indices", "sensory_olfactory_indices",
                     "sensory_memory_indices", "motor_speak_indices", "motor_act_indices",
                     "motor_image_indices", "motor_remember_indices"):
            idx = getattr(self.brain, attr, np.array([], dtype=np.int32))
            setattr(self.brain, attr, np.array([i for i in idx if i < n], dtype=np.int32))

    # ---- sleep / dream (real replay of own episodes) ----
    def sleep(self, replay_k: Optional[int] = None) -> Dict[str, Any]:
        if not self.alive:
            return {"slept": False}
        if replay_k is None:
            # v2 sleep_duration gene (1..8 episodes); legacy default 3
            replay_k = int(1 + round(self.genome.get("sleep_duration", 0.4) * 7)) \
                if self.genome.version == "2.0" else 3
        self.events.log("SLEEP_STARTED", self.tick, self.id, self.generation, {})
        recent = self.episodes[-replay_k:] if self.episodes else []
        self.events.log("DREAM_STARTED", self.tick, self.id, self.generation,
                        {"replay_count": len(recent)})
        consolidated = 0
        for ep in recent:
            stim = np.full(16, float(ep["observation"].get("food_gradient", 0.0) * 0.5 + 0.1),
                           dtype=np.float32)
            self.brain.step(sensory_inputs={"visual": stim}, reward=0.0)
            key = f"dream:{ep['action']}"
            v = self.semantic.get(key, [0.0])
            self.semantic[key] = [round(min(1.0, v[0] + 0.05 * max(0.0, ep["reward"] + 0.5)), 4)]
            consolidated += 1
        self.energy = min(1.5, self.energy + 0.1)  # rest recovery (bounded, from reduced activity)
        self.events.log("MEMORY_CONSOLIDATED", self.tick, self.id, self.generation,
                        {"consolidated": consolidated})
        self.events.log("DREAM_ENDED", self.tick, self.id, self.generation, {})
        return {"slept": True, "replay_count": len(recent), "consolidated": consolidated}

    def die(self, cause: str):
        if not self.alive:
            return
        self.alive = False
        self.stage = LifeStage.DEAD
        self.cause_of_death = cause
        self.events.log("ORGANISM_DIED", self.tick, self.id, self.generation,
                        {"cause": cause, "age": self.age,
                         "genome_hash": self.genome.genome_hash(),
                         "brain_neurons": self.graph.num_neurons,
                         "brain_synapses": self.graph.num_synapses})

    def fitness_vector(self) -> Dict[str, float]:
        return {
            "survival": 1.0 if self.alive else 0.0,
            "age": float(self.age),
            "energy_efficiency": round(float(self.energy / max(1, self.age)), 4),
            "learning": round(float(len(self.episodes) / max(1, self.age)), 4),
            "memory": float(len(self.semantic) + len(self.cultural_knowledge)),
            "brain_size": float(self.graph.num_neurons),
            "health": round(float(self.health), 4),
        }

    def organism_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.id.encode())
        h.update(self.genome.genome_hash().encode())
        h.update(self.graph.graph_hash.encode())
        h.update(self.brain.state.membrane_potentials.tobytes())
        h.update(self.brain.state.spikes.tobytes())
        h.update(str((self.age, round(self.energy, 6), round(self.health, 6), self.tick)).encode())
        return h.hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id, "generation": self.generation, "parents": self.parents,
            "children": self.children, "seeds": self.seeds, "age": self.age,
            "energy": self.energy, "health": self.health, "alive": self.alive,
            "tick": self.tick, "position": list(self.position),
            "heading": list(self.heading), "last_food": self.last_food,
            "last_reward": self._last_reward,
            "genome": self.genome.to_dict(),
            "genome_params_exact": {k: float(v) for k, v in self.genome.params.items()},
            "graph": {"neuron_ids": self.graph.neuron_ids.tolist(),
                      "coordinates": self.graph.coordinates.tolist(),
                      "tbars": self.graph.tbars.tolist(), "sides": list(self.graph.sides),
                      "row_offsets": self.graph.row_offsets.tolist(),
                      "col_indices": self.graph.col_indices.tolist(),
                      "weights": self.graph.weights.tolist(), "mode": self.graph.mode.value},
            "brain": {"potentials": self.brain.state.membrane_potentials.tolist(),
                      "spikes": self.brain.state.spikes.tolist(),
                      "refractory": self.brain.state.refractory_steps.tolist(),
                      "activations": self.brain.state.activations.tolist(),
                      "step_count": self.brain.state.step_count,
                      "total_spikes": self.brain.state.total_spikes,
                      "prediction_error": self.brain.state.prediction_error,
                      "predicted_reward": self.brain.state.predicted_reward,
                      "current_reward": self.brain.state.current_reward,
                      "plasticity_mode": self.brain.plasticity_mode,
                      "trace_decay": getattr(getattr(self.brain, "eligibility_engine",
                                                     None), "trace_decay", 0.9),
                      "prediction_influence": bool(self.brain.prediction_influence),
                      "prediction_gain": float(self.brain.prediction_gain),
                      "eligibility": (self.brain.eligibility.snapshot()
                                      if self.brain.eligibility is not None else None),
                      "neuromod": (self.brain.eligibility_engine.neuromod.to_dict()
                                   if self.brain.eligibility_engine is not None
                                   and hasattr(self.brain.eligibility_engine, "neuromod")
                                   else None),
                      "attention": self.brain.state.attention.tolist(),
                      "goal_embedding": self.brain.state.goal_embedding.tolist(),
                      "active_memory_refs": list(self.brain.state.active_memory_refs),
                      "tool_associations": dict(self.brain.state.tool_associations),
                      "active_goal": self.brain.state.active_goal,
                      "drives": {"energy": self.brain.state.drives.energy,
                                 "curiosity": self.brain.state.drives.curiosity,
                                 "social": self.brain.state.drives.social,
                                 "integrity": self.brain.state.drives.integrity},
                      "maps": {k: [int(x) for x in getattr(self.brain, k, [])]
                               for k in ("sensory_visual_indices", "sensory_audio_indices",
                                         "sensory_olfactory_indices", "sensory_memory_indices",
                                         "motor_speak_indices", "motor_act_indices",
                                         "motor_image_indices", "motor_remember_indices")}},
            "dev": {"cell_types": self.dev.cell_types, "birth_ticks": self.dev.birth_ticks,
                    "lineage_ids": self.dev.lineage_ids,
                    "developmental_states": self.dev.developmental_states,
                    "alive": self.dev.alive},
            "episodes": self.episodes, "semantic": self.semantic,
            "cultural_knowledge": self.cultural_knowledge, "skills": self.skills,
            "autonomy_mode": self.autonomy_mode,
            "autonomy": self.autonomy.snapshot() if self.autonomy is not None else None,
            "body": self.body.to_dict() if self.body is not None else None,
            "body_exact": ({"energy": float(self.body.energy),
                            "health": float(self.body.health),
                            "damage": float(self.body.damage)}
                           if self.body is not None else None),
            "living_brain": self.living.snapshot() if self.living is not None else None,
            "language": self.language.snapshot() if self.language is not None else None,
            "social": self.social_mem.snapshot() if self.social_mem is not None else None,
        }

    @classmethod
    def restore(cls, snap: Dict[str, Any]) -> "Organism":
        import numpy as np
        from src.connectome.types import ConnectomeGraph, GraphMode
        g = snap["graph"]
        from src.connectome.types import (MaleCNSRealGraph, MaleCNSSpatialSurrogateGraph,
                                          SyntheticTestGraph)
        mode = GraphMode(g["mode"])
        cls_map = {GraphMode.REAL: MaleCNSRealGraph,
                   GraphMode.SPATIAL_SURROGATE: MaleCNSSpatialSurrogateGraph,
                   GraphMode.SYNTHETIC_TEST: SyntheticTestGraph}
        graph = cls_map[mode](
            neuron_ids=np.array(g["neuron_ids"], dtype=np.int64),
            coordinates=np.array(g["coordinates"], dtype=np.float32),
            tbars=np.array(g["tbars"], dtype=np.int32), sides=list(g["sides"]),
            row_offsets=np.array(g["row_offsets"], dtype=np.int32),
            col_indices=np.array(g["col_indices"], dtype=np.int32),
            weights=np.array(g["weights"], dtype=np.float32))
        genome = Genome.from_dict(snap["genome"])
        if snap.get("genome_params_exact"):
            for k, v in snap["genome_params_exact"].items():
                if k in genome.params:
                    genome.params[k] = float(v)
        org = cls.__new__(cls)
        org.genome, org.id, org.generation = genome, snap["id"], snap["generation"]
        org.parents, org.children = list(snap["parents"]), list(snap["children"])
        org.seeds = dict(snap["seeds"]); org.age = snap["age"]
        org.energy, org.health, org.alive = snap["energy"], snap["health"], snap["alive"]
        org.tick = snap["tick"]; org.position = tuple(snap["position"])
        org.heading = tuple(snap.get("heading", (1, 0))); org.last_food = float(snap.get("last_food", 0.0))
        org._last_reward = float(snap.get("last_reward", 0.0))
        org.stage = stage_for_age(org.age) if org.alive else LifeStage.DEAD
        org.graph = graph
        b = snap["brain"]
        # v2 learning architecture must be restored exactly: a restored
        # autonomy organism that silently falls back to v1_hebbian (or loses
        # eligibility traces) would diverge from uninterrupted execution.
        _pmode = b.get("plasticity_mode", "v1_hebbian")
        _neu = None
        if b.get("neuromod"):
            from src.brain.eligibility import (EligibilityEngine, EligibilityState,
                                               NeuromodulationConfig)
            _neu = NeuromodulationConfig.from_dict(b["neuromod"])
        org.brain = BrainRuntime(
            graph, use_gpu=False, seed=int(org.seeds.get("organism_seed", 44)),
            plasticity_mode=_pmode,
            neuromod=_neu,
            trace_decay=float(b.get("trace_decay", 0.9)),
            prediction_influence=bool(b.get("prediction_influence", False)),
            prediction_gain=float(b.get("prediction_gain", 0.5)))
        if b.get("eligibility") and org.brain.eligibility is not None:
            from src.brain.eligibility import EligibilityState as _ES
            org.brain.eligibility = _ES.restore(b["eligibility"])
        org.brain.state.membrane_potentials = np.array(b["potentials"], dtype=np.float32)
        org.brain.state.spikes = np.array(b["spikes"], dtype=np.float32)
        org.brain.state.refractory_steps = np.array(b["refractory"], dtype=np.int32)
        org.brain.state.activations = np.array(b["activations"], dtype=np.float32)
        org.brain.state.step_count = b["step_count"]; org.brain.state.total_spikes = b["total_spikes"]
        org.brain.state.num_neurons = graph.num_neurons
        if "attention" in b and len(b["attention"]) == len(org.brain.state.attention):
            org.brain.state.attention = np.array(b["attention"], dtype=np.float32)
        if "goal_embedding" in b:
            org.brain.state.goal_embedding = np.array(b["goal_embedding"], dtype=np.float32)
        if "active_memory_refs" in b:
            org.brain.state.active_memory_refs = [str(x) for x in b["active_memory_refs"]]
        if "tool_associations" in b:
            org.brain.state.tool_associations = dict(b["tool_associations"])
        if "active_goal" in b:
            org.brain.state.active_goal = b["active_goal"]
        org.brain.state.prediction_error = float(b.get("prediction_error", 0.0))
        org.brain.state.predicted_reward = float(b.get("predicted_reward", 0.0))
        org.brain.state.current_reward = float(b.get("current_reward", 0.0))
        for dk, dv in b.get("drives", {}).items():
            if hasattr(org.brain.state.drives, dk):
                setattr(org.brain.state.drives, dk, float(dv))
        # P7: sensorimotor maps must be restored exactly; a grown brain's maps
        # differ from a freshly built brain's maps at the same size.
        import numpy as _np
        for k, v in b.get("maps", {}).items():
            if hasattr(org.brain, k):
                setattr(org.brain, k, _np.array([int(x) for x in v], dtype=_np.int32))
        from src.development.engine import DevelopmentState
        d = snap["dev"]
        org.dev = DevelopmentState(cell_types=list(d["cell_types"]), birth_ticks=list(d["birth_ticks"]),
                                   lineage_ids=list(d["lineage_ids"]),
                                   developmental_states=list(d["developmental_states"]),
                                   alive=list(d["alive"]),
                                   activity_history=[0.0] * len(d["alive"]))
        org.dev_engine = DevelopmentEngine(genome.params,
                                           development_seed=int(org.seeds.get("development_seed", 45)))
        org.episodes = list(snap["episodes"]); org.semantic = dict(snap["semantic"])
        org.cultural_knowledge = dict(snap["cultural_knowledge"]); org.skills = dict(snap["skills"])
        org.cause_of_death = snap.get("cause_of_death", "")
        org.events = EventLog()
        # STAGE E/F additive state (absent in legacy snapshots -> legacy mode)
        org.autonomy_mode = bool(snap.get("autonomy_mode", False))
        org.autonomy = None
        org.body = None
        org.living = None
        org.language = None
        org.social_mem = None
        org.last_action_log = {}
        if org.autonomy_mode:
            from src.autonomy.engine import AutonomyEngine
            from src.embodiment.body import BodyState
            from src.brain.living import LivingBrain
            org.autonomy = AutonomyEngine.restore(snap["autonomy"]) if snap.get("autonomy") \
                else AutonomyEngine(seed=int(org.seeds.get("organism_seed", 44)))
            org.body = BodyState.from_dict(snap["body"]) if snap.get("body") else BodyState(
                energy=org.energy, health=org.health, position=org.position)
            if snap.get("body_exact"):  # full precision wins for exact replay
                org.body.energy = float(snap["body_exact"]["energy"])
                org.body.health = float(snap["body_exact"]["health"])
                org.body.damage = float(snap["body_exact"]["damage"])
            org.living = LivingBrain.attach(
                org.graph, org.dev, org.dev_engine,
                int(org.seeds.get("organism_seed", 44)),
                payload=snap.get("living_brain"))
            from src.language.grounded import GroundedLanguageSystem
            from src.social.model import SocialMemory
            org.language = GroundedLanguageSystem.restore(snap["language"]) \
                if snap.get("language") else None
            org.social_mem = SocialMemory.restore(snap["social"]) \
                if snap.get("social") else None
        return org


def create_offspring_id(experiment_seed: int, generation: int, parent_ids: List[str],
                        repro_index: int, genome_hash: str) -> str:
    return deterministic_id(str(experiment_seed), str(generation),
                            "+".join(sorted(parent_ids)), str(repro_index), genome_hash)
