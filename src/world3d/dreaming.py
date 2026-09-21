"""Dream + imagination pipelines (V6 phase_15/16).

dream_cycle: recent episodes + salient memories + internal state -> local
LLM dream scenario (typed JSON) -> optional SD render -> dream memory with
provenance DREAM/COUNTERFACTUAL/GENERATED_DREAM (or NARRATIVE_ONLY when no
image model is loaded — never a fake image).

imagine: LLM visual concept from memories + goal + theme -> SD render ->
imagination memory (GENERATED_IMAGE). Real observations are never
overwritten: dream/imagined records live in their own store.
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional


def _salient_episodes(agent, k: int = 6) -> List[Dict[str, Any]]:
    eps = list(getattr(agent.org, "episodes", []))
    eps.sort(key=lambda e: abs(float(e.get("reward", 0.0))), reverse=True)
    return eps[:k]


def dream_scenario_llm(text_model, agent, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    from src.prompts.render import load_templates, render, extract_json, check_output
    templates = load_templates()
    internal = {"hunger": round(agent.body3d.hunger, 2),
                "fatigue": round(agent.body3d.fatigue, 2),
                "goal": agent.body3d.goal}
    last_err = ""
    for attempt in range(2):
        t = render("dream_scenario", templates, memories=json.dumps(memories)[:1200],
                   internal=json.dumps(internal), seed=7)
        out = text_model.create_chat_completion(
            messages=[{"role": "system", "content": t["system"]},
                      {"role": "user", "content": t["user"]}],
            max_tokens=t["max_tokens"], temperature=t["temperature"], seed=11 + attempt)
        text = out["choices"][0]["message"]["content"]
        try:
            data = check_output(extract_json(text, "narrative"), t["schema"])
            break
        except ValueError as e:
            last_err = str(e)
    else:
        raise ValueError(f"LLM produced no dream JSON ({last_err})")
    assert isinstance(data.get("narrative"), str) and data["narrative"]
    data.setdefault("entities", [])
    data.setdefault("mood", "neutral")
    data.setdefault("seed", 7)
    return data


def dream_cycle(agent, text_model, image_model=None) -> Dict[str, Any]:
    mems = _salient_episodes(agent)
    scenario = dream_scenario_llm(text_model, agent, mems)
    image_path, image_prov = None, "NARRATIVE_ONLY"
    if image_model is not None:
        ctx = (f"{scenario['narrative'][:220]}, dreamlike, soft light, painting. "
               f"Mood: {scenario.get('mood', 'neutral')}.")
        res = image_model.generate(ctx, mode="FAST", seed=int(scenario["seed"]),
                                   out_path=None)
        image_path, image_prov = res["path"], "GENERATED_DREAM"
    record = {
        "id": hashlib.sha256(f"{agent.org.id}|{agent.world.tick}|{scenario['narrative'][:64]}"
                             .encode()).hexdigest()[:16],
        "tick": agent.world.tick, "organism": agent.org.id,
        "narrative": scenario["narrative"][:2000],
        "entities": list(scenario.get("entities", []))[:12],
        "mood": scenario.get("mood", "neutral"),
        "seed": int(scenario.get("seed", 7)),
        "source_memories": [e.get("tick") for e in mems],
        "image_path": image_path,
        "provenance": f"DREAM/COUNTERFACTUAL/{image_prov}",
        "replays": 0, "created_at": time.time(),
    }
    agent.spatial.db.execute(
        "INSERT OR REPLACE INTO dreams VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (record["id"], record["tick"], record["organism"], record["narrative"],
         json.dumps(record["entities"]), record["mood"], record["seed"],
         json.dumps(record["source_memories"]), record["image_path"],
         record["provenance"], 0))
    agent.spatial.db.commit()
    return record


def replay_dream(agent, dream_id: str) -> Dict[str, Any]:
    row = agent.spatial.db.execute(
        "SELECT * FROM dreams WHERE id=?", (dream_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown dream {dream_id!r}")
    agent.spatial.db.execute("UPDATE dreams SET replays=replays+1 WHERE id=?",
                             (dream_id,))
    agent.spatial.db.commit()
    cols = ["id", "tick", "organism", "narrative", "entities", "mood", "seed",
            "source_memories", "image_path", "provenance", "replays"]
    rec = dict(zip(cols, row))
    rec["replays"] += 1
    return rec


def imagine(agent, theme: str, text_model, image_model) -> Dict[str, Any]:
    """LLM visual concept from memories+goal+theme -> SD render -> memory."""
    from src.prompts.render import load_templates, render, extract_json, check_output
    templates = load_templates()
    mems = _salient_episodes(agent, k=4)
    last_err = ""
    for attempt in range(2):
        t = render("visual_concept", templates, theme=theme[:200],
                   memories=json.dumps(mems)[:900], goal=agent.body3d.goal, seed=3)
        out = text_model.create_chat_completion(
            messages=[{"role": "system", "content": t["system"]},
                      {"role": "user", "content": t["user"]}],
            max_tokens=t["max_tokens"], temperature=t["temperature"], seed=13 + attempt)
        text = out["choices"][0]["message"]["content"]
        try:
            concept = check_output(extract_json(text, "concept"), t["schema"])
            break
        except ValueError as e:
            last_err = str(e)
    else:
        raise ValueError(f"LLM produced no concept JSON ({last_err})")
    res = image_model.generate(
        f"{concept['concept'][:220]}, vivid illustration", mode="FAST",
        seed=int(concept.get("seed", 3)), out_path=None)
    rec = {"theme": theme[:200], "concept": concept["concept"][:600],
           "image_path": res["path"], "tick": agent.world.tick,
           "provenance": "GENERATED_IMAGE",
           "id": hashlib.sha256(f"{agent.org.id}|{theme[:64]}|{agent.world.tick}"
                                .encode()).hexdigest()[:16]}
    agent.spatial.db.execute(
        "INSERT OR REPLACE INTO dreams VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (rec["id"], rec["tick"], agent.org.id, rec["concept"][:2000], "[]",
         "imaginative", 0, "[]", rec["image_path"], rec["provenance"], 0))
    agent.spatial.db.commit()
    return rec
