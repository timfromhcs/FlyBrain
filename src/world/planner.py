"""LLM World Planner for FlyBrain V10.

Implements Phase 7 Requirements:
- Translates natural language prompts into declarative structured WorldIntent.
- Resolves assets via AssetResolver and coordinates spatial placement.
- Offline-first: uses local Qwen LLM when available.
- Strict honesty: records exact inference source (LOCAL_LLM vs RULE_BASED_FALLBACK).
- Never swallows errors with bare except; captures structured failure context.
"""
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from src.assets.compiler.types import AssetClass
from src.assets.resolver import AssetResolver


class WorldPlan:
    def __init__(self, raw_prompt: str, biome: str, structures: List[Dict[str, Any]],
                 water: Optional[Dict[str, Any]] = None,
                 resolved_assets: Optional[List[Dict[str, Any]]] = None,
                 inference_provenance: Optional[Dict[str, Any]] = None):
        self.raw_prompt = raw_prompt
        self.biome = biome
        self.structures = structures
        self.water = water
        self.resolved_assets = resolved_assets or []
        self.inference_provenance = inference_provenance or {
            "source": "RULE_BASED_FALLBACK",
            "runtime_backend": "procedural_heuristic"
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_prompt": self.raw_prompt,
            "biome": self.biome,
            "structures": self.structures,
            "water": self.water,
            "resolved_assets": self.resolved_assets,
            "inference_provenance": self.inference_provenance
        }


class WorldPlanner:
    def __init__(self, resolver: Optional[AssetResolver] = None):
        self.resolver = resolver or AssetResolver()

    def _extract_intent_llm(self, prompt: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Attempts extraction via local LLM; returns (intent_dict, provenance_or_error_dict)."""
        prov: Dict[str, Any] = {
            "source": "LOCAL_LLM",
            "model_id": "Qwen/Qwen3-0.6B-GGUF",
            "quantization": "Q8_0",
            "backend": "llama_cpp",
            "sampling": {"max_tokens": 256, "temperature": 0.1},
            "error_detail": None
        }
        try:
            from src.models.manager import get_model_manager
            mgr = get_model_manager()
            if not mgr.is_available("llm"):
                prov["source"] = "UNAVAILABLE"
                prov["error_detail"] = "Local LLM weights not present or runtime unavailable"
                return None, prov

            llm = mgr.get_llm()
            sys_msg = (
                "You are the FlyBrain World Planner. Output valid JSON only with keys: "
                "'biome', 'structures' (list of {type, material, span_m, width_m, x, y, z}), 'water'."
            )
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.1
            )
            text = resp["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                return parsed, prov
            else:
                prov["error_detail"] = f"No JSON object extracted from output: {text[:80]}"
                return None, prov
        except Exception as e:
            prov["error_detail"] = f"{type(e).__name__}: {str(e)}"
            return None, prov

    def _extract_intent_rule_based(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        biome = "temperate_valley"
        if "pine" in p or "forest" in p:
            biome = "pine_forest"
        elif "mountain" in p or "rock" in p or "highland" in p:
            biome = "rocky_highlands"

        structures = []
        water = None

        if "bridge" in p:
            structures.append({
                "type": "wooden_bridge",
                "material": "wood",
                "span_m": 4.0,
                "width_m": 1.5,
                "x": 2.0,
                "y": 5.0,
                "z": 0.0
            })
            water = {"type": "stream", "x": 2.0, "y": 5.0, "width": 2.5}

        if "tower" in p or "watchtower" in p:
            structures.append({
                "type": "watchtower",
                "material": "wood",
                "sx": 2.5,
                "sy": 2.5,
                "sz": 6.0,
                "x": -5.0,
                "y": 6.0,
                "z": 0.0
            })

        if "cabin" in p or "house" in p or "settlement" in p:
            structures.append({
                "type": "log_cabin",
                "material": "wood",
                "sx": 4.0,
                "sy": 4.0,
                "sz": 2.8,
                "x": -6.0,
                "y": 8.0,
                "z": 0.0
            })

        if not structures:
            structures.append({
                "type": "marker_structure",
                "material": "stone",
                "sx": 1.0,
                "sy": 1.0,
                "sz": 1.0,
                "x": 0.0,
                "y": 6.0,
                "z": 0.0
            })

        return {"biome": biome, "structures": structures, "water": water}

    def plan_world(self, prompt: str, seed: int = 42) -> WorldPlan:
        """Section 46 & 90: Generates structured declarative plan and resolves assets."""
        intent, prov = self._extract_intent_llm(prompt)
        
        if intent and "structures" in intent:
            final_prov = prov or {"source": "LOCAL_LLM"}
            final_prov["source"] = "LOCAL_LLM"
            final_prov["is_deterministic"] = False
        else:
            intent = self._extract_intent_rule_based(prompt)
            final_prov = {
                "source": "RULE_BASED_FALLBACK",
                "model_id": "rule_based_semantic_parser",
                "runtime_backend": "procedural",
                "is_deterministic": True,
                "upstream_error": prov.get("error_detail") if prov else None
            }

        biome = intent.get("biome", "temperate_valley")
        raw_structs = intent.get("structures", [])
        water = intent.get("water")

        resolved = []
        for s in raw_structs:
            stype = s.get("type", "structure")
            params = {k: v for k, v in s.items() if k not in ("type", "x", "y", "z")}
            res = self.resolver.resolve_asset(
                semantic_name=stype,
                category=AssetClass.STRUCTURE,
                params=params,
                seed=seed
            )
            resolved.append({
                **s,
                "resolution": res
            })

        return WorldPlan(
            raw_prompt=prompt,
            biome=biome,
            structures=raw_structs,
            water=water,
            resolved_assets=resolved,
            inference_provenance=final_prov
        )
