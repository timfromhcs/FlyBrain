"""World/spatial/visual memory: local SQLite + local RAG (V6 phase_08/13).

Stores: discovered places, remembered paths, visual frames (geometric
panoramas, NEVER called photos), entity sightings. Retrieval is two-tier:
lexical FTS (always local) + MiniLM embeddings when the model is present
(models/embedding/, offline-capable after first download; otherwise the
semantic tier reports UNAVAILABLE and lexical carries on).
"""
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS places(
  id TEXT PRIMARY KEY, label TEXT, x REAL, y REAL, first_tick INTEGER,
  visits INTEGER DEFAULT 1, note TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS paths(
  name TEXT PRIMARY KEY, waypoints TEXT, uses INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS frames(
  frame_hash TEXT PRIMARY KEY, tick INTEGER, x REAL, y REAL, yaw REAL,
  entities TEXT, internal TEXT, event TEXT);
CREATE TABLE IF NOT EXISTS sightings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT, tick INTEGER,
  x REAL, y REAL, dist REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS places_fts USING fts5(label, note);
CREATE TABLE IF NOT EXISTS dreams(
  id TEXT PRIMARY KEY, tick INTEGER, organism TEXT, narrative TEXT,
  entities TEXT, mood TEXT, seed INTEGER, source_memories TEXT,
  image_path TEXT, provenance TEXT, replays INTEGER DEFAULT 0);
"""


class SpatialMemory:
    def __init__(self, path: str = "diagnostics/world_memory.db"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self._embedder = None
        self._embedder_state = "UNCHECKED"

    # ---- places ----
    def record_place(self, label: str, x: float, y: float, tick: int,
                     note: str = "") -> str:
        pid = hashlib.sha256(f"{label}|{round(x,1)}|{round(y,1)}".encode()).hexdigest()[:12]
        cur = self.db.execute("SELECT visits FROM places WHERE id=?", (pid,)).fetchone()
        if cur:
            self.db.execute("UPDATE places SET visits=visits+1 WHERE id=?", (pid,))
        else:
            self.db.execute("INSERT INTO places VALUES(?,?,?,?,?,?,?)",
                            (pid, label, x, y, tick, 1, note))
            self.db.execute("INSERT INTO places_fts(rowid,label,note) VALUES("
                            "last_insert_rowid(),?,?)", (label, note))
        self.db.commit()
        return pid

    def close(self) -> None:
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass

    # ---- snapshot / restore (save determinism: memory IS agent state) ----
    def snapshot(self) -> Dict[str, Any]:
        return {
            "places": self.db.execute("SELECT * FROM places").fetchall(),
            "paths": self.db.execute("SELECT * FROM paths").fetchall(),
            "frames": self.db.execute("SELECT * FROM frames").fetchall(),
            "sightings": self.db.execute("SELECT * FROM sightings").fetchall(),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        cur = self.db.cursor()
        cur.execute("DELETE FROM places")
        cur.execute("DELETE FROM places_fts")
        cur.execute("DELETE FROM paths")
        cur.execute("DELETE FROM frames")
        cur.execute("DELETE FROM sightings")
        for r in snap.get("places", []):
            cur.execute("INSERT INTO places VALUES(?,?,?,?,?,?,?)", tuple(r))
            cur.execute("INSERT INTO places_fts(rowid,label,note) VALUES("
                        "last_insert_rowid(),?,?)", (r[1], r[6]))
        for r in snap.get("paths", []):
            cur.execute("INSERT INTO paths VALUES(?,?,?)", tuple(r))
        for r in snap.get("frames", []):
            cur.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?)", tuple(r))
        for r in snap.get("sightings", []):
            cur.execute("INSERT INTO sightings VALUES(?,?,?,?,?,?)", tuple(r))
        self.db.commit()

    def where_seen(self, entity: str) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT entity,tick,x,y,dist FROM sightings WHERE entity=? ORDER BY tick DESC LIMIT 10",
            (entity,)).fetchall()
        return [{"entity": r[0], "tick": r[1], "x": r[2], "y": r[3], "dist": r[4]}
                for r in rows]

    def record_sighting(self, entity: str, tick: int, x: float, y: float, dist: float) -> None:
        self.db.execute("INSERT INTO sightings(entity,tick,x,y,dist) VALUES(?,?,?,?,?)",
                        (entity, tick, x, y, dist))
        self.db.commit()

    def record_frame(self, obs: Dict[str, Any], internal: Dict[str, Any],
                     event: str = "") -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?,?,?,?)",
            (obs["frame_hash"], obs["tick"], obs["eye"][0], obs["eye"][1], obs["yaw"],
             json.dumps(obs["entities"], sort_keys=True),
             json.dumps(internal, sort_keys=True), event))
        self.db.commit()

    def record_path(self, name: str, waypoints: List) -> None:
        self.db.execute("INSERT OR REPLACE INTO paths VALUES(?,?,1)",
                        (name, json.dumps(waypoints)))
        self.db.commit()

    # ---- retrieval: lexical (always) ----
    def search_places(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT label,note FROM places_fts WHERE places_fts MATCH ? LIMIT ?",
            (query, limit)).fetchall()
        out = []
        for label, note in rows:
            r = self.db.execute("SELECT x,y,visits FROM places WHERE label=?",
                                (label,)).fetchone()
            if r:
                out.append({"label": label, "note": note, "x": r[0], "y": r[1],
                            "visits": r[2], "tier": "lexical"})
        return out

    # ---- retrieval: semantic (local MiniLM, optional) ----
    def _load_embedder(self):
        if self._embedder_state != "UNCHECKED":
            return
        if os.environ.get("FLYBRAIN_OFFLINE", "0") == "1":
            self._embedder_state = "UNAVAILABLE_OFFLINE"
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.path.join("models", "embedding"))
            self._embedder_state = "LOADED"
        except Exception as e:  # noqa: BLE001
            self._embedder_state = f"UNAVAILABLE:{type(e).__name__}"

    def semantic_status(self) -> Dict[str, Any]:
        self._load_embedder()
        return {"tier": "semantic", "state": self._embedder_state,
                "model": "sentence-transformers/all-MiniLM-L6-v2"}

    def search_places_semantic(self, query: str, limit: int = 5) -> Dict[str, Any]:
        self._load_embedder()
        if self._embedder is None:
            return {"status": "UNAVAILABLE", "reason": self._embedder_state,
                    "fallback": "use search_places (lexical)"}
        rows = self.db.execute("SELECT label,note,x,y,visits FROM places").fetchall()
        if not rows:
            return {"status": "OK", "results": []}
        docs = [f"{r[0]} {r[1]}" for r in rows]
        qv = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        dv = self._embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        sims = sorted(((float(dv[i] @ qv), rows[i]) for i in range(len(rows))),
                      reverse=True)[:limit]
        return {"status": "OK",
                "results": [{"label": r[0], "note": r[1], "x": r[2], "y": r[3],
                             "visits": r[4], "score": round(s, 4), "tier": "semantic"}
                            for s, r in sims]}
