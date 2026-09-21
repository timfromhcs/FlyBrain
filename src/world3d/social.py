"""Embodied social encounters (V6 phase_11): proximity + real teaching.

Friendship NEVER initializes true. Trust/familiarity move only through
measured teach() sessions and recorded interaction outcomes (existing
culture/social machinery). Returns measured deltas, never claims.
"""
import math
from typing import Any, Dict

from src.culture.transmission import teach


def social_encounter(a, b, domain: str = "forage", teacher_seed: int = 48) -> Dict[str, Any]:
    """Two EmbodiedAgents meet. The more skilled teaches; both record."""
    ax, ay = a.body3d.pos[0], a.body3d.pos[1]
    bx, by = b.body3d.pos[0], b.body3d.pos[1]
    dist = math.hypot(ax - bx, ay - by)
    if dist > 3.0:
        return {"status": "TOO_FAR", "dist": round(dist, 2)}
    teacher, student = (a, b) if a.org.skills.get(domain, 0) >= b.org.skills.get(domain, 0) else (b, a)
    t0 = teacher.org.social_mem.trust_of(student.org.id) if teacher.org.social_mem else 0.0
    s0 = student.org.social_mem.trust_of(teacher.org.id) if student.org.social_mem else 0.0
    sess = teach(teacher.org, student.org, domain, teacher.world.tick, teacher_seed)
    teacher.org.social_mem.record_interaction(
        student.org.id, teacher.world.tick, "taught_to",
        min(1.0, sess.learning_gain * 1.5))
    student.org.social_mem.record_interaction(
        teacher.org.id, student.world.tick, "taught_by",
        min(1.0, sess.learning_gain * 2.0))
    t1 = teacher.org.social_mem.trust_of(student.org.id)
    s1 = student.org.social_mem.trust_of(teacher.org.id)
    for ag, other, role in ((teacher, student, "teacher"), (student, teacher, "student")):
        ag.org.episodes.append({"tick": ag.world.tick, "organism_id": ag.org.id,
                                "observation": [f"friend:{other.org.id}"],
                                "action": "SOCIAL_INTERACT", "reward": round(sess.learning_gain, 4),
                                "goal": "socialize", "role": role, "domain": domain})
    return {"status": "INTERACTED", "dist": round(dist, 2), "teacher": teacher.org.id,
            "student": student.org.id, "learning_gain": sess.learning_gain,
            "teacher_trust_delta": round(t1 - t0, 4),
            "student_trust_delta": round(s1 - s0, 4)}
