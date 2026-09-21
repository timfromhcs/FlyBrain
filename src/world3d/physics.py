"""Authoritative 3D rigid-body physics via MuJoCo (V6).

Character pattern: dynamic capsule body, planar velocity servo set from the
action each step, vertical free (gravity integrates). Contacts are resolved
by the solver: walls block, floors support, dynamic bodies get pushed.
Position is NEVER teleported by movement code (only save/restore sets state).

Capabilities verified in tests/test_v6_physics.py: gravity, floor contact +
stable rest, wall blocking, object pushing, raycast queries, save/restore.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
    _MUJOCO = True
except Exception:
    mujoco = None  # type: ignore
    _MUJOCO = False

PHYSICS_DT = 0.01          # 100 Hz substeps
CHARACTER_R = 0.3
CHARACTER_H = 0.9          # capsule cylinder section
EYE_HEIGHT = 1.45


def _box_wall(cx, cy, sx, sy, h, name):
    return (f'<geom name="{name}" type="box" pos="{cx} {cy} {h / 2}" '
            f'size="{sx / 2} {sy / 2} {h / 2}" rgba="0.45 0.4 0.5 1"/>')


def build_model_xml(spec: Dict[str, Any],
                    extra_bodies: Optional[List[Dict[str, Any]]] = None) -> str:
    """Compile the world spec (+ characters/foods) to a MuJoCo model."""
    size = float(spec.get("size", 40.0))
    parts = [f'<mujoco><compiler angle="degree"/>',
             f'<option timestep="0.01" gravity="0 0 -9.81"/>',
             f'<worldbody>',
             f'<geom name="ground" type="plane" size="{size / 2} {size / 2} 0.1" '
             f'material="ground"/>']
    home = spec["home"]
    cx, cy, w, d, h, t = (home["cx"], home["cy"], home["w"], home["d"],
                          home["wall_h"], home["wall_t"])
    dw = home["door_w"]
    # north wall (full), south wall split for door gap, east/west full
    parts.append(_box_wall(cx, cy + d / 2, w, t, h, "wall_north"))
    parts.append(_box_wall(cx - (dw / 2 + (w - dw) / 4), cy - d / 2, (w - dw) / 2, t, h,
                           "wall_south_l"))
    parts.append(_box_wall(cx + (dw / 2 + (w - dw) / 4), cy - d / 2, (w - dw) / 2, t, h,
                           "wall_south_r"))
    parts.append(_box_wall(cx - w / 2, cy, t, d, h, "wall_west"))
    parts.append(_box_wall(cx + w / 2, cy, t, d, h, "wall_east"))
    for f in spec.get("furniture", []):
        if f["kind"] == "static_box":
            parts.append(_box_wall(f["x"], f["y"], f["sx"], f["sy"], f["sz"],
                                   "furn_" + str(f["id"])))
    for tr in spec.get("trees", []):
        parts.append(f'<geom name="trunk_{tr["id"]}" type="cylinder" '
                     f'pos="{tr["x"]} {tr["y"]} 1.0" size="0.25 2.0" '
                     f'rgba="0.35 0.25 0.15 1"/>')
    for rk in spec.get("rocks", []):
        parts.append(f'<geom name="rock_{rk["id"]}" type="sphere" '
                     f'pos="{rk["x"]} {rk["y"]} {rk["r"]}" size="{rk["r"]}" '
                     f'rgba="0.5 0.5 0.55 1"/>')
    # dynamic bodies: crates, doors (hinged), foods, characters
    for f in spec.get("furniture", []):
        if f["kind"] == "dynamic_box":
            parts.append(
                f'<body name="bod_{f["id"]}" pos="{f["x"]} {f["y"]} {f["sz"] / 2}">'
                f'<freejoint/><geom name="dyn_{f["id"]}" type="box" '
                f'size="{f["sx"] / 2} {f["sy"] / 2} {f["sz"] / 2}" mass="{f.get("mass", 4.0)}" '
                f'rgba="0.6 0.45 0.2 1"/></body>')
    for door in spec.get("doors", []):
        parts.append(
            f'<body name="bod_{door["id"]}" pos="{door["x"] - door["w"] / 2} {door["y"]} 1.0">'
            f'<joint name="hinge_{door["id"]}" type="hinge" axis="0 0 1" damping="2.0"/>'
            f'<geom name="dyn_{door["id"]}" type="box" '
            f'pos="{door["w"] / 2} 0 0" size="{door["w"] / 2} 0.05 {door["h"] / 2}" '
            f'mass="6.0" rgba="0.55 0.35 0.2 1"/></body>')
    for fd in spec.get("foods", []):
        parts.append(
            f'<body name="bod_{fd["id"]}" pos="{fd["x"]} {fd["y"]} 0.25">'
            f'<freejoint/><geom name="food_{fd["id"]}" type="sphere" size="0.18" '
            f'mass="0.3" rgba="0.2 0.8 0.3 1"/></body>')
    for ch in (extra_bodies or []):
        parts.append(
            f'<body name="{ch["name"]}" pos="{ch["x"]} {ch["y"]} 1.2">'
            f'<freejoint/><geom name="char_{ch["name"]}" type="capsule" '
            f'size="{CHARACTER_R} {CHARACTER_H / 2}" mass="60.0" '
            f'rgba="0.2 0.7 0.9 1"/>'
            f'<geom name="head_{ch["name"]}" type="sphere" pos="0 0 0.85" size="0.22" '
            f'mass="5.0" rgba="0.9 0.75 0.6 1"/>'
            f'<site name="eye_{ch["name"]}" pos="0 0.18 0.95"/></body>')
    parts.append('</worldbody>')
    parts.append('<asset><material name="ground" rgba="0.25 0.35 0.22 1"/></asset>')
    parts.append('</mujoco>')
    return "\n".join(parts)


class PhysicsWorld:
    """Owns MuJoCo model+data, steps, queries, save/restore."""

    def __init__(self, spec: Dict[str, Any],
                 characters: Optional[List[Dict[str, Any]]] = None):
        if not _MUJOCO:
            raise RuntimeError("mujoco unavailable: 3D physics cannot run here")
        self.spec = spec
        self.xml = build_model_xml(spec, characters)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.0
        # Character bodies under posture stabilization (documented embodiment:
        # strong angular damping keeps the capsule upright; no teleporting,
        # no kinematic freezing — contacts and gravity still fully simulated).
        self._characters: List[str] = [c["name"] for c in (characters or [])]
        self.posture_damping: float = 0.25
        self._cmd: Dict[str, List[float]] = {c["name"]: [0.0, 0.0]
                                             for c in (characters or [])}

    # ---- characters ----
    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"no body {name!r}")
        return bid

    def _qpos_adr(self, name: str) -> int:
        bid = self._body_id(name)
        jnt = int(self.model.body_jntadr[bid])
        return int(self.model.jnt_qposadr[jnt])

    def _dof_adr(self, name: str) -> int:
        return int(self.model.body_dofadr[self._body_id(name)])

    def char_state(self, name: str) -> Dict[str, Any]:
        qa, da = self._qpos_adr(name), self._dof_adr(name)
        qpos = self.data.qpos[qa:qa + 7].copy()
        qvel = self.data.qvel[da:da + 6].copy()
        pos = [round(float(qpos[0]), 4), round(float(qpos[1]), 4), round(float(qpos[2]), 4)]
        vel = [round(float(qvel[0]), 4), round(float(qvel[1]), 4), round(float(qvel[2]), 4)]
        return {"pos": pos, "vel": vel, "quat": [round(float(x), 4) for x in qpos[3:7]]}

    def drive_character(self, name: str, vx: float, vy: float) -> None:
        """Velocity servo (called BEFORE step): planar velocity command.
        Contacts + gravity still resolved by the solver inside step()."""
        da = self._dof_adr(name)
        self.data.qvel[da + 0] = float(vx)
        self.data.qvel[da + 1] = float(vy)
        self._cmd[name] = [float(vx), float(vy)]

    def push_character(self, name: str, fx: float, fy: float, fz: float = 0.0) -> None:
        da = self._dof_adr(name)
        self.data.qfrc_applied[da + 0] = float(fx)
        self.data.qfrc_applied[da + 1] = float(fy)
        self.data.qfrc_applied[da + 2] = float(fz)

    def teleport(self, name: str, x: float, y: float, z: float) -> None:
        """SAVE/RESTORE ONLY. Movement code must never call this."""
        qa, da = self._qpos_adr(name), self._dof_adr(name)
        self.data.qpos[qa:qa + 3] = [float(x), float(y), float(z)]
        self.data.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _stabilize(self, name: str) -> None:
        """Attitude PD controller (real torques, heading left free).

        Drives body-up toward world-up while preserving yaw. Contacts,
        gravity and pushes still act fully — this only resists toppling,
        like vestibular balance. Gains are part of the embodiment model.
        """
        qa, da = self._qpos_adr(name), self._dof_adr(name)
        qw, qx, qy, qz = (float(v) for v in self.data.qpos[qa + 3:qa + 7])
        n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        # current yaw; target = yaw-only orientation (upright, same heading)
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        tw, tx, ty, tz = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        # error quat q_err = q_target * conj(q): vector part ~ rotation axis*angle/2
        ex = tw * -qx + tx * qw + ty * -qz + tz * qy
        ey = tw * -qy + tx * qz + ty * qw + tz * -qx
        ez = tw * -qz + tx * -qy + ty * qx + tz * qw
        ew = tw * qw + tx * qx + ty * qy + tz * qz
        s = 1.0 if ew >= 0 else -1.0
        kp, kd = 300.0, 20.0
        # Get-up maneuver: fallen and (nearly) still -> strong righting.
        # Falls are real events (see body damage); recovery is a real maneuver,
        # not a teleport: torques integrate through the solver.
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        cmd = self._cmd.get(name, [0.0, 0.0])
        still = abs(cmd[0]) + abs(cmd[1]) < 0.05
        if up_z < 0.4 and still:
            kp, kd = 1200.0, 40.0
        wx, wy, wz = (float(v) for v in self.data.qvel[da + 3:da + 6])
        # down-weight yaw correction (heading is driven by locomotion)
        tx_, ty_, tz_ = (kp * s * ex - kd * wx, kp * s * ey - kd * wy,
                         0.25 * (kp * s * ez - kd * wz))
        self.data.qfrc_applied[da + 3] += tx_
        self.data.qfrc_applied[da + 4] += ty_
        self.data.qfrc_applied[da + 5] += tz_

    # ---- stepping ----
    def step(self, dt: float = PHYSICS_DT) -> None:
        for name in self._characters:
            try:
                self._stabilize(name)
            except KeyError:
                pass
        mujoco.mj_step(self.model, self.data)
        self.time = float(self.data.time)
        for name in self._characters:
            try:
                # Applied forces are single-step impulses: re-apply each step
                # for continuous push (no hidden persistent forces).
                da = self._dof_adr(name)
                self.data.qfrc_applied[da:da + 6] = 0.0
            except KeyError:
                pass

    def upright(self, name: str) -> float:
        """Up-axis alignment 0..1 (1 = fully upright)."""
        qa = self._qpos_adr(name)
        qw, qx, qy, qz = (float(v) for v in self.data.qpos[qa + 3:qa + 7])
        # rotate world +Z by inverse quat: z-component of body x-axis... use
        # body up vector = R * (0,0,1); take its z via quat formula
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        return round(float(up_z), 4)

    # ---- queries ----
    def contacts(self) -> List[Dict[str, Any]]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            out.append({"geom1": g1, "geom2": g2,
                        "dist": round(float(c.dist), 5)})
        return out

    def grounded(self, char_name: str) -> bool:
        tag = f"char_{char_name}"
        for c in self.contacts():
            if tag in (c["geom1"], c["geom2"]) and "ground" in (c["geom1"], c["geom2"]):
                return True
        return False

    def raycast(self, origin: List[float], direction: List[float],
                max_dist: float = 20.0, exclude_body: str = "") -> Dict[str, Any]:
        """Single geometric ray (vision primitive). Returns hit entity or miss."""
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, self.data,
                             np.asarray(origin, dtype=np.float64),
                             np.asarray(direction, dtype=np.float64),
                             None, 1, -1, geomid)
        if dist < 0 or dist > max_dist:
            return {"hit": False, "dist": round(float(max_dist), 3)}
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
        if exclude_body and name.endswith(exclude_body):
            return {"hit": False, "dist": round(float(max_dist), 3),
                    "note": "self-hit excluded"}
        return {"hit": True, "dist": round(float(dist), 3), "geom": name,
                "entity": name.split("_", 1)[1] if "_" in name else name}

    # ---- persistence ----
    def snapshot(self) -> Dict[str, Any]:
        return {"time": self.time, "qpos": self.data.qpos.copy().tolist(),
                "qvel": self.data.qvel.copy().tolist(),
                "spec_hash": self.spec.get("spec_hash", "")}

    def restore(self, snap: Dict[str, Any]) -> None:
        if snap.get("spec_hash", "") != self.spec.get("spec_hash", ""):
            raise ValueError("physics snapshot belongs to a different world spec")
        self.data.qpos[:] = np.asarray(snap["qpos"], dtype=np.float64)
        self.data.qvel[:] = np.asarray(snap["qvel"], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        self.time = float(snap.get("time", 0.0))
