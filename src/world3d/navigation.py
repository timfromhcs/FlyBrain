"""Grid navigation with known-vs-true map separation (V6 phase_09).

TRUE map: static obstacles inflated from the world spec (authoritative).
KNOWN map: per-organism discovered cells (free/blocked) + remembered paths.
Planning runs on KNOWN (unknown = blocked unless goal is exploration);
stall detection (no progress while driving) marks cells blocked and replans.
"""
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

CELL = 0.25  # 0.25 m cells: the 1.4 m door gap stays passable after
             # robot-radius inflation (0.5 m cells sealed it by quantization)


def _cells_for_rect(cx, cy, sx, sy, inflate, size) -> Set[Tuple[int, int]]:
    cells = set()
    hx, hy = sx / 2 + inflate, sy / 2 + inflate
    x0 = int(math.floor((cx - hx + size / 2) / CELL))
    x1 = int(math.floor((cx + hx + size / 2) / CELL))
    y0 = int(math.floor((cy - hy + size / 2) / CELL))
    y1 = int(math.floor((cy + hy + size / 2) / CELL))
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            cells.add((ix, iy))
    return cells


def true_blocked_cells(spec: Dict[str, Any]) -> Set[Tuple[int, int]]:
    size = float(spec.get("size", 40.0))
    blocked: Set[Tuple[int, int]] = set()
    home = spec["home"]
    cx, cy, w, d = home["cx"], home["cy"], home["w"], home["d"]
    t, dw = home["wall_t"], home["door_w"]
    # walls as rects (door gap excluded on south)
    segs = [(cx, cy + d / 2, w, t), (cx - w / 2, cy, t, d), (cx + w / 2, cy, t, d),
            (cx - (dw / 2 + (w - dw) / 4), cy - d / 2, (w - dw) / 2, t),
            (cx + (dw / 2 + (w - dw) / 4), cy - d / 2, (w - dw) / 2, t)]
    for (x, y, sx, sy) in segs:
        blocked |= _cells_for_rect(x, y, sx, sy, 0.3, size)
    for f in spec.get("furniture", []):
        if f["kind"] == "static_box":
            blocked |= _cells_for_rect(f["x"], f["y"], f["sx"], f["sy"], 0.3, size)
    for tr in spec.get("trees", []):
        blocked |= _cells_for_rect(tr["x"], tr["y"], 0.5, 0.5, 0.25, size)
    for rk in spec.get("rocks", []):
        blocked |= _cells_for_rect(rk["x"], rk["y"], rk["r"] * 2, rk["r"] * 2, 0.25, size)
    return blocked


def to_cell(x: float, y: float, size: float) -> Tuple[int, int]:
    return (int(math.floor((x + size / 2) / CELL)),
            int(math.floor((y + size / 2) / CELL)))


def to_xy(cell: Tuple[int, int], size: float) -> Tuple[float, float]:
    return ((cell[0] + 0.5) * CELL - size / 2, (cell[1] + 0.5) * CELL - size / 2)


def astar(start: Tuple[int, int], goal: Tuple[int, int],
          blocked: Set[Tuple[int, int]], size: float,
          max_expand: int = 20000) -> Optional[List[Tuple[int, int]]]:
    n = int(size / CELL)
    if goal in blocked:
        return None
    open_h = [(0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g = {start: 0.0}
    expanded = 0
    while open_h and expanded < max_expand:
        _, cur = heapq.heappop(open_h)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        expanded += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < n and 0 <= nb[1] < n) or nb in blocked:
                continue
            ng = g[cur] + (1.4142 if dx and dy else 1.0)
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                came[nb] = cur
                h = math.hypot(nb[0] - goal[0], nb[1] - goal[1])
                heapq.heappush(open_h, (ng + h, nb))
    return None


class Navigator:
    """Per-organism navigator: plans on KNOWN map, records successes."""

    def __init__(self, spec: Dict[str, Any]):
        self.size = float(spec.get("size", 40.0))
        self.true_blocked = true_blocked_cells(spec)
        self.known_free: Set[Tuple[int, int]] = set()
        self.known_blocked: Set[Tuple[int, int]] = set()
        self.remembered_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.route: List[Tuple[float, float]] = []
        self.destination: Optional[Tuple[float, float]] = None
        self.replans = 0
        self._last_pos: Optional[Tuple[float, float]] = None
        self._stall_ticks = 0

    def observe_walkable(self, x: float, y: float, radius: float = 1.5) -> None:
        c = to_cell(x, y, self.size)
        r = int(radius / CELL)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cell = (c[0] + dx, c[1] + dy)
                if cell in self.true_blocked:
                    self.known_blocked.add(cell)
                else:
                    self.known_free.add(cell)

    def plan(self, x: float, y: float, gx: float, gy: float,
             explore_unknown: bool = False) -> Optional[List[Tuple[float, float]]]:
        start, goal = to_cell(x, y, self.size), to_cell(gx, gy, self.size)
        if explore_unknown:
            blocked = set(self.known_blocked)
        else:
            # known-only: unknown cells are blocked (no map magic)
            n = int(self.size / CELL)
            blocked = {(ix, iy) for ix in range(n) for iy in range(n)
                       if (ix, iy) not in self.known_free} | self.known_blocked
            blocked.discard(start)
        cells = astar(start, goal, blocked, self.size)
        if cells is None:
            self.route = []
            return None
        self.route = [to_xy(c, self.size) for c in cells[1:]]
        self.destination = (gx, gy)
        return list(self.route)

    def next_waypoint(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        while self.route:
            wx, wy = self.route[0]
            if math.hypot(wx - x, wy - y) < 0.4:
                self.route.pop(0)
                continue
            return (wx, wy)
        return None

    def note_progress(self, x: float, y: float, driving: bool) -> str:
        """Stall detection: driving without progress -> mark blocked + replan."""
        if self._last_pos is None:
            self._last_pos = (x, y)
            return "ok"
        moved = math.hypot(x - self._last_pos[0], y - self._last_pos[1])
        self._last_pos = (x, y)
        if driving and moved < 0.02:
            self._stall_ticks += 1
        else:
            self._stall_ticks = 0
        if self._stall_ticks >= 10:
            self._stall_ticks = 0
            # block cells ahead along current heading is unknown; block current
            c = to_cell(x, y, self.size)
            self.known_blocked.add(c)
            self.replans += 1
            self.route = []
            return "replan"
        return "ok"

    def remember_path(self, name: str, waypoints: List[Tuple[float, float]]) -> None:
        self.remembered_paths[name] = [tuple(map(float, w)) for w in waypoints]
