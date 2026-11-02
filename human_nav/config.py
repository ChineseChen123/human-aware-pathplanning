"""Validated YAML input. Coordinates are (column, row), starting at zero."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

Point = tuple[int, int]
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
BEHAVIORS = ("fixed", "reroute", "patient")
PLANNERS = ("L0", "L1", "L2", "L3", "L4", "L3+D", "L4+D")
GLOBAL_CONSTANTS_PATH = Path(__file__).resolve().parent.parent / "constants.yaml"
AGENT_FIELDS = {
    "id",
    "start",
    "goal",
    "waypoints",
    "spawn_time",
    "move_interval",
    "behavior",
    "wait_threshold",
    "personal_radius",
    "direction_order",
}
AGENT_DEFAULT_FIELDS = AGENT_FIELDS - {"id", "start", "goal", "waypoints"}


@dataclass(frozen=True)
class Grid:
    rows: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def walkable(self, p: Point) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height and self.rows[y][x] == "."

    def neighbors(self, p: Point, order: str = "NESW") -> list[Point]:
        return [
            q
            for direction in order
            if self.walkable(
                q := (p[0] + DIRECTIONS[direction][0], p[1] + DIRECTIONS[direction][1])
            )
        ]


@dataclass(frozen=True)
class AgentSpec:
    id: str
    start: Point
    goals: tuple[Point, ...]
    spawn_time: int = 0
    move_interval: int = 1
    behavior: str = "patient"
    wait_threshold: int = 3
    personal_radius: float = 2.0
    direction_order: str = "NESW"


@dataclass(frozen=True)
class Settings:
    max_steps: int = 120
    search_horizon: int = 60
    max_expansions: int = 15000
    prediction_steps: int = 5
    alpha: float = 1.0
    beta: float = 3.0
    gamma: float = 4.0
    fairness_weight: float = 0.25
    seconds_per_step: float = 0.35
    seed: int = 0


@dataclass(frozen=True)
class Scenario:
    name: str
    grid: Grid
    robot: AgentSpec
    pedestrians: tuple[AgentSpec, ...]
    settings: Settings
    source: str = ""

    def with_behavior(self, behavior: str | None) -> Scenario:
        if behavior is None:
            return self
        if behavior not in BEHAVIORS:
            raise ValueError(f"Unknown pedestrian behavior: {behavior}")
        return replace(
            self, pedestrians=tuple(replace(p, behavior=behavior) for p in self.pedestrians)
        )

    def resolved(self) -> dict[str, Any]:
        """Self-contained, JSON-safe experiment input, including defaults."""
        return asdict(self)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _point(value: Any, label: str, grid: Grid) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y]")
    p = (_integer(value[0], label), _integer(value[1], label))
    if not grid.walkable(p):
        raise ValueError(f"{label}: {p} is outside the map or inside a wall")
    return p


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _known(data: dict, allowed: set[str], label: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(map(str, extra)))}")


def _read_yaml(path: str | Path) -> Any:
    path = Path(path)
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def _parse_constants(data: Any) -> dict[str, dict]:
    data = _mapping(data, "Global constants")
    _known(data, {"defaults", "simulation"}, "global constants")
    defaults = _mapping(data.get("defaults", {}), "global defaults")
    simulation = _mapping(data.get("simulation", {}), "global simulation")
    _known(defaults, AGENT_DEFAULT_FIELDS, "global defaults")
    _known(simulation, set(Settings.__dataclass_fields__), "global simulation")
    return {"defaults": defaults, "simulation": simulation}


def load_constants(path: str | Path | None = None) -> dict[str, dict]:
    """Read shared values afresh, so edits apply on the next scenario load."""
    return _parse_constants(_read_yaml(path if path is not None else GLOBAL_CONSTANTS_PATH))


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = _read_yaml(path)
    return parse_scenario(data, str(path.resolve()))


def parse_scenario(data: Any, source: str = "", *, constants: dict | None = None) -> Scenario:
    data = _mapping(data, "Scenario")
    _known(data, {"name", "map", "robot", "pedestrians", "defaults", "simulation"}, "scenario")
    shared = load_constants() if constants is None else _parse_constants(constants)
    raw_map = data.get("map")
    rows = raw_map.strip("\n").splitlines() if isinstance(raw_map, str) else raw_map
    if not isinstance(rows, list) or not rows or any(not isinstance(r, str) for r in rows):
        raise ValueError("map must be a multiline string or a list of strings")
    if not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Map rows must have the same nonzero width")
    if any(set(r) - {"#", "."} for r in rows):
        raise ValueError("Use only '#' for walls and '.' for walkable map cells")
    grid = Grid(tuple(rows))
    defaults = {**shared["defaults"], **_mapping(data.get("defaults", {}), "defaults")}
    _known(defaults, AGENT_DEFAULT_FIELDS, "defaults")

    def agent(raw: Any, robot: bool = False) -> AgentSpec:
        raw = _mapping(raw, "agent")
        _known(raw, AGENT_FIELDS, "agent")
        values = {**defaults, **raw}
        ident = values.get("id", "robot" if robot else None)
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError("Each pedestrian needs a nonempty string id")
        if "goal" not in values or "start" not in values:
            raise ValueError(f"{ident} needs start and goal coordinates")
        waypoints = values.get("waypoints", [])
        if not isinstance(waypoints, list):
            raise ValueError(f"{ident}.waypoints must be a list")
        goals = tuple(_point(p, f"{ident}.waypoint", grid) for p in waypoints)
        goals += (_point(values["goal"], f"{ident}.goal", grid),)
        behavior = values.get("behavior", "patient")
        if behavior not in BEHAVIORS:
            raise ValueError(f"{ident}.behavior must be one of {BEHAVIORS}")
        order = values.get("direction_order", "NESW")
        if not isinstance(order, str) or len(order) != 4 or set(order) != set(DIRECTIONS):
            raise ValueError(f"{ident}.direction_order must be a permutation of NESW")
        radius = values.get("personal_radius", 2.0)
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError(f"{ident}.personal_radius must be a finite positive number")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"{ident}.personal_radius must be a finite positive number")
        return AgentSpec(
            id=ident,
            start=_point(values["start"], f"{ident}.start", grid),
            goals=goals,
            spawn_time=_integer(values.get("spawn_time", 0), f"{ident}.spawn_time"),
            move_interval=_integer(values.get("move_interval", 1), f"{ident}.move_interval", 1),
            behavior=behavior,
            wait_threshold=_integer(values.get("wait_threshold", 3), f"{ident}.wait_threshold", 1),
            personal_radius=float(radius),
            direction_order=order,
        )

    robot = agent(data.get("robot"), robot=True)
    people = data.get("pedestrians", [])
    if not isinstance(people, list):
        raise ValueError("pedestrians must be a list")
    pedestrians = tuple(agent(p) for p in people)
    agents = (robot, *pedestrians)
    if len({a.id for a in agents}) != len(agents):
        raise ValueError("Agent ids must be unique, including the robot")
    if len({(a.start, a.spawn_time) for a in agents}) != len(agents):
        raise ValueError("Agents cannot share a start cell at the same spawn time")

    raw_settings = {**shared["simulation"], **_mapping(data.get("simulation", {}), "simulation")}
    _known(raw_settings, set(Settings.__dataclass_fields__), "simulation")
    settings = Settings(**raw_settings)
    for field in ("max_steps", "search_horizon", "max_expansions", "prediction_steps"):
        _integer(getattr(settings, field), field, 1)
    _integer(settings.seed, "seed")
    for field in ("alpha", "beta", "gamma", "fairness_weight", "seconds_per_step"):
        value = getattr(settings, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a finite number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be a finite nonnegative number")
    if settings.alpha <= 0 or settings.seconds_per_step <= 0:
        raise ValueError("alpha and seconds_per_step must be positive")
    if settings.fairness_weight > 1:
        raise ValueError("fairness_weight must be between zero and one")
    if max(a.spawn_time for a in agents) >= settings.max_steps:
        raise ValueError("All spawn times must be before max_steps")

    # Reject permanent disconnections, not dynamic congestion or deadlocks.
    from .search import astar

    for spec in agents:
        start = spec.start
        for goal in spec.goals:
            if astar(grid, start, goal, order=spec.direction_order) is None:
                raise ValueError(f"{spec.id}: no wall-only route from {start} to {goal}")
            start = goal
    name = data.get("name", "Untitled scenario")
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    return Scenario(name, grid, robot, pedestrians, settings, source)
