"""Index saved batches without loading every replay's complete timeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import BEHAVIORS, PLANNERS


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_replay(replay: dict, source: str = "Replay") -> dict:
    if replay.get("format_version") != 1:
        raise ValueError(f"{source}: unsupported replay format")
    for field in ("scenario", "metrics"):
        if not isinstance(replay.get(field), dict):
            raise ValueError(f"{source}: missing {field} object")
    for field in ("frames", "baseline_frames"):
        if not isinstance(replay.get(field), list) or not replay[field]:
            raise ValueError(f"{source}: {field} must be a nonempty timeline")
    return replay


@dataclass(frozen=True)
class ReplayEntry:
    path: Path
    map_key: str
    map_name: str
    behavior: str
    planner: str


class ReplayCatalog:
    def __init__(self, root: Path, entries: list[ReplayEntry], is_batch: bool) -> None:
        self.root = root
        self.is_batch = is_batch
        self.robot_plan_cache: dict[Path, list[dict]] = {}
        map_order = {key: i for i, key in enumerate(dict.fromkeys(e.map_key for e in entries))}
        self.entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    map_order[entry.map_key],
                    BEHAVIORS.index(entry.behavior)
                    if entry.behavior in BEHAVIORS
                    else len(BEHAVIORS),
                    entry.behavior,
                    PLANNERS.index(entry.planner) if entry.planner in PLANNERS else len(PLANNERS),
                    entry.planner,
                ),
            )
        )
        self.maps = dict((entry.map_key, entry.map_name) for entry in self.entries)
        labels = list(self.maps.values())
        for key, label in self.maps.items():
            if labels.count(label) > 1:
                self.maps[key] = f"{label} ({Path(key).stem})"

    @classmethod
    def open(cls, source: str | Path) -> ReplayCatalog:
        source = Path(source).resolve()
        if source.is_file():
            paths, root, is_batch = [source], source.parent, False
        elif source.is_dir():
            root = source
            direct = source / "replay.json"
            is_batch = not direct.is_file()
            paths = [direct] if not is_batch else sorted(source.rglob("replay.json"))
        else:
            raise ValueError(f"Replay file or batch directory does not exist: {source}")
        if not paths:
            raise ValueError(f"No replay.json files found in {source}")
        entries = []
        seen = set()
        for path in paths:
            metrics_path, scenario_path = (
                path.with_name("metrics.json"),
                path.with_name("scenario.json"),
            )
            if metrics_path.is_file() and scenario_path.is_file():
                metrics, scenario = read_json(metrics_path), read_json(scenario_path)
            else:
                replay = validate_replay(read_json(path), str(path))
                metrics, scenario = replay["metrics"], replay["scenario"]
            try:
                name, behavior, planner = scenario["name"], metrics["behavior"], metrics["planner"]
                if not all(isinstance(value, str) and value for value in (name, behavior, planner)):
                    raise ValueError("map name, behavior, and planner must be nonempty strings")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid replay metadata in {path}: {exc}") from exc
            # Existing batches preserve the original source path in scenario.json.
            # For older/custom exports, use the containing map folder instead.
            map_key = scenario.get("source") or str(path.relative_to(root).parts[0])
            identity = (map_key, behavior, planner)
            if identity in seen:
                raise ValueError(
                    f"Multiple replays for {name}, {behavior}, {planner}. "
                    "Open one batch directory at a time."
                )
            seen.add(identity)
            entries.append(ReplayEntry(path, map_key, name, behavior, planner))
        return cls(root, entries, is_batch)

    def load(self, index: int) -> dict:
        entry = self.entries[index]
        replay = validate_replay(read_json(entry.path), str(entry.path))
        if "robot_plans" not in replay and entry.path in self.robot_plan_cache:
            replay["robot_plans"] = self.robot_plan_cache[entry.path]
        return replay

    def options(self, dimension: str, index: int) -> list[tuple[str, str]]:
        current = self.entries[index]
        if dimension == "map":
            return list(self.maps.items())
        if dimension == "behavior":
            values = dict.fromkeys(e.behavior for e in self.entries if e.map_key == current.map_key)
        elif dimension == "planner":
            values = dict.fromkeys(
                e.planner
                for e in self.entries
                if e.map_key == current.map_key and e.behavior == current.behavior
            )
        else:
            raise ValueError(f"Unknown replay selector: {dimension}")
        return [(value, value) for value in values]

    def select(self, dimension: str, value: str, index: int) -> int:
        current = self.entries[index]
        if dimension == "map":
            candidates = [i for i, e in enumerate(self.entries) if e.map_key == value]
        elif dimension == "behavior":
            candidates = [
                i
                for i, e in enumerate(self.entries)
                if e.map_key == current.map_key and e.behavior == value
            ]
        elif dimension == "planner":
            candidates = [
                i
                for i, e in enumerate(self.entries)
                if e.map_key == current.map_key
                and e.behavior == current.behavior
                and e.planner == value
            ]
        else:
            raise ValueError(f"Unknown replay selector: {dimension}")
        if not candidates:
            raise ValueError(f"No saved replay for {dimension}: {value}")
        # Preserve the other selections when possible; incomplete batches fall
        # back deterministically to an existing combination, never a dead link.
        return min(
            candidates,
            key=lambda i: (
                self.entries[i].behavior != current.behavior,
                self.entries[i].planner != current.planner,
                i,
            ),
        )
