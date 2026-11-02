"""Human-versus-robot disruption scatter plots, separated by crowd behavior."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import BEHAVIORS, PLANNERS
from .robot_metrics import saved_robot_reference

METRICS = {
    "total_pedestrian_delay": "Total pedestrian delay (person-ticks)",
    "mean_pedestrian_delay": "Mean pedestrian delay (ticks per person)",
    "max_pedestrian_delay": "Maximum individual delay (ticks)",
    "fairness_delay_score": "Fairness-weighted delay (ticks)",
    "personal_space_exposure": "Personal-space exposure (weighted person-ticks)",
}
DEFAULT_METRIC = "total_pedestrian_delay"
ROBOT_PLOT_ORDER = ("L0", "L1", "L2", "L3", "L3+D", "L4", "L4+D")
PLANNER_COLORS = dict(
    zip(
        PLANNERS,
        (
            "#303b49",
            "#d55e00",
            "#cc79a7",
            "#009e73",
            "#0072b2",
            "#e69f00",
            "#56b4e9",
        ),
    )
)
MAP_MARKERS = ("o", "s", "^", "D", "P", "v", "h", "p", "*", "<", ">", "8")
BEHAVIOR_LABELS = {
    "fixed": "Fixed route: wait when blocked",
    "reroute": "Immediate rerouting",
    "patient": "Patient rerouting",
    "mixed": "Mixed pedestrian behaviors",
    "none": "No pedestrians",
}


@dataclass(frozen=True)
class ResultPoint:
    map_key: str
    planner: str
    behavior: str
    disruption: float
    completed: bool
    robot_disruption: float | None
    horizon_limited: bool = False
    robot_free_flow_time: float | None = None
    robot_elapsed_time: float | None = None


@dataclass(frozen=True)
class PlotData:
    points: tuple[ResultPoint, ...]
    maps: tuple[tuple[str, str], ...]
    metric: str

    @property
    def behaviors(self) -> list[str]:
        present = {p.behavior for p in self.points}
        return [b for b in BEHAVIORS if b in present] + sorted(present - set(BEHAVIORS))


def _number(value, field: str, line: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {line}: {field} must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"CSV row {line}: {field} must be finite and nonnegative")
    return result


def _saved_references(root: Path) -> dict[str, tuple[int, int]]:
    """Recover legacy metrics from recorded inputs, never from live YAML."""
    references = {}
    names = defaultdict(set)
    for path in sorted(root.rglob("scenario.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        key = saved.get("source") or saved["name"]
        if key not in references:
            references[key] = saved_robot_reference(saved)
        names[saved["name"]].add(key)
    for name, keys in names.items():
        if len(keys) == 1:
            references.setdefault(name, references[next(iter(keys))])
    return references


def read_results(source: str | Path, metric: str = DEFAULT_METRIC) -> PlotData:
    if metric not in METRICS:
        raise ValueError(f"Unknown disruption metric {metric!r}; choose from {tuple(METRICS)}")
    source = Path(source)
    if source.is_dir():
        source /= "summary.csv"
    points, maps, seen = [], {}, set()
    references = None
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"scenario", "planner", "behavior", "robot_completed", metric} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"Results CSV is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            label, planner, behavior = [
                (row.get(field) or "").strip() for field in ("scenario", "planner", "behavior")
            ]
            if not label or not behavior:
                raise ValueError(f"CSV row {line}: scenario and behavior must be nonempty")
            if planner not in PLANNERS:
                raise ValueError(f"CSV row {line}: unknown robot configuration {planner!r}")
            raw_completed = (row.get("robot_completed") or "").strip().lower()
            if raw_completed not in {"true", "false", "1", "0"}:
                raise ValueError(f"CSV row {line}: robot_completed must be True or False")
            completed = raw_completed in {"true", "1"}
            value = _number(row[metric], metric, line)
            key = (row.get("source") or "").strip() or label
            identity = (key, behavior, planner)
            if identity in seen:
                raise ValueError(
                    f"CSV row {line}: multiple runs for {label!r}, {behavior}, {planner}; "
                    "repeated-run aggregation requires an explicit success policy"
                )
            seen.add(identity)
            maps.setdefault(key, label)
            optional = {
                field: _number(row[field], field, line) if row.get(field, "").strip() else None
                for field in ("robot_disruption", "robot_free_flow_time", "robot_elapsed_time")
            }
            delay, reference, elapsed = [
                optional[field]
                for field in ("robot_disruption", "robot_free_flow_time", "robot_elapsed_time")
            ]
            if completed and delay is None:
                if elapsed is None or reference is None:
                    if references is None:
                        references = _saved_references(source.parent)
                    saved = references.get(key)
                    if saved is not None:
                        if reference is None:
                            reference = saved[0]
                        if elapsed is None and row.get("robot_arrival_time", "").strip():
                            elapsed = (
                                _number(row["robot_arrival_time"], "robot_arrival_time", line)
                                - saved[1]
                            )
                if elapsed is None or reference is None:
                    raise ValueError(
                        f"CSV row {line}: robot disruption is missing; provide "
                        "robot_disruption, or robot_elapsed_time and robot_free_flow_time, "
                        "or the recorded scenario.json files"
                    )
                if elapsed < reference:
                    raise ValueError(
                        f"CSV row {line}: robot elapsed time is below its free-flow reference"
                    )
                delay = elapsed - reference
            horizon_limited = False
            try:
                count = int(row.get("pedestrian_count", ""))
                horizon_limited = any(
                    int(row.get(field, "")) < count
                    for field in ("pedestrians_completed", "baseline_completed")
                )
            except (TypeError, ValueError):
                pass
            points.append(
                ResultPoint(
                    key,
                    planner,
                    behavior,
                    value,
                    completed,
                    delay if completed else None,
                    horizon_limited,
                    reference,
                    elapsed,
                )
            )
    if not points:
        raise ValueError("The results CSV contains no experiment rows")
    labels = list(maps.values())
    for index, (key, label) in enumerate(maps.items()):
        if labels.count(label) > 1:
            maps[key] = f"{label} [{index + 1}: {Path(key).stem}]"
    return PlotData(tuple(points), tuple(maps.items()), metric)


def map_markers(data: PlotData) -> dict:
    return {
        key: MAP_MARKERS[i] if i < len(MAP_MARKERS) else (i + 3, 0, i * 17 % 360)
        for i, (key, _) in enumerate(data.maps)
    }


def build_figure(data: PlotData, behavior: str):
    """Use a separate categorical panel for runs with no final robot delay."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    from matplotlib.transforms import ScaledTranslation

    points = [p for p in data.points if p.behavior == behavior]
    if not points:
        raise ValueError(f"No results for pedestrian behavior {behavior!r}")
    figure = Figure(figsize=(14.5, max(8.2, 5.7 + len(data.maps) * 0.34)), facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        1, 2, width_ratios=(5.5, 1.1), wspace=0.10, left=0.09, right=0.745, bottom=0.24, top=0.79
    )
    axis = figure.add_subplot(grid[0])
    failed = figure.add_subplot(grid[1], sharey=axis)
    failed.set_facecolor("#fff6e9")
    failed.set_xlim(-0.7, 0.7)
    failed.set_xticks([0], ["Did not finish"])
    failed.set_title("Unfinished", fontsize=11, color="#76532b", pad=13)
    failed.tick_params(axis="y", left=False, labelleft=False)
    failed.set_xlabel("Separate category", fontsize=10, labelpad=15, color="#76532b")
    maximum_x = max((p.robot_disruption for p in data.points if p.completed), default=0)
    span = max(maximum_x, 1)
    axis.set_xlim(-span * 0.09, span * 1.12)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.set_xlabel("Robot disruption: extra travel time (ticks)", fontsize=12, labelpad=14)
    axis.set_ylabel(METRICS[data.metric], fontsize=12, labelpad=12)
    maximum_y = max(max(p.disruption for p in data.points), 1)
    axis.set_yscale("symlog", base=10, linthresh=1, linscale=0.8)
    axis.set_ylim(-0.3, maximum_y * 1.5)
    ticks = [0] + [
        m * 10.0**e
        for e in range(math.ceil(math.log10(maximum_y * 1.5)))
        for m in (1, 2, 5)
        if m * 10.0**e <= maximum_y * 1.5
    ]
    if maximum_y <= 1:
        ticks = [0, 0.25, 0.5, 0.75, 1]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_locator(NullLocator())
    markers = map_markers(data)
    map_indices = {key: i for i, (key, _) in enumerate(data.maps)}
    groups = defaultdict(list)
    for point in points:
        groups[(point.completed, point.robot_disruption, point.disruption)].append(point)
    for group in groups.values():
        columns = min(7, len(group))
        rows = math.ceil(len(group) / columns)
        for i, point in enumerate(group):
            panel = axis if point.completed else failed
            dot = panel.scatter(
                [point.robot_disruption if point.completed else 0],
                [point.disruption],
                marker=markers[point.map_key],
                color=PLANNER_COLORS[point.planner],
                s=66,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            # Separate only exact coincidences in screen space; stored data and
            # the numeric scatter coordinates remain unchanged.
            dx = (i % columns - (columns - 1) / 2) * 8
            dy = (i // columns - (rows - 1) / 2) * 8
            dot.set_offset_transform(
                panel.transData + ScaledTranslation(dx / 72, dy / 72, figure.dpi_scale_trans)
            )
            dot.set_gid(f"{point.planner}-{map_indices[point.map_key]}")
    if not any(not p.completed for p in points):
        failed.text(
            0.5,
            0.5,
            "No unfinished\nruns",
            ha="center",
            va="center",
            transform=failed.transAxes,
            fontsize=10,
            color="#947a5d",
        )
    for panel in (axis, failed):
        panel.grid(axis="y", color="#e3e9ef", linewidth=0.8)
        panel.set_axisbelow(True)
        panel.tick_params(axis="both", labelsize=10, colors="#46586a")
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color("#c9d4de")
    failed.spines["left"].set_visible(False)
    figure.text(
        0.09, 0.93, "Human and robot disruption", fontsize=22, weight="bold", color="#203041"
    )
    figure.text(0.09, 0.875, BEHAVIOR_LABELS.get(behavior, behavior), fontsize=14, color="#586b7c")
    planners = [p for p in ROBOT_PLOT_ORDER if any(q.planner == p for q in data.points)]
    figure.legend(
        handles=[
            Line2D(
                [], [], linestyle="none", marker="o", markersize=7, color=PLANNER_COLORS[p], label=p
            )
            for p in planners
        ],
        title="Human awareness level",
        loc="upper left",
        bbox_to_anchor=(0.77, 0.82),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker=markers[key],
                markersize=7,
                color="#46586a",
                label=textwrap.fill(label, 27),
            )
            for key, label in data.maps
        ],
        title="Map",
        loc="upper left",
        bbox_to_anchor=(0.77, 0.51),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        labelspacing=0.8,
    )
    notes = [
        "Robot disruption = elapsed trip time minus the shortest trip without pedestrians "
        "(linear X scale).",
        "Y is logarithmic above 1, linear near zero. Unfinished robots occupy a separate, "
        "nonnumeric column.",
        "Exact overlaps are separated by a few pixels; scatter-data.csv preserves "
        "the exact measurements.",
    ]
    if any(p.horizon_limited for p in points):
        notes.append("Unfinished pedestrian trips use horizon-limited human disruption.")
    for i, note in enumerate(notes):
        figure.text(0.09, 0.14 - i * 0.03, note, fontsize=9, color="#687b8b")
    return figure


def plot_results(
    source: str | Path, output: str | Path | None = None, metric: str = DEFAULT_METRIC
) -> list[Path]:
    source = Path(source)
    csv_path = source / "summary.csv" if source.is_dir() else source
    data = read_results(csv_path, metric)
    destination = Path(output) if output is not None else csv_path.parent / "graphs"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((destination / ".matplotlib-cache").resolve()))
    with (destination / "scatter-data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "map",
                "source",
                "behavior",
                "planner",
                "robot_completed",
                "robot_disruption",
                "robot_free_flow_time",
                "robot_elapsed_time",
                "human_metric",
                "human_disruption",
            ]
        )
        labels = dict(data.maps)
        for p in data.points:
            writer.writerow(
                [
                    labels[p.map_key],
                    p.map_key,
                    p.behavior,
                    p.planner,
                    p.completed,
                    p.robot_disruption,
                    p.robot_free_flow_time,
                    p.robot_elapsed_time,
                    metric,
                    p.disruption,
                ]
            )
    outputs = []
    for index, behavior in enumerate(data.behaviors):
        figure = build_figure(data, behavior)
        slug = re.sub(r"[^a-z0-9_-]+", "-", behavior.lower()).strip("-") or f"behavior-{index}"
        for extension in ("png", "svg"):
            path = destination / f"{slug}-{metric}.{extension}"
            figure.savefig(path, dpi=200, facecolor="white")
            outputs.append(path)
        figure.clear()
    return outputs
