"""Interactive orthographic results cube and awareness-level slices for pygame."""

from __future__ import annotations

import colorsys
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass

import pygame

from .plotting import (
    BEHAVIOR_LABELS,
    DEFAULT_METRIC,
    METRICS,
    ROBOT_PLOT_ORDER,
    PlotData,
    ResultPoint,
    read_results,
)
from .replays import ReplayCatalog, read_json
from .robot_metrics import saved_robot_reference

INK = (32, 48, 65)
MUTED = (106, 123, 138)
BLUE = (47, 102, 224)
PALETTE = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#303b49")
FACE_VIEWS = {"front": (0.0, 0.0), "side": (math.pi / 2, 0.0), "top": (0.0, math.pi / 2)}


def human_scale(value: float) -> float:
    """Match the 2D symlog scale: exact zero, linear to 1, then base-10 log."""
    return value * (0.8 / 0.9) if value <= 1 else 0.8 / 0.9 + math.log10(value)


def _point(metrics: dict, scenario: dict, key: str, metric: str) -> ResultPoint:
    completed = bool(metrics["robot_completed"])
    reference, spawn = saved_robot_reference(scenario)
    elapsed = metrics.get("robot_elapsed_time")
    if completed and elapsed is None:
        elapsed = metrics["robot_arrival_time"] - spawn
    delay = metrics.get("robot_disruption")
    if completed and delay is None:
        delay = max(0, elapsed - reference)
    count = metrics["pedestrian_count"]
    return ResultPoint(
        key,
        metrics["planner"],
        metrics["behavior"],
        float(metrics[metric]),
        completed,
        delay if completed else None,
        metrics["pedestrians_completed"] < count or metrics["baseline_completed"] < count,
        reference,
        elapsed,
    )


def catalog_data(catalog: ReplayCatalog, metric: str = DEFAULT_METRIC) -> PlotData:
    """Prefer the same CSV as 2D plots; also support replay-only batch exports."""
    if (catalog.root / "summary.csv").is_file():
        return read_results(catalog.root, metric)
    points = []
    for entry in catalog.entries:
        metrics_path = entry.path.with_name("metrics.json")
        scenario_path = entry.path.with_name("scenario.json")
        if metrics_path.is_file() and scenario_path.is_file():
            metrics, scenario = read_json(metrics_path), read_json(scenario_path)
        else:
            replay = read_json(entry.path)
            metrics, scenario = replay["metrics"], replay["scenario"]
        points.append(_point(metrics, scenario, entry.map_key, metric))
    return PlotData(tuple(points), tuple(catalog.maps.items()), metric)


@dataclass
class Camera:
    yaw: float = -0.58
    pitch: float = 0.36
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    animation: tuple | None = None

    def rotate(self, dx: float, dy: float) -> None:
        self.animation = None
        self.yaw += dx * 0.008
        self.pitch = max(-math.pi / 2, min(math.pi / 2, self.pitch + dy * 0.008))

    def snap(self, face: str) -> None:
        yaw, pitch = FACE_VIEWS[face]
        # Take the short route even after repeated full rotations.
        yaw = self.yaw + (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.animation = (self.yaw, self.pitch, self.zoom, self.pan_x, self.pan_y, yaw, pitch, 0.0)

    def update(self, seconds: float) -> None:
        if self.animation is None:
            return
        yaw, pitch, zoom, px, py, target_yaw, target_pitch, elapsed = self.animation
        elapsed = min(0.55, elapsed + max(0, seconds))
        fraction = elapsed / 0.55
        eased = fraction * fraction * (3 - 2 * fraction)
        self.yaw = yaw + (target_yaw - yaw) * eased
        self.pitch = pitch + (target_pitch - pitch) * eased
        self.zoom = zoom + (1 - zoom) * eased
        self.pan_x, self.pan_y = px * (1 - eased), py * (1 - eased)
        self.animation = (
            None if fraction >= 1 else (yaw, pitch, zoom, px, py, target_yaw, target_pitch, elapsed)
        )

    def project(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, depth, z = point
        horizontal = math.cos(self.yaw) * x - math.sin(self.yaw) * depth
        away = math.sin(self.yaw) * x + math.cos(self.yaw) * depth
        vertical = math.cos(self.pitch) * z - math.sin(self.pitch) * away
        distance = math.cos(self.pitch) * away + math.sin(self.pitch) * z
        return horizontal, vertical, distance


class ResultsExplorer:
    def __init__(self, data: PlotData, loader=None) -> None:
        self.data = data
        self.loader = loader
        self.behavior = data.behaviors[0]
        self.camera = Camera()
        self.slices = False
        self.slice_zoom = 1.0
        self.slice_pan = (0.0, 0.0)
        self.hidden_maps: set[str] = set()
        self.legend_page = 0
        self.drag: tuple[int, tuple[int, int]] | None = None
        self.mouse: tuple[int, int] | None = None
        self.plot_rect = pygame.Rect(24, 158, 900, 620)
        self.hits: list[tuple[tuple[float, float], ResultPoint]] = []
        self.colors = {
            key: pygame.Color(PALETTE[i])[:3]
            if i < len(PALETTE)
            else tuple(
                round(c * 255) for c in colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.65, 0.75)
            )
            for i, (key, _) in enumerate(data.maps)
        }

    @classmethod
    def from_catalog(cls, catalog: ReplayCatalog) -> ResultsExplorer:
        return cls(catalog_data(catalog), lambda metric: catalog_data(catalog, metric))

    @classmethod
    def from_replay(cls, replay: dict) -> ResultsExplorer:
        scenario = replay["scenario"]
        key = scenario.get("source") or scenario["name"]

        def load(metric):
            return PlotData(
                (_point(replay["metrics"], scenario, key, metric),),
                ((key, scenario["name"]),),
                metric,
            )

        return cls(load(DEFAULT_METRIC), load)

    @property
    def points(self) -> list[ResultPoint]:
        return [
            p
            for p in self.data.points
            if p.behavior == self.behavior and p.map_key not in self.hidden_maps
        ]

    @property
    def robot_max(self) -> float:
        maximum = max((p.robot_disruption for p in self.data.points if p.completed), default=0)
        raw_step = max(maximum, 1) / 5
        power = 10 ** math.floor(math.log10(raw_step))
        step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw_step)
        return max(1, math.ceil(maximum / step) * step)

    @property
    def human_max(self) -> float:
        return max(1, max(p.disruption for p in self.data.points)) * 1.5

    @property
    def human_ticks(self) -> list[float]:
        if self.human_max <= 1.5:
            return [0, 0.5, 1]
        return [0] + [
            m * 10.0**e
            for e in range(math.ceil(math.log10(self.human_max)))
            for m in (1, 2, 5)
            if m * 10.0**e <= self.human_max
        ]

    def coordinates(self, point: ResultPoint) -> tuple[float, float, float]:
        return (
            2 * point.robot_disruption / self.robot_max - 1 if point.completed else 1.5,
            2 * ROBOT_PLOT_ORDER.index(point.planner) / (len(ROBOT_PLOT_ORDER) - 1) - 1,
            2 * human_scale(point.disruption) / human_scale(self.human_max) - 1,
        )

    def action(self, action: str) -> None:
        if action.startswith("face_"):
            self.slices = False
            self.camera.snap(action.removeprefix("face_"))
        elif action == "slices":
            self.slices = not self.slices
            self.drag = None
        elif action == "reset":
            self.camera = Camera()
            self.slice_zoom, self.slice_pan = 1.0, (0.0, 0.0)
        elif action in {"zoom_in", "zoom_out"}:
            self.wheel(1 if action == "zoom_in" else -1)
        elif action == "behavior":
            values = self.data.behaviors
            self.behavior = values[(values.index(self.behavior) + 1) % len(values)]
        elif action == "metric" and self.loader is not None:
            metrics = list(METRICS)
            self.data = self.loader(metrics[(metrics.index(self.data.metric) + 1) % len(metrics)])
        elif action == "legend_next":
            self.legend_page = (self.legend_page + 1) % math.ceil(len(self.data.maps) / 7)
        elif action == "show_maps":
            self.hidden_maps.clear()
        elif action.startswith("map_"):
            key = self.data.maps[int(action.removeprefix("map_"))][0]
            if key in self.hidden_maps:
                self.hidden_maps.remove(key)
            else:
                self.hidden_maps.add(key)
        self.hits = []

    def pointer_down(self, button: int, position: tuple[int, int], shift: bool = False) -> bool:
        if button in (1, 2, 3) and self.plot_rect.collidepoint(position):
            self.drag = (3 if shift or self.slices else button, position)
            self.camera.animation = None
            return True
        return False

    def pointer_move(self, position: tuple[int, int] | None) -> None:
        self.mouse = position
        if self.drag is None or position is None:
            return
        button, previous = self.drag
        dx, dy = position[0] - previous[0], position[1] - previous[1]
        if self.slices:
            self.slice_pan = (self.slice_pan[0] + dx, self.slice_pan[1] + dy)
        elif button == 1:
            self.camera.rotate(dx, dy)
        else:
            self.camera.pan_x += dx
            self.camera.pan_y += dy
        self.drag = (button, position)

    def wheel(self, amount: float) -> None:
        factor = 1.15 ** max(-10, min(10, amount))
        if self.slices:
            self.slice_zoom = max(0.45, min(6, self.slice_zoom * factor))
        else:
            self.camera.animation = None
            self.camera.zoom = max(0.35, min(4, self.camera.zoom * factor))

    def draw(self, surface: pygame.Surface, ui) -> None:
        self.plot_rect = pygame.Rect(24, 158, ui.width - 356, ui.height - 244)
        ui.text(surface, "INTERACTIVE RESULTS", (28, 18), 12, (27, 139, 112), True)
        ui.text(surface, "Disruption tradeoffs", (28, 40), 27, INK, True)
        ui.text(surface, BEHAVIOR_LABELS.get(self.behavior, self.behavior), (29, 79), 14, MUTED)
        ui.button(surface, "graph_behavior", f"Pedestrians: {self.behavior}  >", (28, 109, 255, 34))
        ui.button(
            surface,
            "graph_metric",
            "Metric: " + METRICS[self.data.metric] + "  >",
            (295, 109, ui.width - 651, 34),
        )
        sidebar = ui.width - 312
        pygame.draw.rect(surface, (255, 255, 255), self.plot_rect, border_radius=14)
        pygame.draw.rect(
            surface, (255, 255, 255), (sidebar, 109, 288, ui.height - 195), border_radius=14
        )
        self.hits = []
        if self.slices:
            self._draw_slices(surface, ui)
        else:
            self._draw_cube(surface, ui)
        ui.text(surface, "VIEW", (sidebar + 18, 126), 12, MUTED, True)
        ui.button(
            surface,
            "graph_slices",
            "Slices by awareness" if not self.slices else "Back to cube",
            (sidebar + 16, 151, 256, 34),
            self.slices,
        )
        ui.text(surface, "SNAP TO TWO AXES", (sidebar + 18, 202), 12, MUTED, True)
        for i, (face, label) in enumerate(
            (
                ("front", "Robot + human disruption"),
                ("side", "Awareness + human disruption"),
                ("top", "Robot disruption + awareness"),
            )
        ):
            ui.button(surface, f"graph_face_{face}", label, (sidebar + 16, 226 + i * 40, 256, 33))
        ui.button(surface, "graph_zoom_out", "-", (sidebar + 16, 355, 45, 32))
        ui.button(surface, "graph_reset", "Reset view", (sidebar + 70, 355, 147, 32))
        ui.button(surface, "graph_zoom_in", "+", (sidebar + 226, 355, 46, 32))
        ui.text(surface, "MAP COLORS  /  CLICK TO HIDE", (sidebar + 18, 410), 11, MUTED, True)
        start = self.legend_page * 7
        for row, (key, label) in enumerate(self.data.maps[start : start + 7]):
            index, y = start + row, 439 + row * 42
            ui.button(surface, f"graph_map_{index}", "", (sidebar + 16, y, 256, 35))
            color = self.colors[key] if key not in self.hidden_maps else (199, 207, 215)
            pygame.draw.circle(surface, color, (sidebar + 27, y + 17), 5)
            ui.text(
                surface,
                ui.fit_text(label, 224, 13),
                (sidebar + 40, y + 8),
                13,
                INK if key not in self.hidden_maps else MUTED,
            )
        y = min(ui.height - 132, 439 + min(7, len(self.data.maps)) * 42 + 8)
        ui.button(surface, "graph_show_maps", "Show all maps", (sidebar + 16, y, 151, 29))
        if len(self.data.maps) > 7:
            ui.button(surface, "graph_legend_next", "More >", (sidebar + 178, y, 94, 29))
        ui.text(
            surface,
            "Drag: rotate   |   Right / Shift-drag: pan   |   Wheel: zoom   |   Hover: values",
            (29, ui.height - 66),
            13,
            MUTED,
        )
        ui.text(
            surface,
            "Human axis: log above 1, linear near zero. Robot axis: extra ticks. "
            "Unfinished robots sit beyond the numeric X range.",
            (29, ui.height - 44),
            12,
            MUTED,
        )
        ui.text(
            surface,
            "Exact overlapping points are spread a few pixels. Slices share scales, zoom, and pan.",
            (29, ui.height - 24),
            12,
            MUTED,
        )
        if not self.points:
            ui.text(
                surface,
                "No visible runs. Show a map or choose another pedestrian behavior.",
                (self.plot_rect.x + 40, self.plot_rect.centery),
                15,
                MUTED,
            )
        self._tooltip(surface, ui)

    def _draw_cube(self, surface: pygame.Surface, ui) -> None:
        rect = self.plot_rect
        scale = min(rect.width / 4.8, rect.height / 3.8) * self.camera.zoom

        def project(p):
            x, y, distance = self.camera.project(p)
            return (
                rect.centerx - 26 + self.camera.pan_x + x * scale,
                rect.centery + self.camera.pan_y - y * scale,
                distance,
            )

        old_clip = surface.get_clip()
        surface.set_clip(rect.inflate(-16, -16))

        def line(a, b, color=(220, 229, 237), width=1):
            pygame.draw.line(surface, color, project(a)[:2], project(b)[:2], width)

        # A pale disjoint plane extends the X axis without inventing failed-run delays.
        polygon = [project((1.5, y, z))[:2] for y, z in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pygame.draw.polygon(surface, (255, 246, 233), polygon)
        pygame.draw.polygon(surface, (235, 211, 176), polygon, 1)
        for depth in (-1, 1):
            for z in (-1, 1):
                line((-1, depth, z), (1, depth, z))
        for x in (-1, 1):
            for z in (-1, 1):
                line((x, -1, z), (x, 1, z))
            for depth in (-1, 1):
                line((x, depth, -1), (x, depth, 1))
        for value in self.human_ticks:
            z = 2 * human_scale(value) / human_scale(self.human_max) - 1
            line((-1, -1, z), (1, -1, z), (234, 239, 244))
            a = project((-1, -1, z))
            if abs(math.cos(self.camera.pitch)) > 0.2:
                ui.text(surface, f"{value:g}", (int(a[0]) - 38, int(a[1]) - 9), 11, MUTED)
        origin = (-1, -1, -1)
        for endpoint in ((1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            line(origin, endpoint, (132, 153, 174), 2)
        if abs(math.cos(self.camera.yaw)) > 0.2:
            for i in range(6):
                value = self.robot_max * i / 5
                a = project((2 * i / 5 - 1, -1, -1))
                ui.text(surface, f"{value:g}", (int(a[0]) - 5, int(a[1]) + 10), 11, MUTED)
            a = project((0, -1, -1))
            ui.text(
                surface, "Robot disruption (ticks)", (int(a[0]) - 67, int(a[1]) + 32), 12, INK, True
            )
        if abs(math.sin(self.camera.yaw)) + abs(math.sin(self.camera.pitch)) > 0.2:
            for i, level in enumerate(ROBOT_PLOT_ORDER):
                depth = 2 * i / (len(ROBOT_PLOT_ORDER) - 1) - 1
                a = project((-1, depth, -1))
                ui.text(surface, level, (int(a[0]) - 45, int(a[1]) + 5), 11, MUTED)
            a = project((-1, 1, -1))
            ui.text(surface, "Human awareness", (int(a[0]) - 70, int(a[1]) - 22), 12, INK, True)
        if abs(math.cos(self.camera.pitch)) > 0.2:
            a = project((-1, -1, 1))
            ui.text(surface, "Human disruption", (int(a[0]) - 45, int(a[1]) - 41), 12, INK, True)
            ui.text(surface, "log scale", (int(a[0]) - 20, int(a[1]) - 24), 11, MUTED)
        a = project((1.5, 0, 1))
        ui.text(surface, "Unfinished", (int(a[0]) - 30, int(a[1]) - 23), 12, (148, 100, 39), True)
        projected = [(project(self.coordinates(p)), p) for p in self.points]
        self._dots(surface, projected)
        surface.set_clip(old_clip)
        lengths = [
            math.hypot(*self.camera.project(tuple(1 if i == axis else 0 for i in range(3)))[:2])
            for axis in range(3)
        ]
        collapsed = min(range(3), key=lengths.__getitem__)
        label = "CUBE  /  7 awareness levels"
        if lengths[collapsed] < 0.05:
            label = (
                "Awareness + human disruption  /  Robot axis collapsed",
                "Robot + human disruption  /  Awareness axis collapsed",
                "Robot disruption + awareness  /  Human axis collapsed",
            )[collapsed]
        ui.text(surface, label, (rect.x + 18, rect.y + 13), 11, MUTED, True)

    def _draw_slices(self, surface: pygame.Surface, ui) -> None:
        rect = self.plot_rect
        columns, rows = 3, math.ceil(len(ROBOT_PLOT_ORDER) / 3)
        width, height = (rect.width - 24) / columns, (rect.height - 24) / rows
        for index, planner in enumerate(ROBOT_PLOT_ORDER):
            cell = pygame.Rect(
                rect.x + 12 + (index % columns) * width,
                rect.y + 12 + (index // columns) * height,
                width - 10,
                height - 10,
            )
            pygame.draw.rect(surface, (248, 250, 253), cell, border_radius=8)
            ui.text(surface, planner, (cell.x + 10, cell.y + 5), 13, BLUE, True)
            inner = pygame.Rect(cell.x + 37, cell.y + 28, cell.width - 53, cell.height - 65)
            axis_width = inner.width * 0.68

            def project(x, z):
                return (
                    inner.x
                    + 12
                    + axis_width / 2
                    + (x - 0.5) * axis_width * self.slice_zoom
                    + self.slice_pan[0],
                    inner.centery
                    - (z - 0.5) * (inner.height - 20) * self.slice_zoom
                    + self.slice_pan[1],
                    0,
                )

            old_clip = surface.get_clip()
            surface.set_clip(inner.inflate(12, 12))
            for value in self.human_ticks:
                z = human_scale(value) / human_scale(self.human_max)
                y = project(0, z)[1]
                pygame.draw.line(surface, (224, 232, 240), (inner.x, y), (inner.right, y))
            failure_x = project(1.25, 0)[0]
            pygame.draw.rect(surface, (255, 242, 223), (failure_x - 8, inner.y, 16, inner.height))
            projected = [
                (
                    project(
                        (p.robot_disruption / self.robot_max) if p.completed else 1.25,
                        human_scale(p.disruption) / human_scale(self.human_max),
                    ),
                    p,
                )
                for p in self.points
                if p.planner == planner
            ]
            self._dots(surface, projected, radius=4)
            surface.set_clip(old_clip)
            for value in self.human_ticks:
                y = project(0, human_scale(value) / human_scale(self.human_max))[1]
                if inner.top <= y <= inner.bottom:
                    ui.text(surface, f"{value:g}", (cell.x + 3, int(y) - 7), 9, MUTED)
            for fraction in (0, 0.5, 1):
                x = project(fraction, 0)[0]
                if inner.left <= x <= inner.right:
                    ui.text(
                        surface,
                        f"{self.robot_max * fraction:g}",
                        (int(x) - 3, inner.bottom + 5),
                        10,
                        MUTED,
                    )
            if inner.left <= failure_x <= inner.right:
                ui.text(surface, "DNF", (int(failure_x) - 9, inner.bottom + 5), 9, (148, 100, 39))
            ui.text(
                surface,
                "Robot delay  /  Human delay (log)",
                (cell.x + 20, cell.bottom - 17),
                10,
                MUTED,
            )
            if not projected:
                ui.text(surface, "No runs", (inner.x + 25, inner.centery - 6), 12, MUTED)

    def _dots(self, surface, projected, radius=6) -> None:
        groups = defaultdict(list)
        for position, point in projected:
            groups[(round(position[0], 2), round(position[1], 2))].append((position, point))
        visible = []
        for group in groups.values():
            for index, (position, point) in enumerate(group):
                angle = index * 2 * math.pi / len(group)
                spread = min(14, 3 + len(group) * 1.5) if len(group) > 1 else 0
                x, y = (
                    position[0] + math.cos(angle) * spread,
                    position[1] + math.sin(angle) * spread,
                )
                visible.append(((x, y, position[2]), point))
        # Farther points first. Hover uses reverse paint order for coincident projections.
        for (x, y, _), point in sorted(visible, key=lambda item: item[0][2], reverse=True):
            center = (round(x), round(y))
            if not surface.get_clip().collidepoint(center):
                continue
            color = self.colors[point.map_key]
            if point.completed:
                pygame.draw.circle(surface, color, center, radius)
                pygame.draw.circle(surface, (255, 255, 255), center, radius, 1)
            else:
                for sign in (-1, 1):
                    pygame.draw.line(
                        surface,
                        color,
                        (x - radius, y - sign * radius),
                        (x + radius, y + sign * radius),
                        2,
                    )
            self.hits.append(((x, y), point))

    def _tooltip(self, surface, ui) -> None:
        if self.mouse is None or self.drag is not None:
            return
        nearest = [(math.dist(self.mouse, xy), p) for xy, p in reversed(self.hits)]
        nearest = sorted(nearest, key=lambda item: item[0])
        if not nearest or nearest[0][0] > 13:
            return
        point = nearest[0][1]
        labels = dict(self.data.maps)
        lines = textwrap.wrap(labels[point.map_key], 43) + [
            f"{point.planner}  /  {point.behavior}",
            f"Human disruption: {point.disruption:g}",
            f"Robot disruption: {point.robot_disruption:g} ticks"
            if point.completed
            else "Robot: did not finish",
        ]
        if point.horizon_limited:
            lines.append("Human trips include horizon-limited delay")
        box = pygame.Rect(
            min(ui.width - 350, self.mouse[0] + 14),
            min(ui.height - len(lines) * 21 - 25, self.mouse[1] + 16),
            332,
            len(lines) * 21 + 16,
        )
        pygame.draw.rect(surface, (32, 48, 65), box, border_radius=8)
        for i, line in enumerate(lines):
            ui.text(surface, line, (box.x + 12, box.y + 8 + i * 21), 12, (255, 255, 255))
