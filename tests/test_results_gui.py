from __future__ import annotations

import math
from dataclasses import replace

import pygame
import pytest

from human_nav.config import BEHAVIORS
from human_nav.plotting import DEFAULT_METRIC, ROBOT_PLOT_ORDER, PlotData, ResultPoint
from human_nav.results_gui import Camera, ResultsExplorer, human_scale


@pytest.fixture
def plot_data():
    maps = (("a", "Corridor"), ("b", "Intersection"))
    points = tuple(
        ResultPoint(
            key,
            planner,
            behavior,
            0 if map_index else level * 10,
            not (map_index == level == 0),
            None if map_index == level == 0 else level,
            False,
            4,
            level + 4,
        )
        for behavior in BEHAVIORS
        for map_index, (key, _) in enumerate(maps)
        for level, planner in enumerate(ROBOT_PLOT_ORDER)
    )
    return PlotData(points, maps, DEFAULT_METRIC)


def test_vertical_transform_preserves_zero_and_logarithmic_spacing():
    assert human_scale(0) == 0
    assert human_scale(0.5) == pytest.approx(human_scale(1) / 2)
    assert human_scale(10) - human_scale(1) == pytest.approx(human_scale(100) - human_scale(10))


def test_cube_coordinates_keep_failure_separate_and_awareness_categorical(plot_data):
    view = ResultsExplorer(plot_data)
    failure = view.points[0]
    assert view.coordinates(failure)[0] > 1
    assert failure.robot_disruption is None
    for point in view.points:
        x, depth, human = view.coordinates(point)
        if point.completed:
            assert -1 <= x <= 1
        assert depth == pytest.approx(2 * ROBOT_PLOT_ORDER.index(point.planner) / 6 - 1)
        assert -1 <= human <= 1
    assert view.coordinates(view.points[-1])[2] == -1  # A measured zero remains zero.


@pytest.mark.parametrize("face,collapsed", [("front", 1), ("side", 0), ("top", 2)])
def test_face_buttons_animate_then_collapse_exactly_one_axis(face, collapsed):
    camera = Camera()
    camera.zoom, camera.pan_x = 2, 50
    before = (camera.yaw, camera.pitch)
    camera.snap(face)
    assert (camera.yaw, camera.pitch) == before
    camera.update(0.2)
    assert camera.animation is not None
    assert (camera.yaw, camera.pitch) != before
    camera.update(1)
    assert camera.animation is None
    assert camera.zoom == 1 and camera.pan_x == 0
    origin = camera.project((0, 0, 0))
    for axis in range(3):
        vector = tuple(1 if i == axis else 0 for i in range(3))
        projected = camera.project(vector)
        length = math.dist(origin[:2], projected[:2])
        assert length == pytest.approx(0 if axis == collapsed else 1, abs=1e-8)


def test_snapping_uses_shortest_rotation_and_drag_interrupts_animation():
    camera = Camera(yaw=math.pi * 8 + 0.1)
    camera.snap("front")
    camera.update(1)
    assert camera.yaw == pytest.approx(math.pi * 8)
    camera.snap("side")
    camera.rotate(20, 10)
    assert camera.animation is None


def test_pan_rotate_zoom_and_slice_controls_do_not_change_measurements(plot_data):
    view = ResultsExplorer(plot_data)
    center = view.plot_rect.center
    assert view.pointer_down(1, center)
    yaw = view.camera.yaw
    view.pointer_move((center[0] + 30, center[1] + 10))
    assert view.camera.yaw > yaw
    view.drag = None
    view.pointer_down(3, center)
    view.pointer_move((center[0] + 15, center[1] - 8))
    assert (view.camera.pan_x, view.camera.pan_y) == (15, -8)
    view.wheel(2)
    assert view.camera.zoom > 1
    view.action("slices")
    assert view.slices and view.drag is None
    view.pointer_down(1, center)
    view.pointer_move((center[0] + 20, center[1] + 15))
    assert view.slice_pan == (20, 15)
    view.wheel(2)
    assert view.slice_zoom > 1
    assert view.data == plot_data
    view.action("reset")
    assert view.slice_zoom == 1 and view.slice_pan == (0, 0)


def test_map_visibility_behavior_and_metric_preserve_shared_scales(plot_data):
    view = ResultsExplorer(plot_data, lambda metric: replace(plot_data, metric=metric))
    colors = view.colors.copy()
    limits = view.robot_max, view.human_max
    view.action("map_0")
    assert all(p.map_key == "b" for p in view.points)
    view.action("behavior")
    assert view.behavior == BEHAVIORS[1]
    assert (view.robot_max, view.human_max) == limits
    assert view.colors == colors
    view.action("show_maps")
    assert len(view.points) == 14
    view.action("metric")
    assert view.data.metric != DEFAULT_METRIC
    view.action("face_top")
    assert not view.slices


def test_zoom_bounds_and_outside_clicks(plot_data):
    view = ResultsExplorer(plot_data)
    assert not view.pointer_down(1, (1, 1))
    for _ in range(30):
        view.wheel(10)
    assert view.camera.zoom == 4
    for _ in range(30):
        view.wheel(-10)
    assert view.camera.zoom == 0.35


@pytest.mark.parametrize("completed", [True, False])
def test_all_zero_success_or_failure_renders_without_losing_points(plot_data, completed):
    # The drawing adapter is a normal viewer; no display window is required.
    from human_nav.config import parse_scenario
    from human_nav.experiments import make_replay, paired_run
    from human_nav.gui import ReplayViewer

    scenario = parse_scenario({"map": ["...."], "robot": {"start": [0, 0], "goal": [3, 0]}})
    ui = ReplayViewer(make_replay(*paired_run(scenario, "L0")), size=(1260, 904))
    data = replace(
        plot_data,
        points=tuple(
            replace(p, completed=completed, disruption=0, robot_disruption=0 if completed else None)
            for p in plot_data.points
        ),
    )
    view = ResultsExplorer(data)
    pygame.font.init()
    surface = pygame.Surface((ui.width, ui.height))
    for slices in (False, True):
        view.slices = slices
        view.draw(surface, ui)
        assert len(view.hits) == len(view.points)
        assert surface.get_clip() == surface.get_rect()
    view.mouse = tuple(round(v) for v in view.hits[0][0])
    view.draw(surface, ui)  # Hover readout must also accept zero/unfinished values.
