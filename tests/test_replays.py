from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from human_nav.cli import main
from human_nav.config import BEHAVIORS, PLANNERS, parse_scenario
from human_nav.experiments import make_replay, paired_run
from human_nav.gui import ReplayViewer
from human_nav.replays import ReplayCatalog


@pytest.fixture
def sample_replay():
    scenario = parse_scenario(
        {
            "name": "Map A",
            "map": ["#######", "#.....#", "#.....#", "#.....#", "#######"],
            "robot": {"start": [1, 2], "goal": [5, 2]},
            "pedestrians": [{"id": "person", "start": [3, 1], "goal": [3, 3]}],
            "simulation": {"max_steps": 15, "search_horizon": 10},
        }
    )
    simulation, baseline, metrics = paired_run(scenario, "L0")
    return make_replay(simulation, baseline, metrics)


def write_replay(folder: Path, replay: dict, sidecars: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "replay.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    if sidecars:
        for name, field in (("metrics", "metrics"), ("scenario", "scenario")):
            (folder / f"{name}.json").write_text(json.dumps(replay[field]), encoding="utf-8")
    return path


@pytest.fixture
def batch_folder(sample_replay, tmp_path):
    root = tmp_path / "batch"
    for map_index in range(2):
        for behavior in BEHAVIORS:
            for planner in PLANNERS:
                replay = deepcopy(sample_replay)
                replay["scenario"]["name"] = f"Map {'AB'[map_index]}"
                replay["scenario"]["source"] = f"/maps/{map_index}.yaml"
                replay["metrics"].update(behavior=behavior, planner=planner)
                write_replay(root / f"{map_index:02d}-map" / behavior / planner, replay)
    return root


def test_catalog_indexes_full_batch_and_loads_timelines_lazily(batch_folder, monkeypatch):
    import human_nav.replays as replays

    calls = []
    original = replays.read_json

    def tracked(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(replays, "read_json", tracked)
    catalog = ReplayCatalog.open(batch_folder)
    assert catalog.is_batch
    assert len(catalog.entries) == 42
    assert "replay.json" not in calls
    assert catalog.load(0)["metrics"]["planner"] == "L0"
    assert calls.count("replay.json") == 1
    assert catalog.options("behavior", 0) == [(b, b) for b in BEHAVIORS]
    assert catalog.options("planner", 0) == [(p, p) for p in PLANNERS]


def test_results_explorer_loads_replay_only_batch_without_timelines(batch_folder, monkeypatch):
    from human_nav import results_gui

    catalog = ReplayCatalog.open(batch_folder)
    calls = []
    original = results_gui.read_json

    def tracked(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(results_gui, "read_json", tracked)
    explorer = results_gui.ResultsExplorer.from_catalog(catalog)
    assert len(explorer.data.points) == 42
    assert "replay.json" not in calls
    assert len(explorer.colors) == 2
    before = explorer.colors.copy()
    explorer.action("metric")
    assert explorer.colors == before


def test_results_controls_work_through_resized_window_and_return_to_replay(batch_folder):
    import pygame

    pygame.font.init()
    viewer = ReplayViewer.from_path(batch_folder)
    viewer.playhead = 2
    viewer.speed = 4
    surface = pygame.Surface((900, 650))
    viewer.draw_window(surface)

    def click(action):
        viewport = viewer.viewport(surface.get_size())
        button = viewer.buttons[action]
        physical = (
            round(viewport.x + button.centerx * viewport.width / viewer.width),
            round(viewport.y + button.centery * viewport.height / viewer.height),
        )
        viewer.handle_window_click(physical, surface.get_size())
        viewer.draw_window(surface)

    click("results")
    assert viewer.show_results
    assert len(viewer.explorer.hits) == len(viewer.explorer.points)
    click("graph_slices")
    assert viewer.explorer.slices
    assert len(viewer.explorer.hits) == len(viewer.explorer.points)
    click("graph_face_front")
    assert not viewer.explorer.slices
    viewer.explorer.camera.update(1)
    assert viewer.explorer.camera.pitch == 0
    click("results")
    assert not viewer.show_results
    assert viewer.playhead == 2 and viewer.speed == 4


def test_results_mouse_event_loop_maps_rotation_pan_and_wheel(batch_folder, monkeypatch):
    import pygame

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    viewer = ReplayViewer.from_path(batch_folder)
    viewer.window_size = (900, 650)
    viewer.action("results")
    viewer.playhead = 1
    viewport = viewer.viewport(viewer.window_size)
    center = viewer.explorer.plot_rect.center
    point = (
        int(viewport.x + center[0] * viewport.width / viewer.width),
        int(viewport.y + center[1] * viewport.height / viewer.height),
    )
    moved = (point[0] + 25, point[1] + 10)
    events = [
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point),
        pygame.event.Event(pygame.MOUSEMOTION, pos=moved),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=moved),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=point),
        pygame.event.Event(pygame.MOUSEMOTION, pos=moved),
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=moved),
        pygame.event.Event(pygame.MOUSEWHEEL, y=2),
    ]
    frames = iter([events, [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(frames))
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: point)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    viewer.run()
    assert viewer.explorer.camera.yaw > -0.58
    assert viewer.explorer.camera.pan_x > 0
    assert viewer.explorer.camera.zoom > 1
    assert viewer.explorer.drag is None
    assert viewer.playhead == 1  # Replay time pauses while inspecting results.


def test_switch_dimensions_preserves_other_selections(batch_folder):
    catalog = ReplayCatalog.open(batch_folder)
    index = catalog.select("planner", "L4+D", 0)
    index = catalog.select("behavior", "patient", index)
    index = catalog.select("map", "/maps/1.yaml", index)
    entry = catalog.entries[index]
    assert (entry.map_name, entry.behavior, entry.planner) == ("Map B", "patient", "L4+D")


def test_incomplete_batch_offers_only_existing_combinations(sample_replay, tmp_path):
    root = tmp_path / "batch"
    for map_key, behavior, planner in (("a", "fixed", "L0"), ("b", "patient", "L4")):
        replay = deepcopy(sample_replay)
        replay["scenario"].update(name=f"Map {map_key}", source=map_key)
        replay["metrics"].update(behavior=behavior, planner=planner)
        write_replay(root / map_key / behavior / planner, replay)
    catalog = ReplayCatalog.open(root)
    index = catalog.select("map", "b", 0)
    assert catalog.options("behavior", index) == [("patient", "patient")]
    assert catalog.options("planner", index) == [("L4", "L4")]
    with pytest.raises(ValueError, match="No saved replay"):
        catalog.select("planner", "L0", index)


def test_single_replay_file_and_run_directory_still_open(sample_replay, tmp_path):
    path = write_replay(tmp_path / "run", sample_replay, sidecars=False)
    for source in (path, path.parent):
        catalog = ReplayCatalog.open(source)
        assert not catalog.is_batch
        assert len(catalog.entries) == 1
        assert catalog.load(0) == json.loads(json.dumps(sample_replay))
        assert ReplayViewer.from_path(source).height == 800


def test_arrived_figures_follow_recorded_time_and_rewind(sample_replay):
    viewer = ReplayViewer(sample_replay)
    assert viewer.arrived_humans(1.9) == {}
    assert viewer.arrived_humans(2) == {"person": (3, 3)}
    assert viewer.arrived_humans(100) == {"person": (3, 3)}
    assert "person" not in viewer.interpolated(viewer.frames, 2)
    viewer.playhead = 100
    viewer.action("reset")
    assert viewer.arrived_humans() == {}
    assert "robot" not in viewer.arrived_humans(100)


def test_robot_can_cross_faded_human_destination_and_is_drawn_in_front():
    import pygame

    from human_nav.gui import BLUE

    scenario = parse_scenario(
        {
            "map": ["#######", "#.....#", "#.....#", "#.....#", "#######"],
            "robot": {"start": [1, 2], "goal": [5, 2]},
            "pedestrians": [{"id": "person", "start": [3, 1], "goal": [3, 2]}],
            "simulation": {"max_steps": 15},
        }
    )
    simulation, baseline, metrics = paired_run(scenario, "L0")
    assert simulation.agents["person"].arrival_time == 1
    assert simulation.agents["person"].position is None
    assert simulation.history[2]["positions"]["robot"] == [3, 2]
    assert simulation.agents["robot"].arrival_time == 4
    viewer = ReplayViewer(make_replay(simulation, baseline, metrics))
    viewer.show_path = viewer.show_halos = False
    pygame.font.init()
    surface = pygame.Surface((viewer.width, viewer.height))
    viewer.playhead = 2
    viewer.draw(surface)
    destination = viewer.center((3, 2))
    assert surface.get_at(destination)[:3] == BLUE
    viewer.playhead = 3
    viewer.draw(surface)
    faded = surface.get_at(destination)[:3]
    assert faded != BLUE and faded != (247, 250, 252)
    assert max(faded) - min(faded) < 35


def test_multiple_pedestrians_can_complete_at_the_same_background_cell():
    scenario = parse_scenario(
        {
            "map": ["#######", "#.....#", "#.....#", "#.....#", "#######"],
            "robot": {"start": [1, 1], "goal": [5, 1]},
            "pedestrians": [
                {"id": "first", "start": [2, 2], "goal": [3, 2]},
                {"id": "second", "start": [1, 2], "goal": [3, 2]},
            ],
            "simulation": {"max_steps": 15},
        }
    )
    simulation, baseline, metrics = paired_run(scenario, "L0")
    assert [simulation.agents[p].arrival_time for p in ("first", "second")] == [1, 2]
    viewer = ReplayViewer(make_replay(simulation, baseline, metrics))
    assert viewer.arrived_humans(2) == {"first": (3, 2), "second": (3, 2)}


def test_empty_folder_and_duplicate_batches_fail_clearly(batch_folder, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No replay.json"):
        ReplayCatalog.open(empty)
    path = next(batch_folder.rglob("replay.json"))
    write_replay(batch_folder / "duplicate" / "run", json.loads(path.read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="Open one batch directory"):
        ReplayCatalog.open(batch_folder)


def test_viewer_switches_without_losing_preferences_and_rewinds(batch_folder):
    viewer = ReplayViewer.from_path(batch_folder)
    viewer.speed = 4
    viewer.show_baseline = True
    viewer.show_halos = False
    viewer.playing = False
    viewer.playhead = 2.5
    viewer.action("next_replay")
    assert viewer.replay_index == 1
    assert viewer.metrics["planner"] == "L1"
    assert viewer.playhead == 0
    assert viewer.speed == 4 and viewer.show_baseline and not viewer.show_halos
    assert not viewer.playing
    viewer.action("previous_replay")
    viewer.action("previous_replay")
    assert viewer.replay_index == len(viewer.catalog.entries) - 1


def test_gui_selectors_and_layout_render_and_accept_clicks(batch_folder, tmp_path):
    import pygame

    pygame.font.init()
    viewer = ReplayViewer.from_path(batch_folder)
    surface = pygame.Surface((viewer.width, viewer.height))
    viewer.draw(surface)
    assert viewer.buttons["select_map"].bottom < viewer.map_y
    viewer.handle_click(viewer.buttons["select_planner"].center)
    viewer.draw(surface)
    assert viewer.dropdown == "planner"
    target = next(box for value, box in viewer.menu_boxes if value == "L4+D")
    viewer.handle_click(target.center)
    assert viewer.metrics["planner"] == "L4+D"
    assert viewer.dropdown is None
    viewer.draw(surface)
    viewer.handle_click(viewer.buttons["select_map"].center)
    viewer.draw(surface)
    target = next(box for value, box in viewer.menu_boxes if value == "/maps/1.yaml")
    viewer.handle_click(target.center)
    assert viewer.scenario["name"] == "Map B"
    assert viewer.metrics["planner"] == "L4+D"
    viewer.screenshot(tmp_path / "batch.png", 1.5)
    assert (tmp_path / "batch.png").stat().st_size > 1000


def test_corrupt_replay_does_not_replace_current_view(batch_folder):
    viewer = ReplayViewer.from_path(batch_folder)
    viewer.catalog.entries[1].path.write_text("invalid json", encoding="utf-8")
    viewer.switch_replay(1)
    assert viewer.replay_index == 0
    assert viewer.metrics["planner"] == "L0"
    assert "Could not open replay" in viewer.notice


def test_open_batch_from_single_viewer_and_keep_settings(sample_replay, batch_folder):
    viewer = ReplayViewer(sample_replay)
    viewer.speed = 2
    viewer.window_size = (900, 640)
    viewer.open_path(batch_folder)
    assert viewer.catalog.is_batch
    assert viewer.height == 904
    assert viewer.speed == 2
    assert viewer.content_offset == 104
    assert viewer.window_size == (900, 640)


@pytest.mark.parametrize("size", [(900, 640), (800, 600), (1100, 700), (1600, 900)])
def test_resized_view_fits_all_controls_and_maps_clicks(batch_folder, size):
    import pygame

    pygame.font.init()
    viewer = ReplayViewer.from_path(batch_folder)
    surface = pygame.Surface(size)
    viewer.draw_window(surface)
    viewport = viewer.viewport(size)
    assert surface.get_rect().contains(viewport)
    assert viewport.width / viewport.height == pytest.approx(
        viewer.width / viewer.height, abs=0.005
    )
    button = viewer.buttons["select_planner"]
    center = (
        round(viewport.x + button.centerx * viewport.width / viewer.width),
        round(viewport.y + button.centery * viewport.height / viewer.height),
    )
    viewer.handle_window_click(center, size)
    assert viewer.dropdown == "planner"
    viewer.draw_window(surface)
    target = next(box for value, box in viewer.menu_boxes if value == "L4+D")
    center = (
        round(viewport.x + target.centerx * viewport.width / viewer.width),
        round(viewport.y + target.centery * viewport.height / viewer.height),
    )
    viewer.handle_window_click(center, size)
    assert viewer.metrics["planner"] == "L4+D"
    assert viewer.window_size == size


def test_letterbox_clicks_do_not_activate_controls(batch_folder):
    viewer = ReplayViewer.from_path(batch_folder)
    viewer.playing = False
    assert viewer.window_to_canvas((0, 0), (1600, 900)) is None
    viewer.handle_window_click((0, 0), (1600, 900))
    assert not viewer.playing


def test_initial_window_fits_smaller_desktop():
    assert ReplayViewer.initial_window_size((1920, 1080)) == (1100, 760)
    assert ReplayViewer.initial_window_size((1024, 768)) == (944, 668)


def test_replay_cli_opens_batch_without_simulation(batch_folder, monkeypatch):
    observed = []
    monkeypatch.setattr(ReplayViewer, "run", lambda self: observed.append(self))
    assert main(["replay", str(batch_folder), "--paused", "--tick", "1.5"]) == 0
    assert len(observed[0].catalog.entries) == 42
    assert observed[0].playhead == 1.5
    assert not observed[0].playing


def test_batch_keyboard_navigation_in_event_loop(batch_folder, monkeypatch):
    import pygame

    viewer = ReplayViewer.from_path(batch_folder)
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    mode_requests = []
    original_set_mode = pygame.display.set_mode

    def tracked_mode(size, flags=0):
        mode_requests.append((size, flags))
        return original_set_mode(size, flags)

    monkeypatch.setattr(pygame.display, "set_mode", tracked_mode)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600, size=(800, 600)),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PAGEDOWN),
            pygame.event.Event(pygame.QUIT),
        ],
    )
    viewer.run()
    assert viewer.replay_index == 1
    assert viewer.window_size == (800, 600)
    assert all(flags & pygame.RESIZABLE for _, flags in mode_requests)
