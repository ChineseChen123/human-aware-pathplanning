from __future__ import annotations

import json
from copy import deepcopy

import pygame
import pytest

from human_nav.config import parse_scenario
from human_nav.engine import Simulation
from human_nav.experiments import make_replay, paired_run, save_run
from human_nav.gui import PLAN_COLOR, ReplayViewer
from human_nav.plan_replay import PlanRebuilder, recorded_scenario


@pytest.fixture
def scenario():
    return parse_scenario(
        {
            "name": "Plan replay test",
            "map": ["#######", "#.....#", "#.....#", "#.....#", "#######"],
            "robot": {"start": [1, 2], "goal": [5, 2]},
            "pedestrians": [{"id": "person", "start": [3, 1], "goal": [3, 3]}],
            "simulation": {"max_steps": 20, "search_horizon": 15},
        }
    )


@pytest.fixture
def replay(scenario):
    return make_replay(*paired_run(scenario, "L4+D"))


@pytest.mark.parametrize("interval,spawn", [(1, 0), (2, 0), (2, 3)])
def test_trace_captures_each_actual_planning_decision(scenario, interval, spawn):
    from dataclasses import replace

    scenario = replace(
        scenario, robot=replace(scenario.robot, move_interval=interval, spawn_time=spawn)
    )
    world = Simulation(scenario, "L4+D").run()
    assert len(world.robot_plans) == world.planner.stats.calls
    assert [p["sequence"] for p in world.robot_plans] == list(range(1, len(world.robot_plans) + 1))
    assert world.robot_plans[0]["tick"] == spawn
    for plan in world.robot_plans:
        tick = plan["tick"]
        assert (tick - spawn) % interval == 0
        assert plan["path"][0] == world.history[tick]["positions"][scenario.robot.id]
        assert plan["proposed"] == (plan["path"][1] if len(plan["path"]) > 1 else plan["path"][0])
        after = world.history[tick + 1]
        actual = after["positions"].get(scenario.robot.id, list(scenario.robot.goals[-1]))
        assert plan["executed"] == actual
    before = deepcopy(world.robot_plans)
    world.planner.last_path.clear()
    assert world.robot_plans == before  # Every decision owns its saved path.


def test_blocked_plan_records_proposed_and_executed_moves_separately():
    scenario = parse_scenario(
        {
            "map": ["#####", "#...#", "#####"],
            "robot": {"start": [1, 1], "goal": [3, 1]},
            "pedestrians": [{"id": "person", "start": [2, 1], "goal": [1, 1]}],
            "simulation": {"max_steps": 4},
        }
    )
    world = Simulation(scenario, "L0").run()
    assert world.robot_plans[0]["proposed"] == [2, 1]
    assert world.robot_plans[0]["executed"] == [1, 1]


def test_no_record_mode_and_forecasts_do_not_collect_plan_traces(scenario):
    world = Simulation(scenario, record=False).run()
    assert world.robot_plans == []
    assert world.history == []
    other = Simulation(scenario)
    other.forecast(5)
    assert other.robot_plans == [] and other.planner.stats.calls == 0


def test_trace_round_trips_through_saved_replay(scenario, tmp_path):
    world, baseline, metrics = paired_run(scenario, "L4+D")
    path = save_run(tmp_path, world, baseline, metrics)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["robot_plans"] == world.robot_plans
    viewer = ReplayViewer.from_path(path)
    assert viewer.current_plan() == raw["robot_plans"][0]


def test_overlay_follows_plan_at_tick_and_handles_rewind_and_arrival(replay):
    viewer = ReplayViewer(replay)
    for plan in reversed(replay["robot_plans"]):
        viewer.playhead = plan["tick"] + 0.25
        assert viewer.current_plan() == plan
    viewer.playhead = replay["metrics"]["robot_arrival_time"]
    assert viewer.current_plan() is None
    viewer.playhead = 0
    assert viewer.current_plan() == replay["robot_plans"][0]
    before = deepcopy(replay)
    pygame.font.init()
    surface = pygame.Surface((viewer.width, viewer.height))
    viewer.show_halos = viewer.show_path = False
    viewer.draw(surface)
    assert pygame.mask.from_threshold(surface, (*PLAN_COLOR, 255), (1, 1, 1, 255)).count() > 0
    assert replay == before
    viewer.action("plan")
    assert not viewer.show_plan


def test_advancing_along_route_does_not_report_route_changed():
    scenario = parse_scenario({"map": ["....."], "robot": {"start": [0, 0], "goal": [4, 0]}})
    viewer = ReplayViewer(make_replay(*paired_run(scenario, "L0")))
    assert all(not viewer.plan_changed(p) for p in viewer.robot_plans)
    revised = deepcopy(viewer.robot_plans[1])
    revised["path"] = [[1, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    viewer.robot_plans[1] = revised
    assert viewer.plan_changed(revised)


def test_rebuild_uses_snapshot_ignores_current_constants_and_verifies_motion(replay, monkeypatch):
    import human_nav.config as config

    monkeypatch.setattr(
        config, "load_constants", lambda *args: pytest.fail("Must not read live defaults")
    )
    legacy = deepcopy(replay)
    legacy.pop("robot_plans")
    # JSON conversion exercises restoration of tuples used by pathfinding.
    legacy = json.loads(json.dumps(legacy))
    assert (
        recorded_scenario(legacy["scenario"]).resolved()
        == recorded_scenario(replay["scenario"]).resolved()
    )
    builder = PlanRebuilder(legacy)
    while not builder.done:
        builder.step()
    assert builder.world.robot_plans == replay["robot_plans"]
    assert "robot_plans" not in legacy


def test_rebuild_refuses_motion_that_does_not_match(replay):
    legacy = deepcopy(replay)
    legacy["frames"][1]["positions"]["robot"] = [1, 1]
    builder = PlanRebuilder(legacy)
    with pytest.raises(ValueError, match="differs at tick 1"):
        builder.step()


def test_old_replay_build_is_on_demand_cached_and_does_not_edit_file(replay, tmp_path):
    legacy = deepcopy(replay)
    legacy.pop("robot_plans")
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    original = path.read_bytes()
    viewer = ReplayViewer.from_path(path)
    assert viewer.robot_plans is None and viewer.plan_rebuilder is None
    viewer.playhead = 1.25
    viewer.action("plan")
    while viewer.plan_rebuilder is not None:
        viewer.update_plan_rebuild()
    assert viewer.robot_plans == replay["robot_plans"]
    assert viewer.playhead == 1.25
    assert path.read_bytes() == original
    viewer.switch_replay(0)
    assert viewer.robot_plans == replay["robot_plans"]
    assert viewer.show_plan


def test_old_replay_rebuild_failure_keeps_original_recording(replay):
    legacy = deepcopy(replay)
    legacy.pop("robot_plans")
    legacy["frames"][1]["positions"]["robot"] = [1, 1]
    viewer = ReplayViewer(legacy)
    viewer.action("plan")
    while viewer.plan_rebuilder is not None:
        viewer.update_plan_rebuild()
    assert viewer.robot_plans is None
    assert "differs" in viewer.notice
    assert "robot_plans" not in legacy
