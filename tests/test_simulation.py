from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from human_nav.config import BEHAVIORS, PLANNERS, Grid, load_scenario, parse_scenario
from human_nav.engine import Simulation, resolve_moves
from human_nav.experiments import batch, evaluate, make_replay, paired_run
from human_nav.planners import RobotPlanner
from human_nav.search import astar, manhattan


@pytest.fixture
def data():
    return {
        "name": "test crossing",
        "map": ["#######", "#.....#", "#.....#", "#.....#", "#######"],
        "robot": {"start": [1, 2], "goal": [5, 2]},
        "pedestrians": [{"id": "person", "start": [3, 1], "goal": [3, 3]}],
        "simulation": {"max_steps": 20, "search_horizon": 15, "prediction_steps": 3},
    }


def test_astar_matches_independent_shortest_distances():
    rng = random.Random(92)
    for _ in range(40):
        rows = tuple(
            "".join("#" if rng.random() < 0.25 else "." for _ in range(9)) for _ in range(8)
        )
        grid = Grid(rows)
        free = [(x, y) for y in range(8) for x in range(9) if grid.walkable((x, y))]
        start, goal = rng.sample(free, 2)
        queue = deque([(start, 0)])
        seen = {start}
        expected = None
        while queue:
            p, distance = queue.popleft()
            if p == goal:
                expected = distance
                break
            for q in grid.neighbors(p):
                if q not in seen:
                    seen.add(q)
                    queue.append((q, distance + 1))
        path = astar(grid, start, goal)
        assert (len(path) - 1 if path else None) == expected


@pytest.mark.parametrize(
    "positions,proposals",
    [
        ({"a": (1, 1), "b": (2, 1)}, {"a": (2, 1), "b": (1, 1)}),
        ({"a": (1, 1), "b": (3, 1)}, {"a": (2, 1), "b": (2, 1)}),
        ({"a": (1, 1), "b": (2, 1), "c": (3, 1)}, {"a": (2, 1), "b": (3, 1), "c": (3, 1)}),
    ],
)
def test_conflict_resolution(positions, proposals):
    result = resolve_moves(positions, proposals, sorted(positions))
    assert len(set(result.values())) == len(result)
    for a in result:
        for b in result:
            if a != b:
                assert not (result[a] == positions[b] and result[b] == positions[a])


def test_random_conflicts_preserve_safety():
    rng = random.Random(10)
    grid = Grid(tuple("......" for _ in range(6)))
    for _ in range(1000):
        cells = rng.sample([(x, y) for x in range(6) for y in range(6)], 12)
        positions = {str(i): p for i, p in enumerate(cells)}
        proposals = {ident: rng.choice([p, *grid.neighbors(p)]) for ident, p in positions.items()}
        priority = list(positions)
        rng.shuffle(priority)
        result = resolve_moves(positions, proposals, priority)
        assert len(set(result.values())) == len(result)
        for a, destination in result.items():
            assert destination in {positions[a], proposals[a]}
            for b in result:
                if a != b:
                    assert not (destination == positions[b] and result[b] == positions[a])


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_all_pairings_are_safe_and_reproducible(data, planner, behavior):
    scenario = parse_scenario(data).with_behavior(behavior)
    first = Simulation(scenario, planner).run()
    second = Simulation(scenario, planner).run()
    assert first.history == second.history
    for frame in first.history:
        positions = list(map(tuple, frame["positions"].values()))
        assert len(positions) == len(set(positions))
    for before, after in zip(first.history, first.history[1:]):
        for ident in set(before["positions"]) & set(after["positions"]):
            assert manhattan(before["positions"][ident], after["positions"][ident]) <= 1
    assert first.agents[scenario.robot.id].arrival_time is not None


def test_forecast_does_not_change_real_world(data):
    world = Simulation(parse_scenario(data))
    before = deepcopy(world.agents)
    frames = world.forecast(3)
    assert world.agents == before
    assert world.tick == 0
    assert len(frames) == 4
    assert frames[1]["person"] == (3, 2)
    assert frames[2]["person"] == (3, 3)  # arrival transition is reserved
    assert "person" not in frames[3]


def test_arrival_at_prediction_boundary_does_not_freeze_forever(data):
    data["robot"].update(start=[2, 2], goal=[3, 2])
    data["pedestrians"][0].update(goal=[3, 2])
    data["simulation"].update(beta=0)
    world = Simulation(parse_scenario(data), "L3")
    forecast = world.forecast(1)
    assert forecast[1]["person"] == (3, 2)
    assert "person" not in forecast.at(2)
    assert world.planner.choose_move(world) == (2, 2)
    assert world.planner.last_path == [(2, 2), (2, 2), (3, 2)]


def test_forecast_tracks_waypoint_progression(data):
    data["pedestrians"][0].update(waypoints=[[3, 2]], goal=[5, 2])
    world = Simulation(parse_scenario(data))
    forecast = world.forecast(2)
    assert forecast.goals[0]["person"] == (3, 2)
    assert forecast.goals[1]["person"] == (5, 2)


def test_occupied_spawn_queues_without_overlapping(data):
    data["robot"].update(start=[2, 2], goal=[5, 2], move_interval=10)
    data["pedestrians"][0].update(start=[3, 2], goal=[1, 2], spawn_time=1)
    world = Simulation(parse_scenario(data), "L0")
    for _ in range(11):
        world.step()
    assert world.agents["person"].actual_spawn == 11
    assert all("person" not in frame["positions"] for frame in world.history[:11])
    assert world.history[11]["positions"]["person"] == [3, 2]


def test_slow_pedestrian_movement_and_waypoints(data):
    data["pedestrians"][0].update(start=[1, 1], goal=[3, 1], waypoints=[[2, 1]], move_interval=2)
    world = Simulation(parse_scenario(data), include_robot=False).run()
    assert world.history[1]["positions"]["person"] == [2, 1]
    assert world.history[2]["positions"]["person"] == [2, 1]
    assert world.agents["person"].arrival_time == 3
    assert world.agents["person"].distance == 2


def test_zero_disruption_when_routes_are_separate(data):
    data["pedestrians"][0].update(start=[1, 1], goal=[5, 1])
    simulation, baseline, metrics = paired_run(parse_scenario(data), "L0")
    assert metrics["total_pedestrian_delay"] == 0
    assert metrics["completion_rate"] == 1
    assert metrics["robot_elapsed_time"] == 4
    assert metrics["robot_free_flow_time"] == 4
    assert metrics["robot_disruption"] == 0
    assert baseline.planner is None
    assert make_replay(simulation, baseline, metrics)["format_version"] == 1


@pytest.mark.parametrize("interval,spawn,expected", [(1, 0, 4), (2, 5, 7), (3, 2, 10)])
def test_robot_free_flow_matches_unimpeded_simulation(data, interval, spawn, expected):
    data["pedestrians"] = []
    data["robot"].update(move_interval=interval, spawn_time=spawn, waypoints=[[3, 2]])
    _, _, metrics = paired_run(parse_scenario(data), "L0")
    assert metrics["robot_free_flow_time"] == expected
    assert metrics["robot_elapsed_time"] == expected
    assert metrics["robot_disruption"] == 0


def test_robot_delay_includes_queued_spawn_and_unfinished_is_unknown(data):
    data["robot"]["spawn_time"] = 5
    scenario = parse_scenario(data)
    world, baseline = Simulation(scenario), Simulation(scenario, include_robot=False)
    assert evaluate(world, baseline)["robot_disruption"] is None
    robot = world.agents[scenario.robot.id]
    robot.actual_spawn = 8
    robot.arrival_time = 15
    metrics = evaluate(world, baseline)
    assert metrics["robot_elapsed_time"] == 10
    assert metrics["robot_disruption"] == 6


def test_robot_already_at_destination_has_zero_reference(data):
    data["robot"]["goal"] = data["robot"]["start"]
    _, _, metrics = paired_run(parse_scenario(data), "L0")
    assert metrics["robot_free_flow_time"] == 0
    assert metrics["robot_elapsed_time"] == 0
    assert metrics["robot_disruption"] == 0


def test_unfinished_trips_are_included_and_improvements_do_not_cancel(data):
    data["pedestrians"].append({"id": "other", "start": [1, 1], "goal": [5, 1]})
    scenario = parse_scenario(data)
    actual = Simulation(scenario)
    baseline = Simulation(scenario, include_robot=False)
    baseline.agents["person"].arrival_time = 4
    actual.agents["person"].arrival_time = None
    baseline.agents["other"].arrival_time = 10
    actual.agents["other"].arrival_time = 2
    metrics = evaluate(actual, baseline)
    assert metrics["total_pedestrian_delay"] == 16
    assert metrics["max_pedestrian_delay"] == 16
    assert metrics["total_pedestrian_improvement"] == 8
    assert metrics["completion_rate"] == 0.5
    assert metrics["pedestrians"][0]["signed_arrival_difference"] is None


def test_baseline_priority_preserves_human_order(data):
    scenario = parse_scenario(data)
    actual, baseline = Simulation(scenario), Simulation(scenario, include_robot=False)
    for tick in range(8):
        actual.tick = baseline.tick = tick
        assert actual.priority() == baseline.priority()


def test_behavior_profiles_react_differently_to_obstruction(data):
    data["robot"].update(start=[3, 2], goal=[5, 2], move_interval=20)
    data["pedestrians"][0].update(start=[2, 2], goal=[4, 2], wait_threshold=3)
    proposals = {}
    for behavior in BEHAVIORS:
        world = Simulation(parse_scenario(data).with_behavior(behavior))
        human = world.agents["person"]
        proposals[behavior] = world._human_proposal(human, {(2, 2), (3, 2)})
    assert proposals["fixed"] == (3, 2)
    assert proposals["patient"] == (3, 2)
    assert proposals["reroute"] != (3, 2)


@pytest.mark.parametrize("behavior", ["reroute", "patient"])
def test_walking_line_follows_without_rerouting(behavior):
    scenario = parse_scenario(
        {
            "map": ["#########", "#.......#", "#.......#", "#.......#", "#########"],
            "robot": {"start": [7, 3], "goal": [1, 3]},
            "defaults": {"behavior": behavior, "wait_threshold": 1},
            "pedestrians": [
                {"id": "a_follower", "start": [1, 2], "goal": [7, 2]},
                {"id": "b_middle", "start": [2, 2], "goal": [7, 2]},
                {"id": "c_leader", "start": [3, 2], "goal": [7, 2]},
            ],
            "simulation": {"max_steps": 12},
        }
    )
    world = Simulation(scenario, include_robot=False).run()
    assert world.history[1]["positions"] == {
        "a_follower": [2, 2],
        "b_middle": [3, 2],
        "c_leader": [4, 2],
    }
    assert all(a.reroutes == 0 and a.waiting_steps == 0 for a in world.agents.values())
    assert world.agents["a_follower"].arrival_time == 6


def test_rerouting_leader_vacates_cell_before_follower_reacts():
    scenario = parse_scenario(
        {
            "map": ["#########", "#.......#", "#.......#", "#.......#", "#########"],
            "robot": {"start": [7, 3], "goal": [1, 3]},
            "defaults": {"behavior": "reroute"},
            "pedestrians": [
                {"id": "a_follower", "start": [1, 2], "goal": [7, 2]},
                {"id": "b_middle", "start": [2, 2], "goal": [7, 2]},
                {"id": "c_leader", "start": [3, 2], "goal": [7, 2], "move_interval": 20},
            ],
            "simulation": {"max_steps": 12},
        }
    )
    world = Simulation(scenario, include_robot=False)
    world.step()  # All three can advance.
    world.step()  # Leader stops; middle detours and follower can still advance.
    assert world.agents["b_middle"].reroutes == 1
    assert world.agents["a_follower"].reroutes == 0
    assert world.agents["a_follower"].position == (3, 2)


def test_patient_threshold_counts_genuine_blocked_attempts():
    scenario = parse_scenario(
        {
            "map": ["#########", "#.......#", "#.......#", "#.......#", "#########"],
            "robot": {"start": [7, 3], "goal": [1, 3]},
            "defaults": {"behavior": "patient", "wait_threshold": 2},
            "pedestrians": [
                {"id": "follower", "start": [2, 2], "goal": [7, 2]},
                {"id": "leader", "start": [3, 2], "goal": [7, 2], "move_interval": 20},
            ],
            "simulation": {"max_steps": 12},
        }
    )
    world = Simulation(scenario, include_robot=False)
    world.step()
    follower = world.agents["follower"]
    assert follower.reroutes == follower.blocked_steps == 0
    world.step()
    assert follower.reroutes == 0 and follower.blocked_steps == 1
    world.step()
    assert follower.reroutes == 1 and follower.blocked_steps == 0


def test_prediction_and_disruption_change_robot_decision(data):
    # Human heads east down a narrow lane. At t=1 the robot can occupy the
    # person's only best exit without colliding with the predicted position.
    data["map"] = ["#########", "#.......#", "####.####", "####.####", "#########"]
    data["robot"].update(start=[4, 2], goal=[4, 1])
    data["pedestrians"][0].update(start=[2, 1], goal=[7, 1])
    data["simulation"].update(beta=0, gamma=20)
    scenario = parse_scenario(data)
    ordinary = Simulation(scenario, "L3")
    disruption = Simulation(scenario, "L3+D")
    assert ordinary.planner.choose_move(ordinary) == (4, 1)
    assert disruption.planner.choose_move(disruption) != (4, 1)


def test_computation_budget_is_enforced(data):
    scenario = parse_scenario(data)
    scenario = replace(scenario, settings=replace(scenario.settings, max_expansions=1))
    world = Simulation(scenario, "L4")
    planner = RobotPlanner(scenario, "L4")
    assert planner.choose_move(world) == scenario.robot.start
    assert planner.stats.budget_exits == 1
    assert planner.stats.expanded == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(map=["###", "##"]),
        lambda d: d["robot"].update(start=[0, 0]),
        lambda d: d["pedestrians"][0].update(move_interval=0),
        lambda d: d["pedestrians"][0].update(behavior="unknown"),
        lambda d: d["simulation"].update(beta=float("nan")),
        lambda d: d["simulation"].update(max_step=20),
        lambda d: d["pedestrians"][0].update(id="robot"),
        lambda d: d["pedestrians"][0].update(direction_order="NNES"),
    ],
)
def test_invalid_scenarios_fail_clearly(data, change):
    change(data)
    with pytest.raises(ValueError):
        parse_scenario(data)


def test_example_scenarios_are_valid():
    files = list((Path(__file__).resolve().parent.parent / "scenarios").glob("*.yaml"))
    assert len(files) == 5
    for file in files:
        load_scenario(file)


def test_all_maps_inherit_shared_threshold_independent_of_working_directory(tmp_path, monkeypatch):
    import human_nav.config as config

    root = Path(__file__).resolve().parent.parent
    constants = tmp_path / "constants.yaml"
    constants.write_text("defaults:\n  wait_threshold: 7\n", encoding="utf-8")
    monkeypatch.setattr(config, "GLOBAL_CONSTANTS_PATH", constants)
    monkeypatch.chdir(tmp_path)
    for path in (root / "scenarios").glob("*.yaml"):
        scenario = load_scenario(path)
        assert all(p.wait_threshold == 7 for p in scenario.pedestrians)
        assert all(p["wait_threshold"] == 7 for p in scenario.resolved()["pedestrians"])


def test_shared_defaults_scenario_and_agent_precedence(data):
    constants = {
        "defaults": {"wait_threshold": 7, "personal_radius": 4},
        "simulation": {"beta": 8, "max_steps": 50},
    }
    data["defaults"] = {"wait_threshold": 5}
    data["pedestrians"][0]["wait_threshold"] = 2
    data["pedestrians"].append({"id": "other", "start": [1, 1], "goal": [5, 1]})
    scenario = parse_scenario(data, constants=constants)
    assert [p.wait_threshold for p in scenario.pedestrians] == [2, 5]
    assert all(p.personal_radius == 4 for p in scenario.pedestrians)
    assert scenario.settings.beta == 8
    assert scenario.settings.max_steps == 20  # Scenario overrides the shared 50.


def test_shared_file_edits_apply_to_new_loads_and_preserve_existing_scenarios(
    data, tmp_path, monkeypatch
):
    import human_nav.config as config

    path = tmp_path / "constants.yaml"
    monkeypatch.setattr(config, "GLOBAL_CONSTANTS_PATH", path)
    path.write_text("defaults:\n  wait_threshold: 1\n", encoding="utf-8")
    first = parse_scenario(data)
    path.write_text("defaults:\n  wait_threshold: 4\n", encoding="utf-8")
    second = parse_scenario(data)
    assert first.pedestrians[0].wait_threshold == 1
    assert second.pedestrians[0].wait_threshold == 4


@pytest.mark.parametrize(
    "constants",
    [
        {"unknown": {}},
        {"defaults": {"wait_thresold": 1}},
        {"defaults": {"wait_threshold": 0}},
        {"simulation": {"beta": -1}},
        {"defaults": []},
    ],
)
def test_invalid_shared_constants_are_rejected(data, constants):
    with pytest.raises(ValueError):
        parse_scenario(data, constants=constants)


def test_batch_produces_21_pairings_and_replay_artifacts(data, tmp_path):
    import yaml

    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    rows = batch([path], tmp_path / "results")
    assert len(rows) == 21
    assert all("robot_disruption" in row and "robot_free_flow_time" in row for row in rows)
    assert len(list((tmp_path / "results").rglob("replay.json"))) == 21
    assert (tmp_path / "results" / "summary.csv").exists()


def test_gui_renders_and_interpolates(data, tmp_path):
    from human_nav.gui import ReplayViewer

    simulation, baseline, metrics = paired_run(parse_scenario(data), "L0")
    viewer = ReplayViewer(make_replay(simulation, baseline, metrics))
    positions = viewer.interpolated(viewer.frames, 0.5)
    assert positions["robot"] == (1.5, 2.0)
    path = tmp_path / "frame.png"
    viewer.screenshot(path, tick=0.5)
    assert path.stat().st_size > 1000


def test_viewer_controls_and_event_loop(data, monkeypatch):
    import pygame

    from human_nav.gui import ReplayViewer

    simulation, baseline, metrics = paired_run(parse_scenario(data), "L0")
    viewer = ReplayViewer(make_replay(simulation, baseline, metrics))
    viewer.action("next")
    assert viewer.playhead == 1 and not viewer.playing
    viewer.action("previous")
    assert viewer.playhead == 0
    viewer.action("faster")
    assert viewer.speed == 2
    viewer.action("slower")
    assert viewer.speed == 1
    viewer.action("baseline")
    assert viewer.show_baseline
    viewer.action("halos")
    assert not viewer.show_halos
    viewer.action("path")
    assert not viewer.show_path
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    viewer.run()
