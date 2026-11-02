"""Unimpeded robot travel time and delay relative to that reference."""

from __future__ import annotations

from .config import Grid, Scenario
from .search import astar


def free_flow_time(grid: Grid, start: tuple[int, int], goals, move_interval: int) -> int:
    distance = 0
    for goal in goals:
        goal = tuple(goal)
        route = astar(grid, start, goal)
        if route is None:
            raise ValueError(f"No wall-only robot route from {start} to {goal}")
        distance += len(route) - 1
        start = goal
    # The first move is eligible at the requested spawn tick, arriving one tick
    # later. Subsequent moves are separated by move_interval ticks.
    return 0 if distance == 0 else 1 + (distance - 1) * move_interval


def robot_reference(scenario: Scenario) -> int:
    robot = scenario.robot
    return free_flow_time(scenario.grid, robot.start, robot.goals, robot.move_interval)


def saved_robot_reference(scenario: dict) -> tuple[int, int]:
    """Use the recorded input, never a potentially edited source YAML file."""
    robot = scenario["robot"]
    duration = free_flow_time(
        Grid(tuple(scenario["grid"]["rows"])),
        tuple(robot["start"]),
        robot["goals"],
        robot["move_interval"],
    )
    return duration, robot["spawn_time"]
