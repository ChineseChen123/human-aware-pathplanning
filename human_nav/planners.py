"""Seven robot configurations sharing a time-expanded A* implementation."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import hypot
from time import perf_counter
from typing import TYPE_CHECKING

from .config import PLANNERS, Point, Scenario
from .engine import Forecast
from .search import distance_field

if TYPE_CHECKING:
    from .engine import Simulation

State = tuple[Point, int]


@dataclass
class PlannerStats:
    calls: int = 0
    expanded: int = 0
    seconds: float = 0.0
    budget_exits: int = 0
    horizon_exits: int = 0
    no_path: int = 0


class RobotPlanner:
    def __init__(self, scenario: Scenario, name: str) -> None:
        if name not in PLANNERS:
            raise ValueError(f"Unknown robot planner {name!r}; choose from {PLANNERS}")
        self.scenario = scenario
        self.name = name
        self.level = int(name[1])
        self.disruption = name.endswith("+D")
        self.stats = PlannerStats()
        self.distances = {
            goal: distance_field(scenario.grid, goal) for goal in scenario.robot.goals
        }
        self.last_path: list[Point] = []
        self.last_forecast: list[dict[str, Point]] = []
        self.last_status = "not planned"
        self.last_cost = 0.0
        self.human_distances: dict[Point, dict[Point, int]] = {}

    def choose_move(self, world: Simulation) -> Point:
        started = perf_counter()
        self.stats.calls += 1
        agent = world.agents[self.scenario.robot.id]
        assert agent.position is not None
        start = agent.position
        lookahead = 1 if self.level == 3 else self.scenario.settings.prediction_steps
        if self.level >= 3:
            forecast = world.forecast(lookahead)
        elif self.level >= 1:
            forecast = Forecast(
                [
                    {
                        p.id: world.agents[p.id].position
                        for p in self.scenario.pedestrians
                        if world.agents[p.id].position is not None
                    }
                ]
            )
        else:
            forecast = Forecast([{}])
        self.last_forecast = forecast
        try:
            path, cost, status = self._search(world, start, agent.goal, forecast)
            self.last_path = path
            self.last_status = status
            self.last_cost = cost
            if status == "budget":
                self.stats.budget_exits += 1
            elif status == "horizon":
                self.stats.horizon_exits += 1
            elif status == "no path":
                self.stats.no_path += 1
            return path[1] if len(path) > 1 else start
        finally:
            self.stats.seconds += perf_counter() - started

    def _search(
        self,
        world: Simulation,
        start: Point,
        goal: Point,
        forecast: Forecast,
    ) -> tuple[list[Point], float, str]:
        settings = self.scenario.settings
        distances = self.distances[goal]
        horizon = min(settings.search_horizon, settings.max_steps - world.tick)
        serial = count()
        initial: State = (start, 0)
        costs = {initial: 0.0}
        parents: dict[State, State] = {}
        h = settings.alpha * distances[start]
        frontier = [(h, h, next(serial), 0.0, initial)]
        expanded = 0
        best_boundary: State | None = None
        best_boundary_key = (float("inf"), float("inf"))
        robot = world.agents[self.scenario.robot.id]
        # Once forecasts freeze, a disconnected state can never reach the goal.
        # Prune it instead of expanding the same impossible waits to the horizon.
        # This is a feasibility test; the A* heuristic remains wall-only.
        freeze_time = len(forecast) if forecast.arrival_offsets else len(forecast) - 1
        terminal_reachable = distance_field(
            self.scenario.grid, goal, set(forecast.at(freeze_time).values())
        )

        def humans(offset: int) -> dict[str, Point]:
            return forecast.at(offset)

        def reconstruct(state: State) -> list[Point]:
            route = [state[0]]
            while state != initial:
                state = parents[state]
                route.append(state[0])
            return route[::-1]

        # Cost tables are frozen for this search. Forecast evaluation does not
        # depend on a candidate robot prefix, so (position, time) is sufficient.
        step_penalties: dict[tuple[Point, int], float] = {}

        def penalty(q: Point, offset: int) -> float:
            key = (q, offset)
            if key in step_penalties:
                return step_penalties[key]
            result = 0.0
            if self.level >= 2:
                for spec in self.scenario.pedestrians:
                    p = humans(offset).get(spec.id)
                    if p is not None:
                        result += (
                            settings.beta
                            * max(0.0, 1.0 - hypot(q[0] - p[0], q[1] - p[1]) / spec.personal_radius)
                            ** 2
                        )
            if self.disruption:
                # Local lost-progress pressure remains informative even with a
                # one-step forecast. Ask whether occupying q removes a human's
                # only best exit from their predicted cell, using map distances.
                # This is a surrogate, not a simulation of actual imposed delay.
                for spec in self.scenario.pedestrians:
                    human = forecast[offset].get(spec.id) if offset < len(forecast) else None
                    if human is None:
                        continue
                    if (
                        spec.id in forecast.arrival_offsets
                        and offset >= forecast.arrival_offsets[spec.id]
                    ):
                        continue
                    goal = forecast.goals[offset].get(spec.id, world.agents[spec.id].goal)
                    if goal not in self.human_distances:
                        self.human_distances[goal] = distance_field(self.scenario.grid, goal)
                    field = self.human_distances[goal]
                    options = [human, *self.scenario.grid.neighbors(human)]
                    if q in options and q != human:
                        best = min(field[p] for p in options)
                        restricted = min(field[p] for p in options if p != q)
                        result += settings.gamma * max(0, restricted - best)
                # Use actual available future forecasts only. Extending a frozen
                # endpoint would count invented future footfall as evidence.
                for future in range(offset + 1, len(forecast)):
                    weight = 1.0 / (future - offset)
                    result += (
                        settings.gamma * weight * sum(p == q for p in forecast[future].values())
                    )
            step_penalties[key] = result
            return result

        while frontier:
            _, _, _, g, state = heappop(frontier)
            if g != costs[state]:
                continue
            p, offset = state
            if offset >= freeze_time and p not in terminal_reachable:
                continue
            if p == goal:
                self.stats.expanded += expanded
                return reconstruct(state), g, "goal"
            if expanded >= settings.max_expansions:
                self.stats.expanded += expanded
                return [start], 0.0, "budget"
            expanded += 1
            if offset == horizon:
                key = (g + settings.alpha * distances[p], distances[p])
                if key < best_boundary_key:
                    best_boundary, best_boundary_key = state, key
                continue
            destinations = self.scenario.grid.neighbors(p, robot.spec.direction_order)
            if not robot.eligible(world.tick + offset):
                destinations = []
            for q in (*destinations, p):
                next_offset = offset + 1
                predicted_now = humans(offset)
                predicted_next = humans(next_offset)
                if q in predicted_next.values():
                    continue
                if any(
                    predicted_now.get(ident) == q and next_p == p
                    for ident, next_p in predicted_next.items()
                ):
                    continue
                new_g = g + settings.alpha + penalty(q, next_offset)
                next_state = (q, next_offset)
                if new_g >= costs.get(next_state, float("inf")):
                    continue
                costs[next_state] = new_g
                parents[next_state] = state
                next_h = settings.alpha * distances[q]
                heappush(frontier, (new_g + next_h, next_h, next(serial), new_g, next_state))
        self.stats.expanded += expanded
        if best_boundary is not None:
            return reconstruct(best_boundary), costs[best_boundary], "horizon"
        return [start], 0.0, "no path"
