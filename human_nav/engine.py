"""Deterministic simultaneous grid movement, independent of rendering."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import AgentSpec, Point, Scenario
from .search import astar, manhattan

if TYPE_CHECKING:
    from .planners import RobotPlanner


class Forecast(list[dict[str, Point]]):
    """Transition occupancy plus known exits and waypoint progression."""

    def __init__(self, frames=()) -> None:
        super().__init__(frames)
        self.arrival_offsets: dict[str, int] = {}
        self.goals: list[dict[str, Point]] = []

    def at(self, offset: int) -> dict[str, Point]:
        if offset < len(self):
            return self[offset]
        return {ident: p for ident, p in self[-1].items() if ident not in self.arrival_offsets}


@dataclass
class AgentState:
    spec: AgentSpec
    position: Point | None = None
    goal_index: int = 0
    route: list[Point] = field(default_factory=list)
    arrival_time: int | None = None
    actual_spawn: int | None = None
    blocked_steps: int = 0
    waiting_steps: int = 0
    distance: int = 0
    rejected_moves: int = 0
    reroutes: int = 0

    @property
    def goal(self) -> Point:
        return self.spec.goals[min(self.goal_index, len(self.spec.goals) - 1)]

    def eligible(self, tick: int) -> bool:
        return (tick - self.spec.spawn_time) % self.spec.move_interval == 0


def resolve_moves(
    positions: dict[str, Point],
    proposals: dict[str, Point],
    priority: list[str],
) -> dict[str, Point]:
    """Resolve vertices, swaps, and movement into agents forced to stay.

    Moving chains and cycles of length >= 3 are permitted. Stationary occupants
    cannot be displaced, regardless of priority. Conflicts only remove moves,
    so the fixed-point iteration always terminates.
    """
    rank = {ident: index for index, ident in enumerate(priority)}
    result = {ident: proposals.get(ident, p) for ident, p in positions.items()}
    destinations: dict[Point, list[str]] = {}
    for ident, destination in result.items():
        destinations.setdefault(destination, []).append(ident)
    for destination, contenders in destinations.items():
        winner = min(contenders, key=lambda ident: (positions[ident] != destination, rank[ident]))
        for ident in contenders:
            if ident != winner:
                result[ident] = positions[ident]
    occupants = {p: ident for ident, p in positions.items()}
    while True:
        cancel = set()
        for ident, destination in result.items():
            if destination == positions[ident]:
                continue
            other = occupants.get(destination)
            if other is None:
                continue
            if result[other] == positions[other] or result[other] == positions[ident]:
                cancel.add(ident)
                if result[other] == positions[ident]:
                    cancel.add(other)
        if not cancel:
            return result
        for ident in cancel:
            result[ident] = positions[ident]


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        planner: str = "L4+D",
        include_robot: bool = True,
        record: bool = True,
    ) -> None:
        self.scenario = scenario
        self.include_robot = include_robot
        self.planner_name = planner if include_robot else "baseline"
        self.record = record
        specs = (*scenario.pedestrians, *((scenario.robot,) if include_robot else ()))
        self.agents = {spec.id: AgentState(spec) for spec in specs}
        # Include the absent robot's slot in baseline priority rotations. Removing
        # it must not independently change the human-human arbitration policy.
        self.priority_universe = sorted([scenario.robot.id, *(p.id for p in scenario.pedestrians)])
        self.tick = 0
        self.space_exposure = 0.0
        self.minimum_distance: float | None = None
        self.planner: RobotPlanner | None = None
        if include_robot:
            from .planners import RobotPlanner

            self.planner = RobotPlanner(scenario, planner)
        self._spawn()
        self.history = [self.snapshot()] if record else []

    def priority(self) -> list[str]:
        offset = self.tick % len(self.priority_universe)
        return self.priority_universe[offset:] + self.priority_universe[:offset]

    def _advance_goals(self, agent: AgentState, at: int) -> None:
        while agent.position == agent.goal and agent.goal_index < len(agent.spec.goals):
            agent.goal_index += 1
            agent.route.clear()
            if agent.goal_index == len(agent.spec.goals):
                agent.arrival_time = at
                agent.position = None
                return

    def _spawn(self) -> None:
        occupied = {a.position for a in self.agents.values() if a.position is not None}
        for ident in self.priority():
            agent = self.agents.get(ident)
            if agent is None or agent.actual_spawn is not None:
                continue
            if agent.spec.spawn_time > self.tick or agent.spec.start in occupied:
                continue
            agent.position = agent.spec.start
            agent.actual_spawn = self.tick
            self._advance_goals(agent, self.tick)
            if agent.position is not None:
                occupied.add(agent.position)

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "positions": {
                ident: list(a.position)
                for ident, a in self.agents.items()
                if a.position is not None
            },
            "arrivals": {
                ident: a.arrival_time
                for ident, a in self.agents.items()
                if a.arrival_time is not None
            },
        }

    @property
    def finished(self) -> bool:
        return all(a.arrival_time is not None for a in self.agents.values())

    def _human_proposal(
        self,
        agent: AgentState,
        occupied: set[Point],
        *,
        allow_reroute: bool = True,
        obstructed: bool | None = None,
    ) -> Point:
        assert agent.position is not None
        if not agent.eligible(self.tick):
            return agent.position
        if not agent.route:
            route = astar(
                self.scenario.grid, agent.position, agent.goal, order=agent.spec.direction_order
            )
            agent.route = route[1:] if route else []
        if not agent.route:
            return agent.position
        if obstructed is None:
            obstructed = agent.route[0] in occupied - {agent.position}
        should_reroute = (
            allow_reroute
            and obstructed
            and (
                agent.spec.behavior == "reroute"
                or (
                    agent.spec.behavior == "patient"
                    and agent.blocked_steps + 1 >= agent.spec.wait_threshold
                )
            )
        )
        if should_reroute:
            agent.reroutes += 1
            route = astar(
                self.scenario.grid,
                agent.position,
                agent.goal,
                blocked=occupied,
                order=agent.spec.direction_order,
            )
            if route is None:
                return agent.position
            agent.route = route[1:]
        return agent.route[0] if agent.route else agent.position

    def step(self) -> dict:
        if self.finished or self.tick >= self.scenario.settings.max_steps:
            return self.snapshot()
        self._spawn()
        positions = {
            ident: a.position for ident, a in self.agents.items() if a.position is not None
        }
        occupied = set(positions.values())
        proposals = {}
        # Forecast before updating any human route memories for this timestep.
        robot_id = self.scenario.robot.id
        if robot_id in positions and self.planner is not None:
            robot = self.agents[robot_id]
            proposals[robot_id] = (
                self.planner.choose_move(self) if robot.eligible(self.tick) else positions[robot_id]
            )
        for ident, p in positions.items():
            agent = self.agents[ident]
            if ident == self.scenario.robot.id and self.planner is not None:
                continue
            else:
                proposals[ident] = self._human_proposal(agent, occupied, allow_reroute=False)
        self._reroute_blocked(positions, proposals)
        for ident, p in proposals.items():
            if not self.scenario.grid.walkable(p) or manhattan(positions[ident], p) > 1:
                raise RuntimeError(f"Illegal proposal by {ident}: {p}")
        executed = resolve_moves(positions, proposals, self.priority())

        # Left-endpoint sampling: one exposure sample per simulated interval,
        # using actual discrete positions, including an agent's final interval.
        robot = positions.get(self.scenario.robot.id)
        if robot is not None:
            for spec in self.scenario.pedestrians:
                human = positions.get(spec.id)
                if human is None:
                    continue
                distance = ((robot[0] - human[0]) ** 2 + (robot[1] - human[1]) ** 2) ** 0.5
                self.minimum_distance = (
                    distance
                    if self.minimum_distance is None
                    else min(distance, self.minimum_distance)
                )
                self.space_exposure += max(0.0, 1.0 - distance / spec.personal_radius) ** 2

        for ident, destination in executed.items():
            agent = self.agents[ident]
            old = positions[ident]
            moved = destination != old
            if moved:
                agent.distance += 1
                agent.blocked_steps = 0
                if agent.route and agent.route[0] == destination:
                    agent.route.pop(0)
            else:
                agent.waiting_steps += 1
                if agent.eligible(self.tick):
                    agent.blocked_steps += 1
            if proposals[ident] != old and not moved:
                agent.rejected_moves += 1
            agent.position = destination
            self._advance_goals(agent, self.tick + 1)
        self.tick += 1
        self._spawn()
        snapshot = self.snapshot()
        if self.record:
            self.history.append(snapshot)
        return snapshot

    def _reroute_blocked(
        self,
        positions: dict[str, Point],
        proposals: dict[str, Point],
    ) -> None:
        """First allow simultaneous following; reroute only rejected moves.

        Consider a blocking pedestrian before the person following them. If the
        leader reroutes and vacates its cell, the follower can keep its route.
        Every attempted repair is checked by the same final collision resolver.
        """
        occupants = {p: ident for ident, p in positions.items()}
        visited = set()
        order = []

        def visit(ident: str) -> None:
            if ident in visited or ident not in positions:
                return
            visited.add(ident)
            blocker = occupants.get(proposals[ident])
            if blocker is not None and blocker != ident:
                visit(blocker)
            order.append(ident)

        for ident in self.priority():
            visit(ident)
        for ident in order:
            agent = self.agents[ident]
            if (
                ident == self.scenario.robot.id
                or not agent.eligible(self.tick)
                or agent.spec.behavior == "fixed"
                or proposals[ident] == positions[ident]
            ):
                continue
            resolved = resolve_moves(positions, proposals, self.priority())
            if resolved[ident] != positions[ident]:
                continue  # Occupied now, but the cell is being vacated safely.
            blocked = {
                positions[other] for other in positions if resolved[other] == positions[other]
            }
            # Also avoid an empty destination lost to another agent's priority.
            blocked.add(proposals[ident])
            proposals[ident] = self._human_proposal(agent, blocked, obstructed=True)

    def run(self) -> Simulation:
        while not self.finished and self.tick < self.scenario.settings.max_steps:
            self.step()
        return self

    def forecast(self, steps: int) -> Forecast:
        """Human-only counterfactual from the current observed state.

        Retains pedestrians' current route memories and blockage counters. Robot
        removal changes future interactions, not the history humans already had.
        Forecasting never mutates the real simulation.
        """
        world = Simulation(self.scenario, include_robot=False, record=False)
        world.agents = {p.id: deepcopy(self.agents[p.id]) for p in self.scenario.pedestrians}
        world.tick = self.tick
        frames = Forecast()
        for offset in range(steps + 1):
            frame = {
                ident: a.position for ident, a in world.agents.items() if a.position is not None
            }
            # An arriving human still occupies their destination during the
            # arrival transition, even though they leave immediately afterward.
            if offset > 0:
                arrivals = {
                    ident: a.spec.goals[-1]
                    for ident, a in world.agents.items()
                    if a.arrival_time == self.tick + offset
                }
                frame.update(arrivals)
                frames.arrival_offsets.update({ident: offset for ident in arrivals})
            frames.append(frame)
            frames.goals.append({ident: a.goal for ident, a in world.agents.items()})
            if offset < steps:
                world.step()
        return frames
