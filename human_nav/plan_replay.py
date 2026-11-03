"""Reconstruct legacy plan traces using recorded settings and verify their motion."""

from __future__ import annotations

from .config import AgentSpec, Grid, Scenario, Settings
from .engine import Simulation


def recorded_scenario(raw: dict) -> Scenario:
    """Restore the resolved snapshot without reading YAML or global defaults."""

    def agent(spec):
        return AgentSpec(
            **{
                **spec,
                "start": tuple(spec["start"]),
                "goals": tuple(tuple(p) for p in spec["goals"]),
            }
        )

    return Scenario(
        raw["name"],
        Grid(tuple(raw["grid"]["rows"])),
        agent(raw["robot"]),
        tuple(agent(p) for p in raw["pedestrians"]),
        Settings(**raw["settings"]),
        raw.get("source", ""),
    )


class PlanRebuilder:
    def __init__(self, replay: dict) -> None:
        self.frames = replay["frames"]
        self.world = Simulation(recorded_scenario(replay["scenario"]), replay["metrics"]["planner"])
        self._verify(0)

    def _verify(self, index: int) -> None:
        expected = self.frames[index]
        actual = self.world.snapshot()
        if any(actual[key] != expected[key] for key in ("tick", "positions", "arrivals")):
            raise ValueError(
                f"Recorded movement differs at tick {expected['tick']}. "
                "Run the scenario again to record its planned routes."
            )

    @property
    def done(self) -> bool:
        return self.world.tick >= len(self.frames) - 1

    def step(self) -> None:
        if not self.done:
            index = self.world.tick + 1
            self.world.step()
            self._verify(index)
