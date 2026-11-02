"""Matched no-robot evaluation, replay traces, and the 3 x 7 benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import BEHAVIORS, PLANNERS, Scenario, load_scenario
from .engine import Simulation
from .robot_metrics import robot_reference


def evaluate(simulation: Simulation, baseline: Simulation) -> dict[str, Any]:
    scenario = simulation.scenario
    if baseline.scenario != scenario or baseline.include_robot:
        raise ValueError("Evaluation requires the same resolved scenario without a robot")
    horizon = scenario.settings.max_steps
    people = []
    for spec in scenario.pedestrians:
        actual = simulation.agents[spec.id]
        reference = baseline.agents[spec.id]
        actual_time = actual.arrival_time if actual.arrival_time is not None else horizon
        baseline_time = reference.arrival_time if reference.arrival_time is not None else horizon
        signed = (
            actual_time - baseline_time
            if actual.arrival_time is not None and reference.arrival_time is not None
            else None
        )
        people.append(
            {
                "id": spec.id,
                "behavior": spec.behavior,
                "arrival_time": actual.arrival_time,
                "baseline_arrival_time": reference.arrival_time,
                "completed": actual.arrival_time is not None,
                "baseline_completed": reference.arrival_time is not None,
                "actual_spawn": actual.actual_spawn,
                "signed_arrival_difference": signed,
                "restricted_delay": max(0, actual_time - baseline_time),
                "restricted_improvement": max(0, baseline_time - actual_time),
                "walking_distance": actual.distance,
                "additional_distance": actual.distance - reference.distance,
                "waiting_steps": actual.waiting_steps,
                "additional_waiting_steps": actual.waiting_steps - reference.waiting_steps,
                "rejected_moves": actual.rejected_moves,
                "reroutes": actual.reroutes,
            }
        )
    delays = [p["restricted_delay"] for p in people]
    average = mean(delays) if delays else 0.0
    maximum = max(delays, default=0)
    robot = simulation.agents[scenario.robot.id]
    free_flow = robot_reference(scenario)
    robot_elapsed = (
        robot.arrival_time - robot.spec.spawn_time if robot.arrival_time is not None else None
    )
    weight = scenario.settings.fairness_weight
    stats = asdict(simulation.planner.stats) if simulation.planner else {}
    return {
        "scenario": scenario.name,
        "planner": simulation.planner_name,
        "behavior": people[0]["behavior"]
        if people and len({p["behavior"] for p in people}) == 1
        else "mixed"
        if people
        else "none",
        "seed": scenario.settings.seed,
        "evaluation_horizon": horizon,
        "simulated_steps": simulation.tick,
        "pedestrian_count": len(people),
        "pedestrians_completed": sum(p["completed"] for p in people),
        "baseline_completed": sum(p["baseline_completed"] for p in people),
        "completion_rate": mean([p["completed"] for p in people]) if people else 1.0,
        "robot_completed": robot.arrival_time is not None,
        "robot_arrival_time": robot.arrival_time,
        "robot_elapsed_time": robot_elapsed,
        "robot_free_flow_time": free_flow,
        "robot_disruption": max(0, robot_elapsed - free_flow)
        if robot_elapsed is not None
        else None,
        "robot_distance": robot.distance,
        "robot_waiting_steps": robot.waiting_steps,
        "robot_rejected_moves": robot.rejected_moves,
        "total_pedestrian_delay": sum(delays),
        "mean_pedestrian_delay": average,
        "max_pedestrian_delay": maximum,
        "total_pedestrian_improvement": sum(p["restricted_improvement"] for p in people),
        "fairness_delay_score": (1 - weight) * average + weight * maximum,
        "personal_space_exposure": simulation.space_exposure,
        "minimum_pedestrian_distance": simulation.minimum_distance,
        "planning": stats,
        "pedestrians": people,
        "notes": [
            "Delay is clipped per person, relative to the matched no-robot run.",
            "Unfinished trips use horizon-restricted time, not imputed final arrivals.",
            "Waiting counts active stationary timesteps, including movement-interval pauses.",
            "Timing is in simulation ticks; playback speed has no effect on metrics.",
            "Robot disruption is extra elapsed time over its wall-only trip without pedestrians; "
            "unfinished runs have no final disruption value.",
        ],
    }


def paired_run(scenario: Scenario, planner: str) -> tuple[Simulation, Simulation, dict]:
    baseline = Simulation(scenario, include_robot=False).run()
    simulation = Simulation(scenario, planner).run()
    return simulation, baseline, evaluate(simulation, baseline)


def make_replay(simulation: Simulation, baseline: Simulation, metrics: dict) -> dict:
    return {
        "format_version": 1,
        "scenario": simulation.scenario.resolved(),
        "metrics": metrics,
        "frames": simulation.history,
        "baseline_frames": baseline.history,
    }


def save_run(
    output: str | Path, simulation: Simulation, baseline: Simulation, metrics: dict
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scenario.json", simulation.scenario.resolved()),
        ("metrics.json", metrics),
        ("replay.json", make_replay(simulation, baseline, metrics)),
    ):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return output / "replay.json"


def batch(sources: list[Path], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, source in enumerate(sources):
        original = load_scenario(source)
        for behavior in BEHAVIORS:
            scenario = original.with_behavior(behavior)
            baseline = Simulation(scenario, include_robot=False).run()
            for planner in PLANNERS:
                simulation = Simulation(scenario, planner).run()
                metrics = evaluate(simulation, baseline)
                destination = output / f"{index:02d}-{source.stem}" / behavior / planner
                save_run(destination, simulation, baseline, metrics)
                flat = {
                    k: v
                    for k, v in metrics.items()
                    if k not in {"pedestrians", "planning", "notes"}
                }
                flat.update({f"planning_{k}": v for k, v in metrics["planning"].items()})
                flat["source"] = str(source)
                rows.append(flat)
                print(
                    f"{source.stem:18} {behavior:7} {planner:4} "
                    f"robot={'arrived' if metrics['robot_completed'] else 'unfinished':10} "
                    f"delay={metrics['total_pedestrian_delay']:4}",
                    flush=True,
                )
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "sources": [str(p.resolve()) for p in sources],
        "behaviors": list(BEHAVIORS),
        "planners": list(PLANNERS),
        "robot_runs": len(rows),
        "baseline_runs": 3 * len(sources),
        "determinism": "No randomness currently used; seed is reserved metadata.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows
