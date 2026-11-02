"""Command-line entry points for running, replaying, and benchmarking scenarios."""

from __future__ import annotations

import argparse
import glob
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import BEHAVIORS, PLANNERS, load_scenario
from .experiments import batch, make_replay, paired_run, save_run

DEFAULT_SCENARIO = Path(__file__).resolve().parent.parent / "scenarios" / "intersection.yaml"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return number


def _sources(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No scenario files match: {pattern}")
        paths.extend(Path(p).resolve() for p in matches)
    return list(dict.fromkeys(paths))


def _output(prefix: str) -> Path:
    return Path("results") / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S-%f}"


def _print_metrics(metrics: dict, replay_path: Path) -> None:
    print(f"{metrics['scenario']} | {metrics['planner']} | {metrics['behavior']}")
    print(
        f"Robot arrival: {metrics['robot_arrival_time']} | "
        f"pedestrians completed: {metrics['pedestrians_completed']}/{metrics['pedestrian_count']}"
    )
    print(
        f"Pedestrian delay: total={metrics['total_pedestrian_delay']}, "
        f"max={metrics['max_pedestrian_delay']} ticks | "
        f"space exposure={metrics['personal_space_exposure']:.3f}"
    )
    if (
        metrics["completion_rate"] < 1
        or metrics["baseline_completed"] < metrics["pedestrian_count"]
    ):
        print("Unfinished pedestrian trips: delays are restricted to the evaluation horizon.")
    print(f"Replay saved: {replay_path.resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human-aware robot navigation experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one paired experiment; optionally open playback")
    run.add_argument("scenario", nargs="?", type=Path, default=DEFAULT_SCENARIO)
    run.add_argument("--planner", choices=PLANNERS, default="L4+D")
    run.add_argument("--behavior", choices=BEHAVIORS, help="Override all pedestrian behaviors")
    run.add_argument("--gui", action="store_true", help="Open the smooth replay viewer")
    run.add_argument("--output", type=Path)
    for option in ("max-steps", "prediction-steps", "search-horizon", "max-expansions"):
        run.add_argument(f"--{option}", type=_positive_int)
    for option in ("alpha", "beta", "gamma"):
        run.add_argument(f"--{option}", type=_nonnegative_float)

    replay = subparsers.add_parser(
        "replay", help="Open one replay or a whole batch without rerunning planning"
    )
    replay.add_argument("path", type=Path, help="Replay JSON, run folder, or batch results folder")
    replay.add_argument("--paused", action="store_true", help="Open without starting playback")
    replay.add_argument("--tick", type=_nonnegative_float, default=0.0)
    snapshot = subparsers.add_parser(
        "snapshot", help="Render a replay frame to a PNG without a window"
    )
    snapshot.add_argument("path", type=Path)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--tick", type=_nonnegative_float, default=0.0)

    benchmark = subparsers.add_parser("batch", help="Run 3 human profiles x 7 robot configurations")
    benchmark.add_argument("scenarios", nargs="+", help="YAML files or wildcard patterns")
    benchmark.add_argument("--output", type=Path)
    plot = subparsers.add_parser(
        "plot", help="Graph batch results separately by pedestrian behavior"
    )
    plot.add_argument("source", type=Path, help="Batch output directory or summary.csv")
    plot.add_argument(
        "--output", type=Path, help="Graph directory (default: beside CSV in graphs/)"
    )
    from .plotting import DEFAULT_METRIC, METRICS

    plot.add_argument("--metric", choices=tuple(METRICS), default=DEFAULT_METRIC)
    plot.add_argument("--gui", action="store_true", help="Also open the interactive results cube")
    validate = subparsers.add_parser("validate", help="Validate scenario files without simulation")
    validate.add_argument("scenarios", nargs="+")

    actual_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(actual_argv or ["run", "--gui"])
    try:
        if args.command == "run":
            scenario = load_scenario(args.scenario).with_behavior(args.behavior)
            fields = (
                "max_steps",
                "prediction_steps",
                "search_horizon",
                "max_expansions",
                "alpha",
                "beta",
                "gamma",
            )
            overrides = {
                field: getattr(args, field) for field in fields if getattr(args, field) is not None
            }
            settings = replace(scenario.settings, **overrides)
            if settings.alpha <= 0:
                raise ValueError("alpha must be positive")
            if (
                max(p.spawn_time for p in (*scenario.pedestrians, scenario.robot))
                >= settings.max_steps
            ):
                raise ValueError("max_steps must be after every agent's scheduled spawn")
            scenario = replace(scenario, settings=settings)
            print("Computing the matched baseline and robot run...", flush=True)
            simulation, baseline, metrics = paired_run(scenario, args.planner)
            replay_path = save_run(
                args.output or _output(args.scenario.stem), simulation, baseline, metrics
            )
            _print_metrics(metrics, replay_path)
            if args.gui:
                from .gui import ReplayViewer

                ReplayViewer(make_replay(simulation, baseline, metrics)).run()
        elif args.command in {"replay", "snapshot"}:
            from .gui import ReplayViewer

            viewer = ReplayViewer.from_path(args.path)
            if args.command == "snapshot":
                viewer.screenshot(args.output, args.tick)
                print(f"Saved {args.output.resolve()}")
            else:
                viewer.playhead = min(args.tick, viewer.last_tick)
                viewer.playing = not args.paused
                viewer.run()
        elif args.command == "batch":
            output = args.output or _output("batch")
            rows = batch(_sources(args.scenarios), output)
            print(f"Saved {len(rows)} robot runs and matched baselines to {output.resolve()}")
            from .plotting import plot_results

            paths = plot_results(output)
            print(
                f"Saved {len(paths) // 2} behavior graphs (PNG and SVG) to "
                f"{(output / 'graphs').resolve()}"
            )
        elif args.command == "plot":
            from .plotting import plot_results

            for path in plot_results(args.source, args.output, args.metric):
                print(f"Saved {path.resolve()}")
            if args.gui:
                from .gui import ReplayViewer
                from .results_gui import ResultsExplorer

                source = args.source if args.source.is_dir() else args.source.parent
                viewer = ReplayViewer.from_path(source)
                viewer.action("results")
                if viewer.explorer is None:
                    raise ValueError(viewer.notice)
                if args.metric != DEFAULT_METRIC:
                    from .plotting import read_results

                    viewer.explorer = ResultsExplorer(
                        read_results(args.source, args.metric),
                        lambda metric: read_results(args.source, metric),
                    )
                viewer.run()
        else:
            for source in _sources(args.scenarios):
                scenario = load_scenario(source)
                print(
                    f"OK {source.name}: {scenario.grid.width}x{scenario.grid.height}, "
                    f"{len(scenario.pedestrians)} pedestrians"
                )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
