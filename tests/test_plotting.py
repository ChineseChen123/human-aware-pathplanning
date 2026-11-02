from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest

from human_nav.cli import main
from human_nav.config import BEHAVIORS, PLANNERS, parse_scenario
from human_nav.plotting import (
    DEFAULT_METRIC,
    PLANNER_COLORS,
    build_figure,
    map_markers,
    plot_results,
    read_results,
)


@pytest.fixture
def csv_rows():
    return [
        {
            "scenario": name,
            "source": f"scenarios/{name}.yaml",
            "planner": planner,
            "behavior": behavior,
            "robot_completed": not (planner == "L0" and map_index == 0),
            "robot_disruption": None
            if planner == "L0" and map_index == 0
            else map_index * 5 + level,
            "robot_free_flow_time": 10,
            "robot_elapsed_time": None
            if planner == "L0" and map_index == 0
            else 10 + map_index * 5 + level,
            "total_pedestrian_delay": map_index * 5 + level,
            "max_pedestrian_delay": level,
            "pedestrian_count": 3,
            "pedestrians_completed": 2 if planner == "L0" else 3,
            "baseline_completed": 3,
        }
        for behavior in BEHAVIORS
        for map_index, name in enumerate(("corridor", "crossing", "open_room"))
        for level, planner in enumerate(PLANNERS)
    ]


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(autouse=True)
def matplotlib_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "font-cache"))


def test_csv_preserves_failures_zero_values_and_map_order(csv_rows, tmp_path):
    data = read_results(write_csv(tmp_path / "summary.csv", csv_rows))
    assert data.behaviors == list(BEHAVIORS)
    assert [label for _, label in data.maps] == ["corridor", "crossing", "open_room"]
    first = data.points[0]
    assert first.disruption == 0
    assert not first.completed
    assert first.robot_disruption is None
    assert first.horizon_limited
    assert len(data.points) == 63
    assert data.points[1].robot_disruption == 1


def test_scatter_uses_color_for_awareness_shape_for_map_and_no_lines(csv_rows, tmp_path):
    from matplotlib.colors import to_rgba
    from matplotlib.markers import MarkerStyle

    data = read_results(write_csv(tmp_path / "summary.csv", csv_rows))
    figure = build_figure(data, "fixed")
    axis, failed = figure.axes
    assert not axis.lines and not failed.lines
    assert len(axis.collections) == 20
    assert len(failed.collections) == 1
    for panel in (axis, failed):
        for dot in panel.collections:
            planner, index = dot.get_gid().split("-")
            key = data.maps[int(index)][0]
            point = next(
                p
                for p in data.points
                if (p.planner, p.map_key, p.behavior) == (planner, key, "fixed")
            )
            assert tuple(dot.get_facecolors()[0]) == to_rgba(PLANNER_COLORS[planner])
            marker = MarkerStyle(map_markers(data)[key])
            expected = marker.get_path().transformed(marker.get_transform())
            assert dot.get_paths()[0].vertices.tolist() == expected.vertices.tolist()
            assert tuple(dot.get_offsets()[0]) == (
                point.robot_disruption if point.completed else 0,
                point.disruption,
            )
            assert point.completed == (panel is axis)
    assert axis.get_xlabel() == "Robot disruption: extra travel time (ticks)"
    assert axis.get_ylabel() == "Total pedestrian delay (person-ticks)"
    assert [t.get_text() for t in failed.get_xticklabels()] == ["Did not finish"]
    other = build_figure(data, "patient").axes[0]
    assert other.get_xlim() == axis.get_xlim()
    assert other.get_ylim() == axis.get_ylim()


def test_linear_x_and_symlog_y_preserve_zero_and_log_spacing(csv_rows, tmp_path):
    axis = build_figure(read_results(write_csv(tmp_path / "summary.csv", csv_rows)), "fixed").axes[
        0
    ]
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "symlog"
    transform = axis.yaxis.get_transform()
    assert math.isfinite(float(transform.transform(0)))
    ten, hundred, thousand = transform.transform([10, 100, 1000])
    assert hundred - ten == pytest.approx(thousand - hundred)
    assert 0 in axis.get_yticks()


@pytest.mark.parametrize("completed", [True, False])
def test_all_zero_and_uniform_outcome_can_render(csv_rows, tmp_path, completed):
    for row in csv_rows:
        row.update(total_pedestrian_delay=0, robot_completed=completed, robot_disruption=0)
    data = read_results(write_csv(tmp_path / "summary.csv", csv_rows))
    figure = build_figure(data, "fixed")
    figure.canvas.draw()
    assert len(figure.axes[0 if completed else 1].collections) == 21
    assert len(figure.axes[1 if completed else 0].collections) == 0
    assert all(
        math.isfinite(v) for axis in figure.axes for v in (*axis.get_xlim(), *axis.get_ylim())
    )


def test_missing_pairing_is_omitted_not_zero(csv_rows, tmp_path):
    del csv_rows[1]
    data = read_results(write_csv(tmp_path / "summary.csv", csv_rows))
    figure = build_figure(data, "fixed")
    assert sum(len(axis.collections) for axis in figure.axes) == 20


def test_metric_can_be_changed(csv_rows, tmp_path):
    data = read_results(write_csv(tmp_path / "summary.csv", csv_rows), "max_pedestrian_delay")
    axis = build_figure(data, "fixed").axes[0]
    assert axis.get_ylabel() == "Maximum individual delay (ticks)"
    assert max(p.disruption for p in data.points) == 6


def test_three_graphs_export_and_exact_values_are_saved(csv_rows, tmp_path):
    source = write_csv(tmp_path / "summary.csv", csv_rows)
    outputs = plot_results(source)
    assert len(outputs) == 6
    for behavior in BEHAVIORS:
        png = source.parent / "graphs" / f"{behavior}-{DEFAULT_METRIC}.png"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert ElementTree.parse(png.with_suffix(".svg")).getroot().tag.endswith("svg")
    with (source.parent / "graphs/scatter-data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 63
    assert rows[0]["robot_disruption"] == ""
    assert rows[0]["human_disruption"] == "0.0"
    assert float(rows[1]["robot_disruption"]) == 1


def test_plot_cli_uses_existing_results(csv_rows, tmp_path, capsys):
    source = write_csv(tmp_path / "summary.csv", csv_rows)
    destination = tmp_path / "charts"
    assert main(["plot", str(source), "--output", str(destination)]) == 0
    assert len(list(destination.glob("*.png"))) == 3
    assert "Saved" in capsys.readouterr().out


def test_plot_gui_flag_opens_interactive_results_without_running_simulation(monkeypatch, tmp_path):
    from human_nav.gui import ReplayViewer
    from human_nav.plotting import PlotData, ResultPoint
    from human_nav.results_gui import ResultsExplorer

    class Viewer:
        explorer = None
        notice = ""
        show_results = False

        def action(self, action):
            assert action == "results"
            self.show_results = True
            self.explorer = ResultsExplorer(
                PlotData(
                    (ResultPoint("map", "L0", "fixed", 0, True, 0),),
                    (("map", "Example"),),
                    DEFAULT_METRIC,
                )
            )

        def run(self):
            assert self.show_results
            calls.append("opened")

    calls = []
    monkeypatch.setattr("human_nav.plotting.plot_results", lambda *args: [])
    monkeypatch.setattr(ReplayViewer, "from_path", lambda path: Viewer())
    assert main(["plot", str(tmp_path), "--gui"]) == 0
    assert calls == ["opened"]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("robot_completed", "unknown", "robot_completed"),
        ("total_pedestrian_delay", "nan", "finite"),
        ("total_pedestrian_delay", "-1", "nonnegative"),
        ("total_pedestrian_delay", "", "number"),
        ("robot_disruption", "-1", "nonnegative"),
        ("robot_disruption", "inf", "finite"),
        ("planner", "unknown", "configuration"),
    ],
)
def test_bad_results_fail_clearly(csv_rows, tmp_path, field, value, message):
    csv_rows[1][field] = value
    with pytest.raises(ValueError, match=message):
        read_results(write_csv(tmp_path / "summary.csv", csv_rows))


def test_legacy_csv_uses_recorded_inputs_not_live_yaml(csv_rows, tmp_path):
    # Four steps, interval two: the unimpeded duration is seven ticks.
    scenario = parse_scenario(
        {
            "name": "corridor",
            "map": ["......"],
            "robot": {"start": [0, 0], "goal": [4, 0], "spawn_time": 5, "move_interval": 2},
            "simulation": {"max_steps": 30},
        }
    ).resolved()
    scenario["source"] = "scenarios/corridor.yaml"  # No live YAML exists here.
    saved = tmp_path / "recorded"
    saved.mkdir()
    (saved / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
    row = csv_rows[1].copy()
    for field in ("robot_disruption", "robot_free_flow_time", "robot_elapsed_time"):
        row.pop(field)
    row["robot_arrival_time"] = 15
    path = write_csv(tmp_path / "summary.csv", [row])
    before = path.read_bytes()
    point = read_results(path).points[0]
    assert point.robot_free_flow_time == 7
    assert point.robot_elapsed_time == 10
    assert point.robot_disruption == 3
    assert path.read_bytes() == before


def test_can_derive_from_elapsed_and_reference_without_snapshot(csv_rows, tmp_path):
    csv_rows[1]["robot_disruption"] = None
    point = read_results(write_csv(tmp_path / "summary.csv", csv_rows)).points[1]
    assert point.robot_disruption == 1


def test_missing_robot_measurement_is_not_invented(csv_rows, tmp_path):
    row = csv_rows[1].copy()
    for field in ("robot_disruption", "robot_free_flow_time", "robot_elapsed_time"):
        row.pop(field)
    with pytest.raises(ValueError, match="robot disruption is missing"):
        read_results(write_csv(tmp_path / "summary.csv", [row]))


def test_duplicate_pairings_require_explicit_aggregation(csv_rows, tmp_path):
    csv_rows.append(csv_rows[0])
    with pytest.raises(ValueError, match="multiple runs"):
        read_results(write_csv(tmp_path / "summary.csv", csv_rows))


def test_missing_columns_and_empty_csv(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("scenario,planner\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_results(path)
    path.write_text(
        "scenario,planner,behavior,robot_completed,total_pedestrian_delay\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no experiment rows"):
        read_results(path)
