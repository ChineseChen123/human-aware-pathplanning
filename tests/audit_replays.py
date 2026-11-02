"""Audit saved benchmark traces, including collisions hidden by agent removal.

Usage: python tests/audit_replays.py results/benchmark
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def audit(root: Path) -> int:
    files = list(root.rglob("replay.json"))
    if not files:
        raise ValueError(f"No replays found in {root}")
    for path in files:
        replay = json.loads(path.read_text(encoding="utf-8"))
        scenario = replay["scenario"]
        specs = {a["id"]: a for a in [scenario["robot"], *scenario["pedestrians"]]}
        robot_id = scenario["robot"]["id"]
        for key in ("frames", "baseline_frames"):
            frames = replay[key]
            for before, after in zip(frames, frames[1:]):
                assert after["tick"] == before["tick"] + 1, path
                old = {ident: tuple(p) for ident, p in before["positions"].items()}
                endpoints = {}
                for ident, p in old.items():
                    if after["arrivals"].get(ident) == after["tick"]:
                        q = tuple(specs[ident]["goals"][-1])
                    else:
                        q = tuple(after["positions"][ident])
                    assert abs(q[0] - p[0]) + abs(q[1] - p[1]) <= 1, path
                    assert scenario["grid"]["rows"][q[1]][q[0]] == ".", path
                    endpoints[ident] = q
                assert len(set(endpoints.values())) == len(endpoints), path
                actual = list(map(tuple, after["positions"].values()))
                assert len(actual) == len(set(actual)), path
                for a, q in endpoints.items():
                    for b in endpoints:
                        if a != b:
                            assert not (q == old[b] and endpoints[b] == old[a]), path

        exposure = 0.0
        for frame in replay["frames"][:-1]:
            robot = frame["positions"].get(robot_id)
            if robot is None:
                continue
            for spec in scenario["pedestrians"]:
                person = frame["positions"].get(spec["id"])
                if person is not None:
                    distance = math.dist(robot, person)
                    exposure += max(0, 1 - distance / spec["personal_radius"]) ** 2
        assert math.isclose(exposure, replay["metrics"]["personal_space_exposure"], abs_tol=1e-9), (
            path
        )
    return len(files)


if __name__ == "__main__":
    print(f"Audited {audit(Path(sys.argv[1]))} saved robot runs and their baseline traces.")
