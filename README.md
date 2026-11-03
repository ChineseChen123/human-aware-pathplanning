# Human-aware robot navigation

A Python simulator for comparing robot efficiency, pedestrian disruption, and
personal space. Pedestrians use **spatial A\***. Robot configurations share a
time-expanded A* implementation. Maps, starts, destinations, speeds, waypoints,
and behavior profiles are editable in YAML.

The [pygame-ce](https://pyga.me/docs/) viewer plays back recorded discrete states
with smooth movement between timesteps. Playback speed never changes simulation
results. No robotics framework, GPU, or external service is needed.

## Run the installed project

A local Python environment and dependencies are already installed in `.venv`.
From this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m human_nav
```

This computes the default intersection experiment and opens its animated replay.
For a different map, robot configuration, or pedestrian behavior:

```powershell
.\.venv\Scripts\python.exe -m human_nav run scenarios/doorway.yaml --planner L4+D --behavior patient --gui
```

Without `--gui`, a run saves results and exits. An existing experiment can be
replayed without recalculating paths:

```powershell
.\.venv\Scripts\python.exe -m human_nav replay results/demo
```

Add `--paused --tick 7.5` to inspect a particular moment before starting playback.

Open an entire saved batch in one viewer:

```powershell
.\.venv\Scripts\python.exe -m human_nav replay results/benchmark --paused
```

You can also click **Open batch...** inside either a single replay or the batch
viewer and select a batch results folder. Use the **Map**, **Pedestrians**, and
**Robot** selectors to choose a saved combination, or **Previous / Next Replay**
to browse all runs. Page Up / Page Down also switch replays and wrap at the ends.
Switching rewinds the timeline and preserves speed, pause state, and overlays.
Changing maps keeps the selected behavior and robot configuration when that
combination exists; incomplete batches offer only available choices. Dropdowns
pause animation while open and scroll when there are many choices.

Batch browsing uses saved recordings and never reruns simulations or applies
changes from the current YAML settings. The viewer indexes lightweight metadata
and loads only the selected timeline. Open one batch directory at a time, rather
than a parent directory containing multiple batches of the same scenarios.

Viewer controls:

The window starts at up to **1100 x 760**, reduced further on smaller desktops.
Drag its edges or maximize it to resize. The whole interface scales to fit while
preserving square grid cells; selectors and timeline clicks follow the resized
layout. Switching replays or opening another batch preserves the window size.

| Action | Control |
|---|---|
| Play or pause | Space / Play button |
| Advance or rewind one discrete timestep | Right / Left |
| Change animation speed | Up / Down or + / - |
| Restart | Home / Reset |
| Jump to a timestep | Click the timeline |
| Personal-space halos | H |
| No-robot baseline ghosts | B |
| Full executed robot route | P |
| Current planned robot route / build it for older replays | R |
| Switch batch replay | Page Up / Page Down or replay buttons |
| Close an open selector | Escape / click outside |
| Close | Escape |

Blue squares are robots; colored numbered circles are pedestrians. S and G mark
the robot's start and final goal. The sidebar shows **final-run metrics**, not
partial metrics at the current playback position.

The **green route** shows the robot's current A* plan, with arrows indicating
direction and circles marking planned waits. It updates at every eligible
planning timestep. The robot briefly flashes a ring at each recalculation, and
the map header shows the plan number, decision tick, route changes, and blocked
moves. A replaced route briefly appears as **dashed amber**. The separate **P**
overlay shows the full actual recorded route in pale blue. **R** toggles the
planned route. Pause, step, rewind, and speed controls work with both overlays.

New recordings save the actual per-step decisions in `robot_plans`, including
planned positions, goal, search outcome, proposed move, and executed move.
Displaying a recording does not rerun its planner. For older recordings, click
**R Build planned routes** to reconstruct just the selected replay. The viewer
uses its recorded scenario snapshot, verifies every movement against the saved
timeline, and caches the recovered plans in memory. It does not edit replay
files or read current YAML defaults. Playback waits while reconstruction runs;
the interface remains responsive. If the current simulation cannot reproduce
the recorded movements, the viewer reports that mismatch and leaves the original
recording intact.

## Install elsewhere

Use standard CPython 3.10+ (not an MSYS Python build). With
[uv](https://docs.astral.sh/uv/), the included `uv.lock` pins dependencies:

```powershell
uv sync --extra dev --locked
uv run python -m human_nav
```

Alternatively:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Dependencies are pygame-ce, PyYAML, and Matplotlib; pytest and Ruff are development tools.

## Edit a scenario

Start with one of the five files in `scenarios/`. Coordinates are **[x, y]**:
column first, row second, starting at zero in the top-left. `#` means wall and
`.` means walkable. Every row must have the same width. Quote rows containing
`#`, or use YAML's multiline `|` form as in the examples.

```yaml
name: Small crossing
map: |
  #######
  #.....#
  #.....#
  #.....#
  #######
robot:
  start: [1, 2]
  goal: [5, 2]
defaults:
  behavior: patient
  move_interval: 1
  personal_radius: 2.0
  direction_order: NESW
pedestrians:
  - id: pedestrian_1
    start: [3, 1]
    goal: [3, 3]
    spawn_time: 0
simulation:
  max_steps: 100
  search_horizon: 45
  max_expansions: 15000
  prediction_steps: 5
  alpha: 1.0
  beta: 3.0
  gamma: 4.0
  fairness_weight: 0.25
  seconds_per_step: 0.35
  seed: 0
```

Shared settings live in the project's root **`constants.yaml`**, loaded
automatically for every scenario, including scenarios outside `scenarios/`.
The file is located relative to the installed project, not the working directory.
The current shared setting is:

```yaml
defaults:
  wait_threshold: 1
simulation: {}
```

Change `wait_threshold` there to change the patient rerouting threshold across
all included maps. With `1`, a pedestrian attempts rerouting on the first
blocked eligible movement attempt. With `3`, it waits on the first two blocked
attempts and tries rerouting on the third. Successful movement resets the counter.
Shared agent defaults can be added under `defaults`, and shared simulation
settings under `simulation`.

Precedence is built-in defaults, then `constants.yaml`, then scenario defaults
or simulation settings, then individual agent fields or supported command-line
overrides. Local overrides are optional; the included maps inherit
`wait_threshold` entirely from the shared file. Changes apply to newly loaded
scenarios; saved results and replay settings remain unchanged.

Every pedestrian needs a unique `id`, `start`,
and `goal`; the robot ID defaults to `robot`. Optional `waypoints` are a list of
coordinates visited before the final goal. `move_interval: 2` permits one move
every two timesteps. `personal_radius` is in cell units. `direction_order` is
any permutation of NESW and breaks route ties deterministically.

`spawn_time` is the requested entry timestep. If its start cell is occupied,
an agent waits outside the map until the cell becomes free. Its movement phase
is anchored to the requested spawn time, and entry delays count toward travel
time. Pedestrians stop participating in movement and collisions after reaching
their final goal, but remain visible as desaturated figures blended into the
destination cell background. They have no personal-space halo and do not block
the robot or other pedestrians; active agents are drawn over them. Shared goal
cells show a faded count such as `2x`. Reaching an intermediate waypoint does not
complete a pedestrian's trip. The robot still leaves after its final arrival.

`seed` is saved as metadata for future stochastic extensions. **The current
model is deterministic and does not use randomness**; changing the seed alone
does not produce a new scenario.

Validate files before experimenting:

```powershell
.\.venv\Scripts\python.exe -m human_nav validate "scenarios/*.yaml"
```

The loader rejects malformed maps, invalid coordinates, duplicate IDs,
same-time overlapping starts, unknown fields, invalid numeric settings, and
goals disconnected by walls. Dynamic congestion and deadlocks are permitted
experimental outcomes.

## Pedestrian navigation

All three profiles use the same spatial A* with Manhattan-distance heuristic.
An initial route considers walls only. Routes are cached rather than recomputed
at every step. The simulator first checks everyone's intended move using the
shared conflict resolver. Following a pedestrian into a cell they are safely
vacating is allowed and does **not** trigger rerouting or count as waiting.
Only genuinely rejected moves trigger the configured rerouting response.
Blocking pedestrians are considered before their followers, so a leader's
successful detour can let the rest of the line continue on its existing route.
Rerouting treats cells that remain occupied, plus a rejected destination, as
temporary obstacles; it does not predict future timesteps.

| Profile | When the next route cell is blocked |
|---|---|
| `fixed` | Keep the route and attempt to advance; wait if conflict resolution rejects the move |
| `reroute` | Immediately try a replacement route; wait if none exists |
| `patient` | Try a replacement route after `wait_threshold` blocked eligible movement attempts |

This corrects the earlier occupancy-only rule, which could make pedestrians
detour simply because someone was walking immediately ahead of them. The change
applies to new simulations and human forecasts. Existing saved replays retain
their original behavior; rerun a batch to evaluate the updated model.

A common simultaneous resolver prevents shared destinations, opposite-direction
edge swaps, and movement into agents forced to stay. Moving chains and longer
cycles are allowed. Stationary occupants cannot be displaced. Other conflicts
use a rotating priority over stable agent IDs. Baselines retain the absent
robot's priority slot, preserving the human-human arbitration schedule.

## Robot configurations

| Configuration | Human model | Social cost |
|---|---|---|
| `L0` | Ignore humans during planning | None |
| `L1` | Freeze current human positions | Occupied cells forbidden |
| `L2` | Freeze current human positions | Occupancy + personal space |
| `L3` | Predict one timestep | Occupancy + personal space |
| `L4` | Predict `prediction_steps` timesteps | Occupancy + personal space |
| `L3+D` | Predict one timestep | Occupancy + personal space + interference |
| `L4+D` | Predict several timesteps | Occupancy + personal space + interference |

Every configuration uses the same physical resolver, so even L0 cannot execute
a collision. Replanning occurs on every eligible robot movement timestep.
Predictions simulate humans interacting with each other **without the robot**,
starting from the current observed state and retaining their existing route
memories. The predictor knows pedestrian goals and behavior profiles. It does
not assume that its forecasts will remain correct after the robot intervenes.

After the prediction window, people freeze at their last predicted positions.
Known arrivals are reserved during their arrival transition and cease to occupy
space afterward, including when the arrival is the last predicted timestep.
The viewer's faded destination figures are visual only. Future
entries beyond the prediction window are unknown to that planning call.

The per-transition objective is:

`alpha + beta * proximity + gamma * interference`

Only the relevant terms are enabled for each configuration. Moves and waits
both cost `alpha`. Proximity sums `max(0, 1 - distance/radius)^2` over people.
All planners use `alpha * wall_only_distance_to_goal` as their admissible A*
heuristic. Reverse distance fields are preprocessing; pedestrian routing uses A*.
Frozen-map feasibility checks prune disconnected states without changing the
heuristic.

The disruption variants use two nonnegative **surrogates**, not actual delay:

1. **Exit obstruction:** At a predicted human position, compare the best
   remaining map distance available among its neighboring cells plus waiting,
   with and without the robot's candidate cell. Penalize lost best progress.
2. **Future cell demand:** Penalize the candidate cell if a human is predicted
   to use it later in the available forecast, discounted by `1 / steps_ahead`.

Exit obstruction makes `L3+D` meaningful even when the forecast contains only
one future position. Same-time occupancy alone would be redundant with forbidden
collision constraints. These interference costs stop at the actual forecast
window; extrapolated frozen positions are not counted as new evidence of demand.

Planning expands at most `max_expansions` nodes and considers at most
`search_horizon` future timesteps. If a goal lies beyond that horizon, the
planner uses the best boundary route under `g + h` and executes its first step.
No route or an expansion-budget exit produces a safe wait. These events are
logged. Optimality applies to the modeled finite search, not to the coupled
realized multi-agent system.

## Evaluation and saved results

Each robot run is matched with the same scenario without the robot. Humans
continue moving after the robot arrives. Each output directory contains:

- `scenario.json`: all resolved inputs and defaults.
- `metrics.json`: aggregate and per-pedestrian measurements.
- `replay.json`: full discrete trace and matched baseline for playback.

For pedestrian i, delay is `max(0, robot_arrival_i - baseline_arrival_i)`.
Clip each person's difference before summing, so another person's earlier
arrival cannot cancel their inconvenience. Improvements are reported separately.

For an unfinished trip, replace its unknown final arrival by the common
evaluation horizon **only to calculate horizon-restricted elapsed time**. This
is accumulated disruption within the observation period, not a claim about
final delay. Completion counts for both conditions remain explicit; zero
restricted delay does not establish that an unfinished trip was unaffected.

Metrics include total/mean/maximum human delay, completion rates, signed
completed-trip differences, personal-space exposure, minimum separation,
robot arrival/distance/waits, and planning time/expansions/budget exits.
`robot_free_flow_time` is the shortest wall-only trip duration through all robot
waypoints at its configured movement interval. `robot_disruption` is
`max(0, robot_elapsed_time - robot_free_flow_time)`, in ticks. Elapsed time starts
at the scheduled spawn, so delayed entry, waiting, and detours count. Unfinished
robots have no final elapsed time or disruption value.
The optional fairness score is
`(1 - fairness_weight) * mean_delay + fairness_weight * max_delay`.

Personal-space exposure samples **actual positions at the start of each
simulation interval**, summing the same proximity kernel over people and time.
Times and distances are ticks and cells. `seconds_per_step` controls playback
only. Waiting and walking-distance differences are diagnostics, not additional
terms added to measured delay; unfinished-trip differences are partial totals.

## Run all 21 pairings

```powershell
.\.venv\Scripts\python.exe -m human_nav batch "scenarios/*.yaml" --output results/benchmark
```

For each map this runs all three homogeneous crowd profiles against all seven
robot configurations: 21 robot runs and 3 reusable no-robot baselines. Five
included maps therefore produce **105 robot runs plus 15 baselines**. Batch
output includes `summary.csv`, `manifest.json`, and replay/metric files for
every pairing. The batch command also generates the results graphs described
below. A single run preserves mixed behaviors from YAML unless
`--behavior` overrides them; batch intentionally applies homogeneous profiles.

The included layouts are an intersection, corridor with passing bay, doorway,
crowded route with a detour, and open room. They are development examples, not
a tuned final research benchmark. Create population/timing/density variants as
additional YAML files. Lock weights before evaluating a held-out benchmark.
Results can include human or robot deadlocks; more awareness is not guaranteed
to perform better under every scenario and forecast assumption.

## Graph the results

Generate graphs from an existing batch without rerunning the simulation:

```powershell
.\.venv\Scripts\python.exe -m human_nav plot results/benchmark
```

This saves a separate graph for each pedestrian behavior in
`results/benchmark/graphs/`, in both PNG and SVG formats. The batch command
generates these graphs automatically after completing its runs.

- **Horizontal axis:** robot disruption (extra trip time in ticks), on a linear scale.
- **Vertical axis:** total pedestrian delay in person-ticks, by default.
- **Color:** human-awareness level (`L0` through `L4+D`).
- **Shape:** map. Colors and shapes stay consistent across behavior graphs.
- **Separate right column:** robots that did not finish before the time limit.
  This column is categorical; it does not assign a numeric disruption to failures.
- **Scatter points only:** no lines connect observations.

All behavior graphs use the same **symmetric logarithmic (symlog) vertical scale**:
base-10 logarithmic above 1, and linear from 0 to 1 so exact zero-disruption
outcomes remain visible. Values are neither replaced by an epsilon nor dropped.
All graphs also share numeric X limits. Exact overlapping points are separated
by a few screen pixels for visibility. The exported `scatter-data.csv` contains
the exact measurements, including robot reference durations recovered for older
batches. Unfinished pedestrian trips still use the recorded horizon-limited
human delay. Missing pairings are omitted.

New simulations save robot disruption in their metrics and batch CSV. For older
batches, plotting derives it from recorded arrival times and `scenario.json`
snapshots, so editing the original scenario YAML cannot change the reference.
Keep those snapshots beside an older CSV when moving a batch.

Select another disruption metric or output directory:

```powershell
.\.venv\Scripts\python.exe -m human_nav plot results/benchmark/summary.csv --metric max_pedestrian_delay --output results/max-delay-graphs
```

Available metrics: `total_pedestrian_delay`, `mean_pedestrian_delay`,
`max_pedestrian_delay`, `fairness_delay_score`, and `personal_space_exposure`.
Each map/behavior/configuration pairing must have one row. Repeated trials
are rejected rather than silently averaging their success/failure markers.

## Explore results in the GUI

Open a saved batch with the replay viewer and click **3D results**. To generate
the PNG/SVG plots and open the explorer directly:

```powershell
.\.venv\Scripts\python.exe -m human_nav plot results/final --gui
```

The interactive cube shows linear robot disruption left to right, human
disruption vertically (the same symlog scale as the 2D plots), and the seven
awareness configurations along the depth axis. Awareness is categorical:
equally spaced levels do not imply equal increases in capability. **Map colors**
stay consistent across behaviors and views. Circles are completed robot trips;
crosses occupy a separate pale plane beyond the numeric robot-disruption range
for unfinished trips. In a face view the third axis is collapsed, so points from
different levels or outcomes can overlap. Hover for their exact measurements.

- Left-drag to rotate; right/middle-drag or Shift-drag to pan.
- Use the mouse wheel or **+ / -** to zoom. **Reset view** restores framing.
- The three **Snap to two axes** buttons smoothly rotate to robot/human,
  awareness/human, or robot/awareness views. Keys **1**, **2**, **3** also work.
- **Slices by awareness** (or **S**) replaces the cube with seven separate
  robot-versus-human disruption plots, perpendicular to the awareness axis.
  Slices share axes, zoom, and pan; drag to pan them together. Each retains a
  separate **DNF** column. Use the toggle again to return to the cube.
- Click **Pedestrians** to cycle behaviors, or **Metric** to cycle human metrics.
- Click map names to hide/show them; **Show all maps** restores the full batch.
- Hover a point for map, awareness, behavior, and exact disruption values.
  Exact overlaps are spread slightly on screen; the recorded data is unchanged.
- **Back to replay** or **Esc** returns to playback at the same timestep. Replay
  time pauses while inspecting results. **R** resets the graph view.

The explorer uses the existing pygame-ce renderer and mouse events, without
requiring a browser, graphics driver extension, or additional dependencies.
Window resizing and mouse coordinates follow the same scaled layout as replay
playback. A replay-only batch without `summary.csv` can also be explored from
the replay window; metrics are loaded from its saved sidecar files when present.

Useful overrides:

```powershell
.\.venv\Scripts\python.exe -m human_nav run scenarios/open_room.yaml --planner L4+D --prediction-steps 10 --gamma 8 --gui
.\.venv\Scripts\python.exe -m human_nav snapshot results/demo --tick 7.5 --output results/demo/preview.png
```

## Tests and structure

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check human_nav tests run_demo.py
```

Tests cover shortest routes, randomized collision safety, all 21 pairings,
deterministic traces, forecast isolation and arrival handling, behavior profiles,
nontrivial interference costs, censored delay, input validation, batch artifacts,
and smooth frame rendering. If the system temporary directory is restricted,
pass pytest `--basetemp` with a fresh path inside this project.

To audit saved benchmark traces, including arrival transitions where agents
disappear from the next frame:

```powershell
.\.venv\Scripts\python.exe tests/audit_replays.py results/benchmark
```

| File | Responsibility |
|---|---|
| `config.py` | YAML schema and validation |
| `search.py` | Pedestrian A* and reverse distance fields |
| `engine.py` | Movement, spawning, collision resolution, human forecasting |
| `planners.py` | Seven robot configurations |
| `experiments.py` | Paired metrics, batch runner, replay export |
| `robot_metrics.py` | Robot free-flow reference duration |
| `plotting.py` | Per-behavior disruption scatter plots and unfinished-run column |
| `results_gui.py` | Interactive results cube, animated camera, and awareness slices |
| `gui.py` | pygame-ce interpolation and playback controls |
| `replays.py` | Lazy batch index and map/behavior/robot selection |
| `plan_replay.py` | Verified reconstruction of plans for older recordings |
| `cli.py` | Run, replay, validate, batch, snapshot commands |

The engine and headless experiments do not import pygame. Rendering is a
separate consumer of recorded results.
