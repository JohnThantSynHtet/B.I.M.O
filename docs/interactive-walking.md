# Interactive walking simulator

From the B.I.M.O repository root in WSL2 Ubuntu:

```bash
env -u PYTHONPATH uv sync --locked
env -u PYTHONPATH uv run --locked bimo-walk
```

Requires a working graphical display (WSLg in the verified setup). It starts
the existing forward firmware gait and continues until the window is closed.
There is no duration limit, experiment output directory, training, or hardware
connection. Start with another motion or tempo if desired:

```bash
env -u PYTHONPATH uv run --locked bimo-walk --motion stand --speed 0.5
```

Click/focus the viewer before using the controls. The current motion, pause
state, tempo, and controls appear inside the window and in the launch terminal.

| Control | Action |
| --- | --- |
| 1 | Stand |
| 2 | Forward sequence |
| 3 | Backward sequence |
| 4 | Turn-left sequence |
| 5 | Turn-right sequence |
| P | Pause/resume physics and gait phase; camera remains interactive |
| R | Reset to the settled standing pose; preserve motion, tempo, and pause state |
| - | Reduce gait tempo by 0.25×, down to 0.25× |
| + or = | Increase gait tempo by 0.25×, up to 3× (keypad +/− also supported) |
| C | Recenter on the robot without changing zoom or viewing angles |
| Left mouse drag | Rotate camera |
| Right mouse drag | Pan camera |
| Mouse wheel | Zoom |
| Window close | End the session |

Use **P**, not MuJoCo's built-in Space shortcut, for pause. Built-in simulation
panels/physics controls are deliberately outside this sandbox's control path.
The camera is not forced to follow the robot; use C if it walks out of frame.

## Timing and behavior

`bimo.interactive.WalkingSession` reuses `JointPlayback`, the pinned Sesame model,
and calibrated standing initialization. It runs 0.002-second physics steps and
0.02-second command updates. Tempo changes keyframe duration (`0.15 / speed`
seconds), not gravity, physics timestep, amplitude, or torque limits. The default
0.25 amplitude and 3 rad/s command slew limit remain in effect. Faster tempo may
be limited by the actuator commands and is not a promised ground speed.

Motion/tempo changes restart the firmware sequence and startup ramp while
retaining the last command as the slew limiter's starting point. Switching to
stand therefore returns toward standing within the same slew bound. Reset is
an intentional pose/time reset, with 0.5 simulated seconds of settling; it does
not reset the user's camera. Pausing freezes both physics and phase exactly.

The loop advances one control interval per displayed frame. A slow renderer
slows wall-clock playback rather than changing simulation dt or accumulating
a catch-up backlog. Long interactive sessions are exploratory, not benchmark
measurements. The upstream sequence generator's per-update work grows with
elapsed sequence time; reset/reselect a motion to restart its phase if a very
long session becomes slow.

The display owns a separate model/data. Keyboard callbacks enqueue input; only
the simulation thread changes physics/control state. Camera interaction and
MuJoCo's built-in perturbations cannot alter the simulated robot. Scene options
are restored to avoid built-in shortcuts overlapping the walking keys.

Numerical warnings, non-finite state, or stalled simulation time pause the
session and show an error while leaving the window open. Use R to reset, then
P to resume. Physical tipping does not terminate this interactive sandbox;
reset when desired. The existing gait is not validated forward walking and may
slide, drift, saturate actuators, or fall. This is simulation-only.

## Regression boundary and verification

`bimo-gait`, `bimo-smoke`, replay rendering, experiment metrics, and their
deterministic tests retain their existing behavior. Live interactive sessions
do not write or replace experiment artifacts. No dependencies were added.

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
env -u PYTHONPATH uv run --locked bimo-smoke
```

Tests cover continuous stepping past five seconds, pause/resume, all five
motions, transition slew limits, repeatable reset, tempo boundaries, numerical
error recovery, and the viewer event loop remaining open while paused. The
viewer loop is tested headlessly with a fake window; real WSLg interaction is
checked separately for keyboard controls, camera rotation/zoom, and closing.

Verified on the current WSL2/WSLg setup: **49 tests passed**, including all 35
existing tests. The actual viewer was exercised for selection, pause/resume,
reset, tempo changes, camera rotation/zoom, and window close. The final on-screen
guide and running state were visually checked. The five-second smoke run passed;
the headless standing/gait comparison retains its Phase 1 metrics.
