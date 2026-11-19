# Phase 1: deterministic scripted locomotion baseline

## Implementation and control contract

`bimo.gait` adapts the existing Sesame `FirmwareSequencePolicy`; it does not copy
the firmware keyframes or implement a new physical model. The upstream policy
provides a fixed-motion mode and public `servo_targets_rad` property, usable
without `sesame_ml.envs`. Its `predict()` return is a clipped **Gym residual**,
not an absolute joint command, so this runner deliberately ignores that return.
The fixed-motion path does not read the empty observation mapping supplied to it.
The policy's AUTO mode and CPG controller require an environment observation
contract and are excluded from this minimal baseline.

The default gait is `forward`, with 0.25 amplitude about the calibrated standing
angles, 0.15-second interpolated firmware keyframes, a 0.5-second startup ramp,
and 3 rad/s maximum commanded slew. Physics runs at 500 Hz, command updates at
50 Hz. The physics timestep must divide the control period exactly. Model,
actuator-order, and finite-state checks remain active. No randomness or training
is used. Forward/backward and left/right turning sequences are selectable, but
only the default forward sequence is performance-characterized here.

Targets are clamped to the intersection of canonical mechanical, MJCF joint,
and actuator control ranges, then slew-limited. Actual state can still overshoot
targets because this is dynamic simulation with soft constraints. Command-rate
limits do **not** cap physical joint velocity. Model torque saturation remains
unchanged. These safeguards are for bounded simulation, not hardware approval.

`bimo.experiment` owns stepping, measurements, and artifacts. Both trials start
with a fresh model/data and 0.5 seconds holding stand. That settling interval is
excluded from reported displacement and duration. Durations round up to a whole
physics step. Gait phase starts after settling; standing uses identical settings
without joint excursions. Fall criteria are checked at every physics step and
stop further stepping/commands at the first detected fall. Numerical warnings,
non-finite states, and stalled time are errors rather than successful results.

## Reproduce

From the B.I.M.O root in WSL2 Ubuntu:

```bash
env -u PYTHONPATH uv sync --locked
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/reference --render-last
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/repeat
cmp outputs/reference/report.json outputs/repeat/report.json
cmp outputs/reference/stand.csv outputs/repeat/stand.csv
cmp outputs/reference/gait.csv outputs/repeat/gait.csv
```

Each output directory must be new; existing data is never overwritten. On an
error, a directory may contain partial CSVs; only a completed `report.json`
constitutes a finished comparison. Rendering errors occur after report creation
and do not invalidate saved physics results. The command returns nonzero for
setup/numerical/rendering errors; a correctly measured fall is an experiment
outcome reported in JSON, not a process failure.

Examples of configurable controls:

```bash
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/slower \
  --motion forward --amplitude 0.2 --frame-seconds 0.2 \
  --ramp-seconds 1 --max-speed-rad-s 2 --seconds 5
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/view --viewer
```

`--settle-seconds`, `--fall-height-m`, and `--fall-tilt-deg` expose evaluation
settings. Defaults classify base height below 0.025 m **or** body-up tilt above
60 degrees as a fall. These are explicit heuristic thresholds, not validated
physical fall labels. Fixed-world flat-floor height is assumed; these settings
must be reconsidered for uneven terrain. The model includes its existing target
and obstacles; no geometry, friction, solver, or torque settings were altered.

## Visualization

`--viewer` replays the measured gait at simulated timing in MuJoCo's passive 3D
viewer. It uses a **separate model/data**, so UI interactions cannot affect the
recorded experiment. Close the window to stop replay; experiment files remain.
Display is subsampled to 50 Hz and includes the terminal pose. A sufficiently
slow display may lag; it never changes physics or recorded timestamps.
`--render-last` writes `gait-final.png` (640 × 480). Rendering is optional and
imported only when requested. Both use the existing MuJoCo installation; PNG
encoding uses Pillow already supplied by Sesame ML.

WSLg/display support is required for the interactive viewer. With no display,
omit visualization flags; for offscreen images an available EGL backend can be
selected via `MUJOCO_GL=egl` before process start. No renderer is needed for the
headless tests. See the [MuJoCo Python viewer API](https://mujoco.readthedocs.io/en/3.3.3/python.html).

## Measurement definitions

- XY displacement is final minus initial base position in world coordinates,
  in meters. Forward/lateral components project this onto the **initial** base
  heading (+x forward, +y left), not its changing instantaneous heading.
- Mean XY and forward velocity divide net displacement by actual trial duration.
  Mean path speed divides summed XY segment lengths by duration. Peak planar
  speed is the largest per-step XY finite difference divided by its timestep.
  Path length is not forward progress; sliding and jitter contribute to it.
- Heading drift is unwrapped ZYX yaw relative to the initial heading. Both final
  and maximum absolute drift are reported in degrees, avoiding wrap discontinuities.
- Body stability includes minimum height, height standard deviation, absolute
  roll/pitch RMS, and maximum body-up tilt. Pose transforms are refreshed after
  integration with `mj_forward` before measurement.
- `*_violations_samples` counts **joint-or-actuator × post-step samples**, using
  a 1e-6 rad or Nm tolerance, not distinct incidents. Command violations check
  applied targets; joint violations check simulated angles against MJCF limits;
  mechanical violations check canonical servo limits; force violations check
  model actuator force ranges. Mechanical ranges are narrower than the MJCF
  distal-joint ranges, so their counts can differ even while standing.
- Force saturations count actuator samples at the model's torque bound, not
  force violations. Maximum canonical mechanical excursion, maximum joint speed,
  and peak absolute torque quantify severity. Requested target clips and slew
  interventions count affected joints at **control updates**, separately.
- A fall terminates that trial. Compare reported actual durations; a shortened
  gait cannot be compared fairly with a full-duration stand using displacement
  alone. `equal_duration` makes this mismatch explicit.

CSV files record every physics step plus the initial settled state, with base
pose, angles, diagnostics, model-order `qpos_0...qpos_14`, and named joint targets
and torques. JSON records configurations, versions, upstream Git provenance,
and SHA-256 digests of B.I.M.O Python source and upstream Python/model/assets.
Outputs omit wall-clock timestamps so identical runs on the same stack can be
byte-compared. Cross-machine bitwise identity is not promised.

## Verified reference and next steps

Reference artifacts are under `outputs/phase1-reference/` (ignored by Git).
The measured default configuration is deliberately retained rather than tuned
after inspecting its performance. Both trials ran 5 seconds after settling:

| Measurement | Standing | Default forward gait |
| --- | ---: | ---: |
| Forward displacement | approximately 0 m | -0.037984 m |
| Lateral displacement | approximately 0 m | +0.066089 m |
| Mean forward velocity | approximately 0 m/s | -0.007597 m/s |
| XY path length | approximately 0 m | 0.298863 m |
| Final heading drift | approximately 0 degrees | +0.120943 degrees |
| Maximum absolute heading drift | approximately 0 degrees | 19.745261 degrees |
| Maximum tilt | 0.018431 degrees | 1.366056 degrees |
| Minimum base height | 0.046681 m | 0.046360 m |
| Falls detected | no | no |
| Applied-command / MJCF-joint / force violation samples | 0 / 0 / 0 | 0 / 0 / 0 |
| Canonical mechanical-range violation samples | 10,000 | 2,317 |
| Maximum canonical mechanical excursion | 0.000306 rad | 0.081511 rad |
| Torque saturation samples | 0 | 746 |
| Maximum actual joint speed | approximately 0 rad/s | 27.442823 rad/s |

The movement is backward and sideways despite the upstream sequence name.
It does **not** improve forward locomotion over standing. Final heading nearly
returns to its start but hides a substantial intermediate excursion. Mechanical
excursions and torque saturation also rule out interpreting zero applied-command
violations as physical safety. The tiny steady stand excursion is permitted by
the model's wider distal-joint ranges and compliant dynamics.

Verification on this WSL2 stack: **35 tests passed**, including the original 13.
Two full five-second runs produced byte-identical reports and both CSVs.
Final-pose rendering succeeded and the image was visually inspected. A one-second
interactive viewer replay opened and completed successfully (exit code 0).
`uv pip check` and `git diff --check` passed. No new dependencies, upstream edits,
ML training, physical control, commits, or pushes were performed. Bitwise
repeatability on other operating systems/CPUs and actual hardware fidelity have
not been established.

Next, inspect foot contacts and servo/leg coordinate conventions to explain the
backward/sideways bias, then evaluate a small declared set of gait parameters
against explicit forward-progress, heading, saturation, and fall criteria.
Do not treat a visually moving robot as a validated gait. The high physical
joint speed despite bounded commands warrants actuator-model validation before
any hardware transfer. Gym environment and PyTorch repairs remain separate;
ML training, hardware control, commits, and pushes require approval.
