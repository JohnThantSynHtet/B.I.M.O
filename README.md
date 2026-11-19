# B.I.M.O

An open-source quadruped robotics project exploring machine learning, reinforcement learning, and autonomous locomotion.

## Goals
- Build a functional quadruped robot
- Simulate locomotion using MuJoCo
- Develop and optimize walking gaits
- Experiment with reinforcement learning
- Transfer trained controllers to physical hardware

## Open-Source Foundations
- [Sesame Robot](https://github.com/dorianborian/sesame-robot)
- [Sesame ML](https://github.com/lukehollis/sesame-ml)

## Headless simulation smoke test

From this repository in WSL2 Ubuntu, with Python 3.11, Git, and uv installed:

```bash
env -u PYTHONPATH uv sync --locked
env -u PYTHONPATH uv run --locked bimo-smoke
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
```

The first sync requires network access. Dependencies live in B.I.M.O's own
`.venv`; Sesame ML is installed from a pinned Git revision, so no sibling checkout
or absolute path is required. No training extras are installed. The lockfile
pins resolved dependencies; Python is restricted to 3.11.

`bimo-smoke --seconds 5` loads Sesame's model and holds its calibrated stand
targets, then prints JSON. Expected baseline: 8 actuators, 13 bodies (including
the world), 0.002-second timestep, 2,500 steps, approximately 5 simulated seconds.
Durations must be finite and in `(0, 60]`, rounded up to whole physics steps.
The command exits nonzero on model-contract drift, non-finite state, simulation
warnings, or incorrect time advancement. It needs no display, GPU, ROS, or policy.

This is a simulation plumbing check, not evidence of walking, learned behavior,
or hardware safety. Exact floating-point results may vary by platform.

See [the audit and next milestone](docs/setup-audit.md) and
[engineering instructions](AGENTS.md).

## Scripted gait experiments and 3D replay

```bash
env -u PYTHONPATH uv sync --locked
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/forward-01
# Optional: replay in a 3D window and save the final rendered pose.
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/forward-view \
  --viewer --render-last
```

Each command compares standing against a bounded, scaled Sesame firmware gait
from the same settled pose. It writes a provenance/configuration/metrics report
and per-physics-step CSVs to a **new** directory. No Gym environment or training
stack is needed. Visualization replays measured poses after simulation.

The default five-second forward sequence moved backward and sideways in the
verified model; it is a reproducible control baseline, **not successful forward
walking**. See [Phase 1 measurements, controls, and limitations](docs/scripted-locomotion.md).

## Attribution and licensing

The model, calibration, and meshes are supplied by
[Sesame ML](https://github.com/lukehollis/sesame-ml), based on
[Sesame Robot](https://github.com/dorianborian/sesame-robot). Both inspected
repositories carry Apache-2.0 licenses. This integration imports their package;
it does not copy model code or assets into B.I.M.O. Preserve applicable upstream
license and notice files when redistributing dependencies or derived assets.
B.I.M.O does not yet declare a license for its own code; that remains an owner
decision.
