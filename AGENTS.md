# B.I.M.O engineering guidance

## Context and boundaries

B.I.M.O is an ML-driven quadruped project built on Sesame Robot hardware and
Sesame ML simulation assets. Current tools include a headless stand-pose smoke
test and deterministic firmware-gait experiments with optional 3D replay.
Neither is trained locomotion or validated hardware control.

- Modify this repository only unless the user explicitly authorizes otherwise.
- Treat sibling `sesame-robot` and `sesame-ml` checkouts as read-only references.
- `src/bimo/` owns B.I.M.O orchestration and validation. Sesame ML owns the
  physical model, meshes, joint ordering, and calibrated stand initialization.
- `tests/` owns automated checks; `docs/` owns setup evidence and design notes.
- Keep rendering, Gymnasium environments, training, and hardware adapters outside
  the smoke-test import path. Do not import `sesame_ml.envs` for model loading.
- Gait playback uses absolute `servo_targets_rad`, not the firmware policy's
  normalized residual return value. Preserve canonical actuator ordering.
- Run measured physics independently of rendering. Replay recorded poses in a
  separate model so mouse/UI interactions cannot change experiment results.
- Report requested-command clipping, actual state/force violations, saturation,
  and fall termination separately. Never call displacement alone successful walking.
- No ML training, hardware modification/control, commits, or pushes without
  explicit user approval. Collection-only inspection is not permission to run
  upstream training or hardware tests.

## Coding standards

- Prefer small typed functions, explicit inputs, and standard-library tools.
- Plan complex changes first; keep changes scoped and explain tradeoffs.
- Do not embed machine-specific paths or mutate upstream files at runtime.
- Reuse upstream model APIs; do not duplicate calibration, meshes, or physics.
- Validate numerical inputs and fail on non-finite state or MuJoCo warnings.
- Keep dependencies minimal. Update `pyproject.toml` and `uv.lock` together.
- Preserve upstream attribution and licenses. Do not assign a project license
  without the owner's decision or imply all upstream assets are original work.
- Never infer physical safety or policy quality from a successful smoke test.

## Required validation

Run from this repository in WSL/Linux, with Python 3.11 and uv:

```bash
env -u PYTHONPATH uv sync --locked
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -q
env -u PYTHONPATH uv run --locked bimo-smoke
env -u PYTHONPATH uv run --locked bimo-gait --output outputs/new-experiment
```

ROS 2 can inject incompatible Python paths and pytest plugins. Keep the isolation
flags; explicitly enable plugins if future tests need them. Tests must exercise
the real model, duration boundaries, failure detection, and CLI output. Do not
silently skip missing required simulation dependencies. Review the final diff
and report executed results, failures, and limitations. Do not invent success.
For gait changes, compare standing and gait from identical settled states,
verify repeatability, and test units/heading wrap, command limits, fall stops,
and artifact overwrite refusal. Keep generated CSV/images under ignored outputs/;
document the exact configuration and measured results. Rendering tests are
optional on machines without graphics; identify any unverified viewer behavior.

For approved read-only upstream tests, disable bytecode and pytest's cache
(`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`). Preserve existing dirty files.
