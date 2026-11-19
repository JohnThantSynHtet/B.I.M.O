# Setup audit — 2026-11-18

## Scope and provenance

Inspected WSL2 Ubuntu repositories without modifying upstream source or their
environments. B.I.M.O started clean on `main` at
`f4777d5addbcc4c15dda160ccbbb4cd9b78c7718`, containing a README and ignore file.
Sesame Robot was at `449ad1e237fa419f575ea1bb183716134e02310c`; Sesame ML was at
`a3910f67b4aeb9ca9080a1ba1ffa971e53e35a18`. Both upstream worktrees were dirty;
their diffs were empty with `--ignore-space-at-eol`. Those changes were preserved.
This is a bounded model/setup audit, not a full hardware or security review.

The B.I.M.O dependency is pinned to the inspected Sesame ML commit. Its model
loader supplies packaged meshes, validates actuator ordering, and initializes
the calibrated stand pose. B.I.M.O adds a small validation/CLI layer, not a new
physical model. Model counts are an intentional regression contract.

## Reproduced baseline

The existing upstream environment reported Python 3.11.18, MuJoCo 3.6.0,
NumPy 2.4.6, Gymnasium 1.3.0, pytest 9.1.1, torch 2.13.0, and SB3 2.9.0.
Version metadata alone does not imply an import works.

- Direct model run: 8 actuators, 13 bodies, timestep 0.002 s, 2,500 steps,
  elapsed simulation time 4.999999999999671 s, final base height
  0.04668060542254152 m.
- Upstream `tests/test_model.py`: **3 passed** (2.61 s).
- Upstream collection: **63 tests collected, 5 collection errors**, plus the
  training module skipped because SB3 could not import `torchgen`.
- No upstream training tests were executed.

The new isolated B.I.M.O environment was installed successfully from `uv.lock`.
Its automated suite passed **13 tests in 7.65 s**, covering the five-second real
model run, invalid durations, fractional steps, repeatability, model-contract
drift, non-finite state, MuJoCo warnings, stalled time, and CLI success/failure.
The installed `bimo-smoke` command reproduced the same five-second baseline and
base height listed above. `uv pip check` reported all 23 installed packages
compatible; torch is absent from this environment. The lock resolves Gymnasium
1.4.0 (the upstream environment has 1.3.0); neither environment's Gym API was
validated, and the smoke runner does not import the missing Sesame environments.
`git diff --check` passed. Rendering and hardware behavior were not tested.

Read-only upstream commands, from B.I.M.O with the documented sibling layout:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ../sesame-ml/.venv/bin/python -m pytest -p no:cacheprovider \
  ../sesame-ml/tests/test_model.py -q
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ../sesame-ml/.venv/bin/python -m pytest -p no:cacheprovider \
  ../sesame-ml/tests --collect-only -q
```

## Blockers and proposed repairs (not implemented)

### Missing Sesame environment implementation

`sesame_ml/envs/__init__.py` imports `SesameEnv`, `SesameEnvConfig`, and `Task`
from `sesame_ml.envs.core`. The directory and Git HEAD tree contain only
`__init__.py`; no tracked `core` implementation exists at the inspected revision.
The import resolves to the expected local source checkout, so this is not merely
an incorrect Python path. The five affected modules are CLI, environments,
evaluation, planning, and simulation-runtime tests.

This is missing Sesame application code, not missing `gymnasium.core`.
Reinstalling Gymnasium or adding an empty shim cannot restore its behavior.
Recommended repair: obtain a complete, maintainer-supported upstream revision
with the environment implementation, review the diff, then update the pin in a
separate approved change. If no such revision exists, a B.I.M.O-owned environment
adapter is a separate feature with observation/action, reward, reset/seed, and
termination contracts. Validate it with upstream behavior tests and Gymnasium's
[environment checker](https://gymnasium.farama.org/main/introduction/create_custom_env/).
Do not guess reward semantics merely to make imports pass.

### Incomplete PyTorch installation

SB3 import reaches `torch.utils._python_dispatch`, which imports `torchgen` and
fails. `find_spec("torchgen")` returned `None`. Installed torch's RECORD lists
132 `torchgen/` files, and **0 of 132 exist**. This is direct evidence of an
incomplete installed distribution. The reason those files disappeared is not
established; it could be an interrupted or altered installation/cache.

After approval, build a separate training environment and install a coherent
PyTorch/SB3 dependency set from a verified source. Choose CPU first unless GPU
requirements are established; use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for the
appropriate build. If repairing the old environment instead, preserve its
dependency manifest and force a fresh torch download/reinstall, bypassing cache.
Do not install an unrelated `torchgen` package or edit site-packages as a patch.
Acceptance: `import torch, torchgen, stable_baselines3`, tensor arithmetic,
dependency consistency, and environment checks all pass. Training still requires
separate approval. No repair was attempted in this audit.

### ROS isolation

All pytest commands unset `PYTHONPATH` and disable automatic plugin discovery.
This reproduced the known baseline without ROS plugin failures. A dedicated
B.I.M.O environment limits dependency coupling; the isolation flags are still
required in ROS-sourced shells.

## Next milestone: visible, scripted locomotion baseline

First add an optional MuJoCo viewer and a bounded scripted gait runner using the
same model contract. Keep rendering optional and headless tests unchanged.
Verify stand pose visually before gait motion; record a short repeatable video
with commands, dependency/model revision, and timestep. Measure displacement,
base height/orientation, falls, and actuator limits. A moving animation alone is
not a locomotion success criterion. This phase needs no policy training and can
proceed independently of the missing Gym environment.

Then, in a separately approved repair milestone, restore the environment and
training stack. Require seeded reset/step tests, correct action and observation
spaces, explicit termination/truncation, Gymnasium checks, and deterministic
scripted-policy evaluation. Agree on task, reward, evaluation seeds, compute
budget, and checkpoint/normalization format before authorizing a small PPO run.
Keep hardware deployment outside both milestones.
