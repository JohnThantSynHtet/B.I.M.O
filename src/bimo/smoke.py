"""Exercise the upstream model without importing Gym environments or ML policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
import sys

import mujoco
import numpy as np
from sesame_ml.model import load_model, make_data


@dataclass(frozen=True)
class SmokeResult:
    actuators: int
    bodies: int
    steps: int
    timestep: float
    requested_seconds: float
    simulated_seconds: float
    base_height: float
    mujoco_version: str
    numpy_version: str
    sesame_ml_version: str


def _check_health(data: mujoco.MjData) -> None:
    for name in ("qpos", "qvel", "qacc", "ctrl", "sensordata"):
        if not np.isfinite(getattr(data, name)).all():
            raise RuntimeError(f"Non-finite simulation state: {name}")
    if any(warning.number for warning in data.warning):
        raise RuntimeError("MuJoCo reported a simulation warning")


def run_smoke(seconds: float = 5.0) -> SmokeResult:
    """Hold Sesame's calibrated stand targets for a bounded headless run.

    Duration is rounded up to whole physics steps. This checks numerical health,
    not locomotion, policy quality, or physical safety.
    """
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("seconds must be finite and in (0, 60]")
    model = load_model()
    if (model.nu, model.nbody) != (8, 13):
        raise ValueError(f"Expected 8 actuators and 13 bodies; got {model.nu}, {model.nbody}")
    timestep = float(model.opt.timestep)
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("Model timestep must be finite and positive")
    steps = math.ceil(seconds / timestep)
    data = make_data(model)
    _check_health(data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        _check_health(data)
    if not math.isclose(data.time, steps * timestep, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("Simulation time did not advance as expected")
    return SmokeResult(
        actuators=model.nu,
        bodies=model.nbody,
        steps=steps,
        timestep=timestep,
        requested_seconds=seconds,
        simulated_seconds=float(data.time),
        base_height=float(data.qpos[2]),
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        sesame_ml_version=version("sesame-ml"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="Simulated duration (0, 60]")
    args = parser.parse_args()
    try:
        result = run_smoke(args.seconds)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"bimo-smoke: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
