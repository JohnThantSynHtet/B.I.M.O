"""Deterministic scripted gait versus stationary baseline, without Gym or training."""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import platform

import mujoco
import numpy as np
from sesame_ml.constants import CONTROL_DT, JOINT_LIMITS_RAD, JOINT_NAMES
from sesame_ml.model import load_model, make_data, package_root

from bimo.gait import GaitConfig, JointPlayback
from bimo.smoke import _check_health


@dataclass(frozen=True)
class RunConfig:
    seconds: float = 5.0
    settle_seconds: float = 0.5
    fall_height_m: float = 0.025
    fall_tilt_deg: float = 60.0

    def __post_init__(self) -> None:
        for name, low, high in (
            ("seconds", 0.02, 60), ("settle_seconds", 0.1, 5),
            ("fall_height_m", 0.001, 0.045), ("fall_tilt_deg", 1, 89),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def attitude(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """ZYX roll/pitch/yaw and angle between body +z and world +z, in radians."""
    r = rotation.reshape(3, 3)
    return (
        math.atan2(r[2, 1], r[2, 2]), math.asin(float(np.clip(-r[2, 0], -1, 1))),
        math.atan2(r[1, 0], r[0, 0]), math.acos(float(np.clip(r[2, 2], -1, 1))),
    )


def summarize(rows: list[dict], config: RunConfig) -> dict:
    def values(key: str) -> np.ndarray:
        return np.array([row[key] for row in rows])
    xy = np.column_stack((values("x_m"), values("y_m")))
    delta = xy[-1] - xy[0]
    duration = rows[-1]["time_s"]
    yaw = np.unwrap(values("yaw_rad"))
    c, s = math.cos(yaw[0]), math.sin(yaw[0])
    forward, lateral = np.array([[c, s], [-s, c]]) @ delta
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    falls = (values("z_m") < config.fall_height_m) | (
        values("tilt_rad") > math.radians(config.fall_tilt_deg)
    )
    result = {
        "duration_s": duration, "completed": duration + 1e-9 >= config.seconds,
        "displacement_xy_m": delta.tolist(), "net_displacement_m": float(np.linalg.norm(delta)),
        "forward_displacement_m": float(forward), "lateral_displacement_m": float(lateral),
        "path_length_m": path,
        "mean_forward_velocity_m_s": float(forward / duration) if duration else 0.0,
        "mean_path_speed_m_s": path / duration if duration else 0.0,
        "mean_velocity_xy_m_s": (delta / duration).tolist() if duration else [0.0, 0.0],
        "peak_planar_speed_m_s": float((np.linalg.norm(np.diff(xy, axis=0), axis=1)
            / np.diff(values("time_s"))).max(initial=0)),
        "heading_drift_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "max_heading_drift_deg": float(np.degrees(np.abs(yaw - yaw[0]).max())),
        "min_height_m": float(values("z_m").min()),
        "height_std_m": float(values("z_m").std()),
        "max_tilt_deg": float(np.degrees(values("tilt_rad").max())),
        "roll_rms_deg": float(np.degrees(np.sqrt(np.mean(values("roll_rad") ** 2)))),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(values("pitch_rad") ** 2)))),
        "fall_detected": bool(falls.any()),
        "first_fall_time_s": float(values("time_s")[np.flatnonzero(falls)[0]]) if falls.any() else None,
    }
    for name in ("command_violations", "joint_violations", "mechanical_violations", "force_violations", "force_saturations"):
        result[name + "_samples"] = int(values(name)[1:].sum())
    result["max_joint_speed_rad_s"] = float(values("max_joint_speed_rad_s")[1:].max(initial=0))
    result["max_mechanical_excursion_rad"] = float(values("mechanical_excursion_rad").max())
    result["max_abs_torque_Nm"] = float(values("max_abs_torque_Nm").max())
    return result


def run_trial(gait: GaitConfig, config: RunConfig) -> tuple[dict, list[dict]]:
    model = load_model()
    if (model.nu, model.nbody) != (8, 13):
        raise ValueError("Model contract changed: expected 8 actuators, 13 bodies")
    dt = float(model.opt.timestep)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid physics timestep")
    control_steps = round(CONTROL_DT / dt)
    if control_steps < 1 or not math.isclose(control_steps * dt, CONTROL_DT, abs_tol=1e-12):
        raise ValueError("Control period must be an integer number of physics steps")
    controller = JointPlayback(model, gait)
    data = make_data(model)
    data.ctrl[:] = controller.previous
    for _ in range(math.ceil(config.settle_seconds / dt)):
        mujoco.mj_step(model, data)
        _check_health(data)
    start = float(data.time)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    joints = model.actuator_trnid[:, 0]
    qaddr, vaddr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]

    def outside(vector, limits):
        return int(np.count_nonzero((vector < limits[:, 0] - 1e-6) | (vector > limits[:, 1] + 1e-6)))

    def sample(elapsed):
        # mj_step integrates qpos after computing derived quantities. Refresh
        # transforms/forces so every CSV row refers to its stated current time.
        mujoco.mj_forward(model, data)
        _check_health(data)
        roll, pitch, yaw, tilt = attitude(data.xmat[base])
        row = dict(zip(("x_m", "y_m", "z_m"), map(float, data.xpos[base])))
        row.update(time_s=elapsed, roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw, tilt_rad=tilt,
            command_violations=outside(data.ctrl, np.column_stack((controller.low, controller.high))),
            joint_violations=outside(data.qpos[qaddr], model.jnt_range[joints]),
            mechanical_violations=outside(data.qpos[qaddr], JOINT_LIMITS_RAD),
            force_violations=outside(data.actuator_force, model.actuator_forcerange),
            force_saturations=int(np.count_nonzero(np.isclose(
                np.abs(data.actuator_force), model.actuator_forcerange[:, 1], atol=1e-6, rtol=0))),
            max_joint_speed_rad_s=float(np.abs(data.qvel[vaddr]).max()),
            mechanical_excursion_rad=float(np.maximum(
                JOINT_LIMITS_RAD[:, 0] - data.qpos[qaddr],
                data.qpos[qaddr] - JOINT_LIMITS_RAD[:, 1]).max(initial=0)),
            max_abs_torque_Nm=float(np.abs(data.actuator_force).max()))
        row.update({f"qpos_{i}": float(value) for i, value in enumerate(data.qpos)})
        for i, name in enumerate(JOINT_NAMES):
            row[f"target_{name}_rad"] = float(data.ctrl[i])
            row[f"torque_{name}_Nm"] = float(data.actuator_force[i])
        return row

    rows = [sample(0.0)]
    for step in range(math.ceil(config.seconds / dt)):
        previous = rows[-1]
        if previous["z_m"] < config.fall_height_m or previous["tilt_rad"] > math.radians(config.fall_tilt_deg):
            break
        if step % control_steps == 0:
            data.ctrl[:] = controller.next_targets(step * dt)
        mujoco.mj_step(model, data)
        if not math.isclose(data.time - start, (step + 1) * dt, abs_tol=1e-9):
            raise RuntimeError("Simulation time did not advance as expected")
        rows.append(sample((step + 1) * dt))
    result = summarize(rows, config)
    result.update(physics_dt_s=dt, control_dt_s=CONTROL_DT,
        settle_actual_s=start, requested_target_clips=controller.clipped_targets,
        slew_limited_targets=controller.slew_limited_targets,
        stop_reason="fall" if result["fall_detected"] else "duration")
    return result, rows


def provenance() -> dict:
    def digest(root, files):
        result = hashlib.sha256()
        for path in sorted(files):
            result.update(path.relative_to(root).as_posix().encode())
            result.update(b"\0")
            result.update(path.read_bytes())
        return result.hexdigest()
    root = package_root()
    source = Path(__file__).parent
    metadata = distribution("sesame-ml").read_text("direct_url.json")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ("bimo", "sesame-ml", "mujoco", "numpy")},
        "sesame_source": json.loads(metadata) if metadata else None,
        "bimo_source_sha256": digest(source, source.glob("*.py")),
        "sesame_source_assets_sha256": digest(root, (
            p for p in root.rglob("*") if p.suffix in (".py", ".xml", ".stl", ".yaml")
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    for name, default in asdict(GaitConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    for name, default in asdict(RunConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--viewer", action="store_true", help="Replay the measured gait in a 3D viewer")
    parser.add_argument("--render-last", action="store_true", help="Save a 3D image of the final gait state")
    args = parser.parse_args()
    try:
        gait = GaitConfig(**{name: getattr(args, name) for name in asdict(GaitConfig())})
        config = RunConfig(**{name: getattr(args, name) for name in asdict(RunConfig())})
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"schema_version": 1, "gait": asdict(gait), "run": asdict(config), "provenance": provenance(), "trials": {}}
        for label, trial_gait in (("stand", replace(gait, motion="stand")), ("gait", gait)):
            result, rows = run_trial(trial_gait, config)
            report["trials"][label] = result
            with (args.output / f"{label}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        report["comparison"] = {
            "forward_displacement_difference_m": report["trials"]["gait"]["forward_displacement_m"] - report["trials"]["stand"]["forward_displacement_m"],
            "equal_duration": report["trials"]["gait"]["duration_s"] == report["trials"]["stand"]["duration_s"],
        }
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        (args.output / "report.json").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if args.render_last or args.viewer:
            from bimo.visualization import render_last, replay
            if args.render_last:
                render_last(rows, args.output / "gait-final.png")
            if args.viewer:
                replay(rows)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"bimo-gait: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
