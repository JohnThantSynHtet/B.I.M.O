import json
import math
import subprocess
import sys

import mujoco
import numpy as np
import pytest
from sesame_ml.constants import CONTROL_DT, STAND_ANGLES_RAD
from sesame_ml.model import load_model

from bimo import experiment
from bimo.experiment import RunConfig, attitude, run_trial, summarize
from bimo.gait import GaitConfig, JointPlayback
from bimo.visualization import set_frame


@pytest.mark.parametrize("field,value", [
    ("amplitude", float("nan")), ("amplitude", -0.1), ("amplitude", 1.1),
    ("frame_seconds", 0), ("ramp_seconds", 0), ("max_speed_rad_s", 20),
    ("motion", "auto"),
])
def test_invalid_gait(field, value):
    with pytest.raises(ValueError):
        GaitConfig(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("seconds", 0), ("seconds", float("inf")), ("settle_seconds", -1),
    ("fall_height_m", 0.05), ("fall_tilt_deg", 90),
])
def test_invalid_run(field, value):
    with pytest.raises(ValueError):
        RunConfig(**{field: value})


def test_clamping_slew_and_invalid_targets():
    controller = JointPlayback(load_model(), GaitConfig())
    previous = controller.previous.copy()
    target = controller.limit(np.full(8, 100.0))
    assert np.all(target >= controller.low)
    assert np.all(target <= controller.high)
    assert np.max(np.abs(target - previous)) <= 3 * CONTROL_DT + 1e-12
    assert controller.clipped_targets == 8
    assert controller.slew_limited_targets > 0
    with pytest.raises(ValueError):
        controller.limit(np.full(8, np.nan))
    with pytest.raises(ValueError):
        controller.limit(np.zeros(7))


def test_firmware_absolute_targets_and_ramp():
    controller = JointPlayback(load_model(), GaitConfig(amplitude=1))
    assert controller.next_targets(0) == pytest.approx(STAND_ANGLES_RAD, abs=1e-9)
    target = controller.next_targets(CONTROL_DT)
    assert np.max(np.abs(target - STAND_ANGLES_RAD)) > 0
    assert np.max(np.abs(target - STAND_ANGLES_RAD)) <= 3 * CONTROL_DT + 1e-9


def test_real_model_repeatability_and_stationary_equivalence():
    config = RunConfig(seconds=0.3, settle_seconds=0.1)
    first = run_trial(GaitConfig(), config)
    assert run_trial(GaitConfig(), config) == first
    stand, rows = run_trial(GaitConfig(motion="stand"), config)
    assert run_trial(GaitConfig(amplitude=0), config) == (stand, rows)
    assert stand["completed"]
    assert not stand["fall_detected"]
    assert stand["command_violations_samples"] == 0
    assert stand["force_violations_samples"] == 0
    assert first[0]["forward_displacement_m"] != stand["forward_displacement_m"]
    assert len(rows) == 151


def test_metrics_known_trajectory_and_heading_wrap():
    _, original = run_trial(GaitConfig(motion="stand"), RunConfig(seconds=0.02, settle_seconds=0.1))
    rows = [dict(original[0]), dict(original[-1])]
    for i, row in enumerate(rows):
        row.update(time_s=float(i), x_m=float(i), y_m=0.0, z_m=0.04,
            yaw_rad=math.radians(179 if i == 0 else -179),
            roll_rad=0.0, pitch_rad=0.0, tilt_rad=0.0)
    result = summarize(rows, RunConfig(seconds=1))
    assert result["net_displacement_m"] == 1
    assert result["mean_path_speed_m_s"] == 1
    assert result["forward_displacement_m"] == pytest.approx(math.cos(math.radians(179)))
    assert result["heading_drift_deg"] == pytest.approx(2)
    assert not result["fall_detected"]
    rows[-1]["z_m"] = 0.01
    result = summarize(rows, RunConfig(seconds=1))
    assert result["first_fall_time_s"] == 1


def test_attitude_known_rotation():
    matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert attitude(matrix) == pytest.approx((0, 0, math.pi / 2, 0))


def test_fall_stops_before_more_commands(monkeypatch):
    monkeypatch.setattr(experiment, "attitude", lambda r: (0, 0, 0, math.radians(70)))
    result, rows = run_trial(GaitConfig(), RunConfig(seconds=0.02, settle_seconds=0.1))
    assert len(rows) == 1
    assert result["stop_reason"] == "fall"
    assert result["first_fall_time_s"] == 0
    assert not result["completed"]


def test_stalled_physics_fails(monkeypatch):
    monkeypatch.setattr(experiment.mujoco, "mj_step", lambda m, d: None)
    with pytest.raises(RuntimeError, match="time"):
        run_trial(GaitConfig(), RunConfig(seconds=0.02, settle_seconds=0.1))


def test_visualization_reconstructs_recorded_pose():
    _, rows = run_trial(GaitConfig(), RunConfig(seconds=0.04, settle_seconds=0.1))
    model = load_model()
    data = mujoco.MjData(model)
    set_frame(model, data, rows[-1])
    assert data.xpos[model.body("base").id] == pytest.approx([
        rows[-1][key] for key in ("x_m", "y_m", "z_m")
    ])


def test_cli_artifacts_repeatability_and_no_overwrite(tmp_path):
    base = [sys.executable, "-m", "bimo.experiment", "--seconds", "0.04", "--settle-seconds", "0.1"]
    for name in ("one", "two"):
        result = subprocess.run(base + ["--output", str(tmp_path / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["comparison"]["equal_duration"]
        assert report["provenance"]["sesame_source"]["vcs_info"]["commit_id"]
    for name in ("report.json", "stand.csv", "gait.csv"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    result = subprocess.run(base + ["--output", str(tmp_path / "one")], capture_output=True, text=True)
    assert result.returncode != 0


def test_no_gym_environment_or_training_imports():
    result = subprocess.run([sys.executable, "-c",
        "import bimo.experiment, sys; "
        "assert 'sesame_ml.envs' not in sys.modules; "
        "assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules; "
        "assert 'mujoco.viewer' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
