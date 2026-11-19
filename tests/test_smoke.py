import json
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from bimo import smoke


def test_five_second_stand():
    result = smoke.run_smoke()
    assert (result.actuators, result.bodies) == (8, 13)
    assert result.steps == 2500
    assert result.simulated_seconds == pytest.approx(5.0)
    assert 0.045 < result.base_height < 0.049


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 61])
def test_invalid_duration(seconds):
    with pytest.raises(ValueError, match="seconds"):
        smoke.run_smoke(seconds)


def test_fractional_step_and_repeatability():
    first = smoke.run_smoke(0.005)
    second = smoke.run_smoke(0.005)
    assert first.steps == 3
    assert first.simulated_seconds == pytest.approx(0.006)
    assert first == second


def test_model_contract_drift(monkeypatch):
    monkeypatch.setattr(smoke, "load_model", lambda: mujoco.MjModel.from_xml_string("<mujoco/>"))
    with pytest.raises(ValueError, match="8 actuators and 13 bodies"):
        smoke.run_smoke()


@pytest.mark.parametrize("failure", ["nonfinite", "warning", "stalled"])
def test_simulation_failure(monkeypatch, failure):
    def broken_step(model, data):
        if failure == "nonfinite":
            data.qvel[0] = np.nan
        elif failure == "warning":
            data.warning[0].number = 1

    monkeypatch.setattr(smoke.mujoco, "mj_step", broken_step)
    with pytest.raises(RuntimeError):
        smoke.run_smoke(0.002)


def test_cli_json():
    run = subprocess.run(
        [sys.executable, "-m", "bimo.smoke", "--seconds", "0.01"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(run.stdout)["steps"] == 5
    assert not run.stderr


def test_cli_rejects_invalid_duration():
    run = subprocess.run(
        [sys.executable, "-m", "bimo.smoke", "--seconds", "nan"],
        capture_output=True, text=True,
    )
    assert run.returncode == 1
    assert "seconds must be finite" in run.stderr
    assert not run.stdout
