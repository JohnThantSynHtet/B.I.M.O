import numpy as np
import pytest

from bimo.interactive import MAX_SPEED, MIN_SPEED, WalkingSession


def test_continuous_stepping_and_pause_resume():
    session = WalkingSession()
    start = session.data.time
    for _ in range(260):
        session.tick()
    assert session.data.time == pytest.approx(start + 5.2)
    assert session.fault is None
    session.handle_key(ord("P"))
    state, ctrl, time = session.data.qpos.copy(), session.data.ctrl.copy(), session.data.time
    for _ in range(5):
        session.tick()
    np.testing.assert_array_equal(session.data.qpos, state)
    np.testing.assert_array_equal(session.data.ctrl, ctrl)
    assert session.data.time == time
    session.handle_key(ord("P"))
    session.tick()
    assert session.data.time == pytest.approx(time + 0.02)


@pytest.mark.parametrize("key,motion", [("1", "stand"), ("2", "forward"), ("3", "backward"), ("4", "turn_left"), ("5", "turn_right")])
def test_motion_switch_preserves_slew_limit(key, motion):
    session = WalkingSession()
    for _ in range(40):
        session.tick()
    before = session.data.ctrl.copy()
    session.handle_key(ord(key))
    assert session.motion == motion
    session.tick()
    assert np.max(np.abs(session.data.ctrl - before)) <= 3 * 0.02 + 1e-9
    assert session.fault is None


def test_reset_is_repeatable_and_preserves_selection():
    session = WalkingSession("backward", 1.5)
    initial = session.data.qpos.copy()
    for _ in range(10):
        session.tick()
    expected = session.data.qpos.copy()
    session.handle_key(ord("P"))
    session.handle_key(ord("R"))
    assert session.paused and session.motion == "backward" and session.speed == 1.5
    np.testing.assert_array_equal(session.data.qpos, initial)
    session.handle_key(ord("P"))
    for _ in range(10):
        session.tick()
    np.testing.assert_array_equal(session.data.qpos, expected)


def test_tempo_bounds_and_physics_timestep():
    session = WalkingSession()
    for _ in range(30):
        session.handle_key(ord("+"))
    assert session.speed == MAX_SPEED
    assert session.controller.config.frame_seconds == pytest.approx(0.05)
    for _ in range(30):
        session.handle_key(ord("-"))
    assert session.speed == MIN_SPEED
    assert session.controller.config.frame_seconds == pytest.approx(0.6)
    assert session.model.opt.timestep == 0.002
    session.tick()
    assert session.fault is None


@pytest.mark.parametrize("speed", [0, 4, float("nan"), float("inf")])
def test_invalid_speed(speed):
    with pytest.raises(ValueError):
        WalkingSession(speed=speed)


def test_numerical_fault_pauses_until_reset(monkeypatch):
    session = WalkingSession()
    with monkeypatch.context() as patch:
        patch.setattr(session.controller, "next_targets", lambda t: (_ for _ in ()).throw(RuntimeError("test fault")))
        session.tick()
    assert session.paused and session.fault == "test fault"
    session.handle_key(ord("P"))
    assert session.paused
    session.handle_key(ord("R"))
    assert session.fault is None
    session.handle_key(ord("P"))
    session.tick()
    assert not session.paused


def test_viewer_loop_processes_keys_and_stays_open_when_paused(monkeypatch):
    from contextlib import nullcontext
    import mujoco.viewer
    from bimo import interactive

    session = WalkingSession()

    class Viewer:
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        frames = 0
        texts = []
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def lock(self):
            return nullcontext()

        def is_running(self):
            commands = ["P", "3=", "RP", "1C"]
            if self.frames == len(commands):
                return False
            for key in commands[self.frames]:
                self.callback(ord(key))
            return True

        def sync(self):
            self.frames += 1

        def set_texts(self, text):
            assert text[0] == mujoco.mjtFont.mjFONT_BIG
            self.texts.append(text[2])

    viewer = Viewer()

    def launch(model, data, *, key_callback, **kwargs):
        assert model is not session.model and data is not session.data
        viewer.callback = key_callback
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(interactive.time, "sleep", lambda seconds: None)
    interactive.run_viewer(session)
    assert viewer.closed and viewer.frames == 4
    assert "PAUSED" in viewer.texts[0]
    assert "backward" in viewer.texts[1] and "1.25x" in viewer.texts[1]
    assert "RUNNING" in viewer.texts[-1] and session.motion == "stand"
    assert session.data.time == pytest.approx(0.54)
