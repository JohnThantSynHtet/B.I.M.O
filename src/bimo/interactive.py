"""Live scripted walking sandbox, separate from deterministic experiments."""

import argparse
from dataclasses import replace
import math
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np
from sesame_ml.constants import CONTROL_DT
from sesame_ml.model import load_model, make_data

from bimo.gait import GaitConfig, JointPlayback
from bimo.smoke import _check_health
from bimo.visualization import camera


MOTIONS = {"1": "stand", "2": "forward", "3": "backward", "4": "turn_left", "5": "turn_right"}
MIN_SPEED, MAX_SPEED = 0.25, 3.0
HELP = "1 Stand | 2 Forward | 3 Backward | 4 Left | 5 Right\nP Pause/resume | R Reset | -/+ Tempo | C Recenter\nMouse: left-drag rotate | right-drag pan | wheel zoom"


class WalkingSession:
    """Single-threaded physics/control state; no viewer or wall-clock dependency."""

    def __init__(self, motion: str = "forward", speed: float = 1.0):
        self.base_config = GaitConfig(motion=motion)
        if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be in [{MIN_SPEED}, {MAX_SPEED}]")
        self.speed = speed
        self.motion = motion
        self.paused = False
        self.fault: str | None = None
        self.model = load_model()
        if (self.model.nu, self.model.nbody) != (8, 13):
            raise ValueError("Expected 8 actuators and 13 bodies")
        dt = float(self.model.opt.timestep)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid physics timestep")
        self.substeps = round(CONTROL_DT / dt)
        if self.substeps < 1 or not math.isclose(self.substeps * dt, CONTROL_DT, abs_tol=1e-12):
            raise ValueError("Control period must be an integer number of physics steps")
        self.reset()

    def _controller(self) -> JointPlayback:
        config = replace(self.base_config, motion=self.motion,
                         frame_seconds=round(self.base_config.frame_seconds / self.speed, 12))
        return JointPlayback(self.model, config)

    def reset(self) -> None:
        """Restore a settled stand pose; preserve selected motion, tempo and pause."""
        self.data = make_data(self.model)
        self.controller = self._controller()
        self.data.ctrl[:] = self.controller.previous
        for _ in range(round(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            _check_health(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.control_ticks = 0
        self.fault = None

    def handle_key(self, key: int) -> None:
        """Process a GLFW key code on the owner thread, never in the callback."""
        char = chr(key).upper() if 0 <= key < 128 else ""
        if char == "P":
            if self.fault is None:
                self.paused = not self.paused
            return
        if char == "R":
            self.reset()
            return
        motion, speed = self.motion, self.speed
        if char in MOTIONS:
            motion = MOTIONS[char]
        elif char in ("+", "=") or key == 334:  # keypad +
            speed = min(MAX_SPEED, round(speed + 0.25, 2))
        elif char == "-" or key == 333:  # keypad -
            speed = max(MIN_SPEED, round(speed - 0.25, 2))
        if (motion, speed) != (self.motion, self.speed):
            previous = self.controller.previous.copy()
            self.motion, self.speed = motion, speed
            self.controller = self._controller()
            # Reconfiguration restarts phase/ramp, but never jumps the command.
            self.controller.previous = previous
            self.control_ticks = 0

    def tick(self) -> None:
        if self.paused:
            return
        try:
            start = float(self.data.time)
            self.data.ctrl[:] = self.controller.next_targets(self.control_ticks * CONTROL_DT)
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
                _check_health(self.data)
            if not math.isclose(self.data.time - start, CONTROL_DT, abs_tol=1e-9):
                raise RuntimeError("Simulation time did not advance as expected")
            mujoco.mj_forward(self.model, self.data)
            _check_health(self.data)
            self.control_ticks += 1
        except (ValueError, RuntimeError) as exc:
            self.fault = str(exc)
            self.paused = True

    def status(self) -> str:
        state = "PAUSED" if self.paused else "RUNNING"
        result = f"B.I.M.O | {state} | {self.motion} | tempo {self.speed:.2f}x"
        if self.fault:
            result += f"\nSimulation error: {self.fault}\nR resets; P resumes after reset."
        return result


def run_viewer(session: WalkingSession) -> None:
    import mujoco.viewer

    # A separate display model prevents built-in GUI physics sliders/perturbations
    # from bypassing the session's controls. Only camera/visual settings are shared.
    display_model = load_model()
    display_data = make_data(display_model)
    keys: SimpleQueue[int] = SimpleQueue()
    with mujoco.viewer.launch_passive(
        display_model, display_data, key_callback=keys.put,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            initial_camera = camera(display_model, display_data)
            viewer.cam.lookat[:] = initial_camera.lookat
            viewer.cam.distance = initial_camera.distance
            viewer.cam.azimuth = initial_camera.azimuth
            viewer.cam.elevation = initial_camera.elevation
        while viewer.is_running():
            start = time.monotonic()
            recenter = False
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                recenter |= key == ord("C")
                session.handle_key(key)
            session.tick()
            with viewer.lock():
                # MuJoCo also binds some of our keys (e.g. C toggles contacts).
                # Restore scene options so those shortcuts do not hide geometry
                # or add debug overlays. Camera changes remain untouched.
                mujoco.mjv_defaultOption(viewer.opt)
                # Keep the last valid rendered state on a numerical error.
                if np.isfinite(session.data.qpos).all():
                    display_data.qpos[:] = session.data.qpos
                    display_data.qvel[:] = 0
                    display_data.ctrl[:] = session.data.ctrl
                    mujoco.mj_forward(display_model, display_data)
                if recenter:
                    viewer.cam.lookat[:] = display_data.xpos[display_model.body("base").id]
            viewer.set_texts((mujoco.mjtFont.mjFONT_BIG,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, session.status() + "\n" + HELP, ""))
            viewer.sync()
            # No catch-up backlog: slow rendering slows wall-clock playback,
            # while every physics/control step retains its fixed simulated dt.
            time.sleep(max(0, CONTROL_DT - (time.monotonic() - start)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=list(MOTIONS.values()), default="forward")
    parser.add_argument("--speed", type=float, default=1.0, help="Gait tempo multiplier, 0.25 to 3.0")
    args = parser.parse_args()
    try:
        session = WalkingSession(args.motion, args.speed)
        print(HELP, flush=True)
        run_viewer(session)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"bimo-walk: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
