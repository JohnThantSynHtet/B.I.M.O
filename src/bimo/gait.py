"""Bounded absolute-joint playback of Sesame's existing firmware sequences."""

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from sesame_ml.constants import (
    CONTROL_DT, JOINT_LIMITS_RAD, SERVO_MAX_SPEED_RAD_S, STAND_ANGLES_RAD,
    validate_joint_vector,
)
from sesame_ml.policies.firmware_sequence import (
    FirmwareMotion, FirmwareSequenceConfig, FirmwareSequencePolicy,
)


@dataclass(frozen=True)
class GaitConfig:
    motion: str = "forward"
    amplitude: float = 0.25
    frame_seconds: float = 0.15
    ramp_seconds: float = 0.5
    max_speed_rad_s: float = 3.0

    def __post_init__(self) -> None:
        if self.motion not in ("stand", "forward", "backward", "turn_left", "turn_right"):
            raise ValueError("Unsupported motion")
        for name, low, high in (
            ("amplitude", 0, 1), ("frame_seconds", 0.05, 2),
            ("ramp_seconds", 0.1, 10), ("max_speed_rad_s", 0.01, SERVO_MAX_SPEED_RAD_S),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


class JointPlayback:
    """Position/rate limits constrain commands, not actual joint velocities."""

    def __init__(self, model: mujoco.MjModel, config: GaitConfig):
        self.config = config
        joints = model.actuator_trnid[:, 0]
        if model.nu != 8 or not np.all(model.actuator_ctrllimited):
            raise ValueError("Expected eight range-limited position actuators")
        self.low = np.maximum.reduce([
            JOINT_LIMITS_RAD[:, 0], model.actuator_ctrlrange[:, 0], model.jnt_range[joints, 0],
        ])
        self.high = np.minimum.reduce([
            JOINT_LIMITS_RAD[:, 1], model.actuator_ctrlrange[:, 1], model.jnt_range[joints, 1],
        ])
        if np.any(self.low > self.high):
            raise ValueError("Empty joint command range")
        self.previous = np.clip(STAND_ANGLES_RAD, self.low, self.high)
        self.clipped_targets = 0
        self.slew_limited_targets = 0
        self.policy = None
        if config.motion != "stand":
            self.policy = FirmwareSequencePolicy(FirmwareSequenceConfig(
                motion=FirmwareMotion(config.motion), control_dt=CONTROL_DT,
                frame_duration_s=config.frame_seconds, interpolate=True,
            ))
            self.policy.reset()

    def limit(self, requested: np.ndarray) -> np.ndarray:
        requested = validate_joint_vector(requested, name="requested targets")
        clipped = np.clip(requested, self.low, self.high)
        self.clipped_targets += int(np.count_nonzero(np.abs(requested - clipped) > 1e-9))
        delta = clipped - self.previous
        bound = self.config.max_speed_rad_s * CONTROL_DT
        self.slew_limited_targets += int(np.count_nonzero(np.abs(delta) > bound + 1e-9))
        self.previous = np.clip(self.previous + np.clip(delta, -bound, bound), self.low, self.high)
        return self.previous.copy()

    def next_targets(self, elapsed: float) -> np.ndarray:
        target = STAND_ANGLES_RAD.copy()
        if self.policy is not None:
            # Fixed motion never reads observations. Ignore Gym residual output;
            # only this public property contains absolute joint angles in radians.
            self.policy.predict({})
            blend = self.config.amplitude * min(1.0, elapsed / self.config.ramp_seconds)
            target += blend * (self.policy.servo_targets_rad - STAND_ANGLES_RAD)
        return self.limit(target)
