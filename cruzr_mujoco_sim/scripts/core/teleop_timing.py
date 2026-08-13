"""Timing helpers for the MuJoCo teleop render/control loop."""

import time


def control_substeps_for_fps(fps, timestep):
    """Return the nearest integer physics substeps for one rendered frame."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    return max(1, round(1.0 / (float(fps) * float(timestep))))


class CumulativeSubstepScheduler:
    """Alternate integer physics steps without accumulating control-clock drift."""

    def __init__(self, fps, timestep):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        self.steps_per_tick = 1.0 / (float(fps) * float(timestep))
        if self.steps_per_tick < 1.0:
            raise ValueError("target control rate exceeds the physics step rate")
        self.ticks = 0
        self.physics_steps = 0

    def next_substeps(self):
        self.ticks += 1
        cumulative_target = round(self.ticks * self.steps_per_tick)
        substeps = cumulative_target - self.physics_steps
        if substeps < 1:
            raise RuntimeError("substep scheduler produced a non-positive step count")
        self.physics_steps = cumulative_target
        return substeps


class FramePacer:
    """Sleep between frames to keep the interactive loop near a target FPS."""

    def __init__(self, fps, perf_counter=None, sleep=None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period = 1.0 / float(fps)
        self._perf_counter = perf_counter or time.perf_counter
        self._sleep = sleep or time.sleep
        self._deadline = self._perf_counter() + self.period

    def sleep_until_next_frame(self):
        now = self._perf_counter()
        remaining = self._deadline - now
        if remaining > 0:
            self._sleep(remaining)
            now = self._perf_counter()
        if now > self._deadline + self.period:
            self._deadline = now + self.period
        else:
            self._deadline += self.period
