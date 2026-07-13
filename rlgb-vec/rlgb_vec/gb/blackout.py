"""
Consecutive black-frame detector.

A frame is "black" if frame.max() < threshold.
After blackout_steps consecutive black frames the episode terminates.
This catches Pokemon Red softlocks and death-then-fade screens.
"""

import numpy as np

from rlgb_vec.config import GBVecConfig


class BlackoutDetector:
    def __init__(self, cfg: GBVecConfig):
        self._threshold = cfg.blackout_threshold
        self._limit     = cfg.blackout_steps
        self._count     = 0

    def reset(self):
        self._count = 0

    def check(self, frame: np.ndarray) -> bool:
        """Return True if this frame triggers a blackout termination."""
        if int(frame.max()) < self._threshold:
            self._count += 1
            return self._count >= self._limit
        self._count = 0
        return False
