import time
from collections import deque

import numpy as np


class TemporalGazeTracker:
    def __init__(
        self,
        iris_alpha: float = 0.55,
        l2cs_alpha: float = 0.75,
        head_alpha: float = 0.7,
        final_alpha: float = 0.6,
        history_size: int = 7,
    ):
        self.iris_alpha = iris_alpha
        self.l2cs_alpha = l2cs_alpha
        self.head_alpha = head_alpha
        self.final_alpha = final_alpha

        self.iris_vec = None
        self.l2cs_vec = None
        self.head_pose = None
        self.final_vec = None

        self.history = deque(maxlen=history_size)
        self.frame_idx = 0
        self.start_time = time.time()

    @staticmethod
    def _ema(prev, cur, alpha):
        if cur is None:
            return prev
        if prev is None:
            return cur
        return (
            alpha * prev[0] + (1.0 - alpha) * cur[0],
            alpha * prev[1] + (1.0 - alpha) * cur[1],
        )

    @staticmethod
    def _ema_head(prev, cur, alpha):
        if cur is None:
            return prev
        if prev is None:
            return cur
        return {
            "yaw": alpha * prev["yaw"] + (1.0 - alpha) * cur["yaw"],
            "pitch": alpha * prev["pitch"] + (1.0 - alpha) * cur["pitch"],
            "roll": alpha * prev["roll"] + (1.0 - alpha) * cur["roll"],
        }

    @staticmethod
    def _mag(vec):
        if vec is None:
            return 0.0
        return float(np.hypot(vec[0], vec[1]))

    def update(self, iris_vec, l2cs_vec, head_pose, blink_like=False):
        self.frame_idx += 1

        self.iris_vec = self._ema(self.iris_vec, iris_vec, self.iris_alpha)
        self.l2cs_vec = self._ema(self.l2cs_vec, l2cs_vec, self.l2cs_alpha)
        self.head_pose = self._ema_head(self.head_pose, head_pose, self.head_alpha)

        fused = None
        if self.iris_vec is not None and self.l2cs_vec is not None:
            iris_strength = min(1.0, self._mag(self.iris_vec))
            head_strength = 0.0
            if self.head_pose is not None:
                head_strength = min(
                    1.0,
                    max(abs(self.head_pose["yaw"]) / 20.0, abs(self.head_pose["pitch"]) / 20.0),
                )

            iris_weight = 0.72 + 0.18 * iris_strength + 0.06 * head_strength
            iris_weight = float(np.clip(iris_weight, 0.70, 0.95))
            l2cs_weight = 1.0 - iris_weight

            if blink_like:
                iris_weight = 0.55
                l2cs_weight = 0.45

            fused = (
                iris_weight * self.iris_vec[0] + l2cs_weight * self.l2cs_vec[0],
                iris_weight * self.iris_vec[1] + l2cs_weight * self.l2cs_vec[1],
            )
        elif self.iris_vec is not None:
            fused = self.iris_vec
        elif self.l2cs_vec is not None:
            fused = self.l2cs_vec

        self.final_vec = self._ema(self.final_vec, fused, self.final_alpha)

        snapshot = {
            "frame": self.frame_idx,
            "t": time.time() - self.start_time,
            "iris_vec": self.iris_vec,
            "l2cs_vec": self.l2cs_vec,
            "head_pose": self.head_pose,
            "final_vec": self.final_vec,
            "blink_like": blink_like,
        }
        self.history.append(snapshot)
        return snapshot