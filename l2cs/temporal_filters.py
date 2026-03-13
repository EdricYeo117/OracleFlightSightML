import numpy as np


class TemporalVectorFilter:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.value = None

    def reset(self):
        self.value = None

    def update(self, vec):
        if vec is None:
            return self.value

        vec = np.asarray(vec, dtype=np.float32)

        if self.value is None:
            self.value = vec
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * vec

        return self.value


class TemporalHeadPoseFilter:
    def __init__(self, alpha=0.7):
        self.alpha = float(alpha)
        self.value = None

    def reset(self):
        self.value = None

    def update(self, pose):
        if pose is None:
            return self.value

        pose = {
            "yaw": float(pose["yaw"]),
            "pitch": float(pose["pitch"]),
            "roll": float(pose["roll"]),
        }

        if self.value is None:
            self.value = pose
        else:
            self.value = {
                "yaw": self.alpha * self.value["yaw"] + (1.0 - self.alpha) * pose["yaw"],
                "pitch": self.alpha * self.value["pitch"] + (1.0 - self.alpha) * pose["pitch"],
                "roll": self.alpha * self.value["roll"] + (1.0 - self.alpha) * pose["roll"],
            }

        return self.value


class TemporalGazeTracker:
    def __init__(
        self,
        iris_alpha=0.20,
        l2cs_alpha=0.35,
        head_alpha=0.50,
        final_alpha=0.10,
    ):
        self.iris_filter = TemporalVectorFilter(alpha=iris_alpha)
        self.l2cs_filter = TemporalVectorFilter(alpha=l2cs_alpha)
        self.final_filter = TemporalVectorFilter(alpha=final_alpha)
        self.head_filter = TemporalHeadPoseFilter(alpha=head_alpha)

        self.frame_idx = 0

    def reset(self):
        self.iris_filter.reset()
        self.l2cs_filter.reset()
        self.final_filter.reset()
        self.head_filter.reset()
        self.frame_idx = 0

    @staticmethod
    def _mag(vec):
        if vec is None:
            return 0.0
        return float(np.hypot(vec[0], vec[1]))

    def update(
        self,
        iris_vec,
        l2cs_vec,
        head_pose=None,
        blink_like=False,
        degrade_primary=False,
        primary_confidence=1.0,
    ):
        self.frame_idx += 1

        iris_vec_s = self.iris_filter.update(iris_vec)
        l2cs_vec_s = self.l2cs_filter.update(l2cs_vec)
        head_pose_s = self.head_filter.update(head_pose)

        final_raw = None
        iris_weight = None
        l2cs_weight = None

        if iris_vec_s is not None and l2cs_vec_s is not None:
            iris_strength = min(1.0, self._mag(iris_vec_s))
            head_yaw_abs = abs(head_pose_s["yaw"]) if head_pose_s is not None else 0.0

            iris_weight = 0.72 + 0.10 * iris_strength
            iris_weight = float(np.clip(iris_weight, 0.68, 0.82))

            if degrade_primary:
                if head_yaw_abs > 35.0:
                    iris_weight = min(iris_weight, 0.68)
                elif head_yaw_abs > 25.0:
                    iris_weight = min(iris_weight, 0.72)

                if primary_confidence < 0.995:
                    iris_weight = min(iris_weight, 0.70)

            if blink_like:
                iris_weight = min(iris_weight, 0.35)

            l2cs_weight = 1.0 - iris_weight

            final_raw = np.array(
                [
                    iris_weight * iris_vec_s[0] + l2cs_weight * l2cs_vec_s[0],
                    iris_weight * iris_vec_s[1] + l2cs_weight * l2cs_vec_s[1],
                ],
                dtype=np.float32,
            )

        elif iris_vec_s is not None:
            final_raw = iris_vec_s
            iris_weight = 1.0
            l2cs_weight = 0.0

        elif l2cs_vec_s is not None:
            final_raw = l2cs_vec_s
            iris_weight = 0.0
            l2cs_weight = 1.0

        final_vec_s = self.final_filter.update(final_raw)

        return {
            "frame": self.frame_idx,
            "iris_vec": iris_vec_s,
            "l2cs_vec": l2cs_vec_s,
            "final_vec": final_vec_s,
            "head_pose": head_pose_s,
            "iris_weight": iris_weight,
            "l2cs_weight": l2cs_weight,
            "blink_like": bool(blink_like),
            "degrade_primary": bool(degrade_primary),
            "primary_confidence": float(primary_confidence),
        }