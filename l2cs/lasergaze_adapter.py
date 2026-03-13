import numpy as np

from .AffineTransformer import AffineTransformer
from .EyeballDetector import EyeballDetector
from .face_model import (
    BASE_FACE_MODEL,
    DEFAULT_LEFT_EYE_CENTER_MODEL,
    DEFAULT_RIGHT_EYE_CENTER_MODEL,
)
from .landmarks import (
    OUTER_HEAD_POINTS,
    NOSE_BRIDGE,
    NOSE_TIP,
    LEFT_IRIS,
    LEFT_PUPIL,
    RIGHT_IRIS,
    RIGHT_PUPIL,
    ADJACENT_LEFT_EYELID_PART,
    ADJACENT_RIGHT_EYELID_PART,
    BASE_LANDMARKS,
)
from .temporal_filters import TemporalVectorFilter


class LaserGazeAdapter:
    def __init__(
        self,
        avg_gain_x=2.5,
        avg_gain_y=2.0,
        invert_x=False,
        invert_y=True,
        left_alpha=0.60,
        right_alpha=0.60,
        avg_alpha=0.55,
    ):
        self.left_detector = EyeballDetector(DEFAULT_LEFT_EYE_CENTER_MODEL)
        self.right_detector = EyeballDetector(DEFAULT_RIGHT_EYE_CENTER_MODEL)

        self.left_filter = TemporalVectorFilter(alpha=left_alpha)
        self.right_filter = TemporalVectorFilter(alpha=right_alpha)
        self.avg_filter = TemporalVectorFilter(alpha=avg_alpha)

        self.avg_gain_x = avg_gain_x
        self.avg_gain_y = avg_gain_y
        self.invert_x = invert_x
        self.invert_y = invert_y

    def reset(self):
        self.left_detector.reset()
        self.right_detector.reset()
        self.left_filter.reset()
        self.right_filter.reset()
        self.avg_filter.reset()

    @staticmethod
    def _normalize_gaze_vec_2d(gaze_vec_3d, gain_x=2.5, gain_y=2.0, invert_x=False, invert_y=False):
        if gaze_vec_3d is None:
            return None

        v = np.asarray(gaze_vec_3d, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return None

        v = v / norm

        dx = v[0] * gain_x
        dy = v[1] * gain_y

        if invert_x:
            dx = -dx
        if invert_y:
            dy = -dy

        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))
        return np.array([dx, dy], dtype=np.float32)

    def process(self, mesh_result, timestamp_ms):
        result = {
            "points_2d": None,
            "points_3d": None,
            "left_gaze_vec": None,
            "right_gaze_vec": None,
            "left_vec_2d": None,
            "right_vec_2d": None,
            "avg_raw_3d": None,
            "avg_vec_2d": None,
            "left_detector": self.left_detector,
            "right_detector": self.right_detector,
        }

        if mesh_result is None or "points_2d" not in mesh_result:
            return result

        result["points_2d"] = mesh_result["points_2d"]

        if "face_landmarks" not in mesh_result or mesh_result["face_landmarks"] is None:
            return result

        raw_lms = mesh_result["face_landmarks"]
        points_3d = np.array([[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32)
        result["points_3d"] = points_3d

        if len(points_3d) < 478:
            return result

        mp_hor_pts = [points_3d[i] for i in OUTER_HEAD_POINTS]
        mp_ver_pts = [points_3d[i] for i in [NOSE_BRIDGE, NOSE_TIP]]

        at = AffineTransformer(
            points_3d[BASE_LANDMARKS, :],
            BASE_FACE_MODEL,
            mp_hor_pts,
            mp_ver_pts,
            [BASE_FACE_MODEL[4], BASE_FACE_MODEL[5]],
            [BASE_FACE_MODEL[6], BASE_FACE_MODEL[7]],
        )

        if not at.success:
            return result

        left_indices = LEFT_IRIS + ADJACENT_LEFT_EYELID_PART
        right_indices = RIGHT_IRIS + ADJACENT_RIGHT_EYELID_PART

        left_eye_points_model = np.array(
            [at.to_m2(points_3d[i]) for i in left_indices if at.to_m2(points_3d[i]) is not None],
            dtype=np.float32,
        )
        right_eye_points_model = np.array(
            [at.to_m2(points_3d[i]) for i in right_indices if at.to_m2(points_3d[i]) is not None],
            dtype=np.float32,
        )

        if len(left_eye_points_model) > 0:
            self.left_detector.update(left_eye_points_model, timestamp_ms)
        if len(right_eye_points_model) > 0:
            self.right_detector.update(right_eye_points_model, timestamp_ms)

        weighted_vecs = []
        weighted_conf = []

        if self.left_detector.center_detected:
            left_eyeball_center = at.to_m1(self.left_detector.eye_center)
            left_pupil = points_3d[LEFT_PUPIL]
            left_gaze_vec = left_pupil - left_eyeball_center
            result["left_gaze_vec"] = left_gaze_vec

            left_vec_2d = self._normalize_gaze_vec_2d(
                left_gaze_vec,
                gain_x=self.avg_gain_x,
                gain_y=self.avg_gain_y,
                invert_x=self.invert_x,
                invert_y=self.invert_y,
            )
            result["left_vec_2d"] = self.left_filter.update(left_vec_2d)

            weighted_vecs.append(left_gaze_vec)
            weighted_conf.append(max(1e-6, float(self.left_detector.current_confidence)))

        if self.right_detector.center_detected:
            right_eyeball_center = at.to_m1(self.right_detector.eye_center)
            right_pupil = points_3d[RIGHT_PUPIL]
            right_gaze_vec = right_pupil - right_eyeball_center
            result["right_gaze_vec"] = right_gaze_vec

            right_vec_2d = self._normalize_gaze_vec_2d(
                right_gaze_vec,
                gain_x=self.avg_gain_x,
                gain_y=self.avg_gain_y,
                invert_x=self.invert_x,
                invert_y=self.invert_y,
            )
            result["right_vec_2d"] = self.right_filter.update(right_vec_2d)

            weighted_vecs.append(right_gaze_vec)
            weighted_conf.append(max(1e-6, float(self.right_detector.current_confidence)))

        if weighted_vecs:
            weights = np.asarray(weighted_conf, dtype=np.float32)
            weights = weights / np.sum(weights)

            avg_raw_3d = np.sum(
                np.stack(weighted_vecs, axis=0) * weights[:, None],
                axis=0,
            )
            result["avg_raw_3d"] = avg_raw_3d

            avg_vec_2d = self._normalize_gaze_vec_2d(
                avg_raw_3d,
                gain_x=self.avg_gain_x,
                gain_y=self.avg_gain_y,
                invert_x=self.invert_x,
                invert_y=self.invert_y,
            )
            result["avg_vec_2d"] = self.avg_filter.update(avg_vec_2d)

        return result