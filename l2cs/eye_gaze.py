import numpy as np


LEFT_EYE = {
    "outer_corner": 33,
    "inner_corner": 133,
    "upper_lid": 159,
    "lower_lid": 145,
    "iris": [468, 469, 470, 471],
}

RIGHT_EYE = {
    "outer_corner": 263,
    "inner_corner": 362,
    "upper_lid": 386,
    "lower_lid": 374,
    "iris": [473, 474, 475, 476],
}


def _safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def _unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return v / _safe_norm(v, eps)


def _project_scalar(vec: np.ndarray, axis_unit: np.ndarray) -> float:
    return float(np.dot(vec, axis_unit))


class EyeGazeEstimator:
    def __init__(
        self,
        center_tolerance_x: float = 0.08,
        center_tolerance_y: float = 0.10,
        smoothing: float = 0.35,
    ):
        self.center_tolerance_x = float(center_tolerance_x)
        self.center_tolerance_y = float(center_tolerance_y)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._last_eye_dx = None
        self._last_eye_dy = None

    def _iris_center(self, points_2d: np.ndarray, iris_indices):
        iris_pts = points_2d[iris_indices].astype(np.float32)
        return iris_pts.mean(axis=0)

    def _eye_metrics(self, points_2d: np.ndarray, eye_cfg):
        outer_corner = points_2d[eye_cfg["outer_corner"]].astype(np.float32)
        inner_corner = points_2d[eye_cfg["inner_corner"]].astype(np.float32)
        upper_lid = points_2d[eye_cfg["upper_lid"]].astype(np.float32)
        lower_lid = points_2d[eye_cfg["lower_lid"]].astype(np.float32)
        iris_center = self._iris_center(points_2d, eye_cfg["iris"])

        horizontal_vec = inner_corner - outer_corner
        vertical_vec = lower_lid - upper_lid

        eye_width = _safe_norm(horizontal_vec)
        eye_height = _safe_norm(vertical_vec)

        if eye_width < 1.0 or eye_height < 1.0:
            return None

        x_axis = _unit(horizontal_vec)
        y_axis = _unit(vertical_vec)

        iris_from_outer = iris_center - outer_corner
        iris_from_upper = iris_center - upper_lid

        iris_x = _project_scalar(iris_from_outer, x_axis)
        iris_y = _project_scalar(iris_from_upper, y_axis)

        iris_x_norm = iris_x / eye_width
        iris_y_norm = iris_y / eye_height

        eye_openness = eye_height / eye_width

        return {
            "outer_corner": outer_corner,
            "inner_corner": inner_corner,
            "upper_lid": upper_lid,
            "lower_lid": lower_lid,
            "iris_center": iris_center,
            "eye_width": float(eye_width),
            "eye_height": float(eye_height),
            "iris_x_norm": float(iris_x_norm),
            "iris_y_norm": float(iris_y_norm),
            "eye_openness": float(eye_openness),
        }

    def _smooth(self, dx: float, dy: float):
        if self._last_eye_dx is None or self._last_eye_dy is None:
            self._last_eye_dx = dx
            self._last_eye_dy = dy
            return dx, dy

        a = self.smoothing
        smoothed_dx = a * self._last_eye_dx + (1.0 - a) * dx
        smoothed_dy = a * self._last_eye_dy + (1.0 - a) * dy

        self._last_eye_dx = smoothed_dx
        self._last_eye_dy = smoothed_dy
        return smoothed_dx, smoothed_dy

    def _classify_direction(self, eye_dx: float, eye_dy: float) -> str:
        x_left = eye_dx <= -self.center_tolerance_x
        x_right = eye_dx >= self.center_tolerance_x
        y_up = eye_dy <= -self.center_tolerance_y
        y_down = eye_dy >= self.center_tolerance_y

        if not (x_left or x_right or y_up or y_down):
            return "center"
        if x_left and y_up:
            return "up-left"
        if x_right and y_up:
            return "up-right"
        if x_left and y_down:
            return "down-left"
        if x_right and y_down:
            return "down-right"
        if x_left:
            return "left"
        if x_right:
            return "right"
        if y_up:
            return "up"
        return "down"

    def estimate(self, points_2d: np.ndarray):
        if points_2d is None or len(points_2d) < 477:
            return None

        left = self._eye_metrics(points_2d, LEFT_EYE)
        right = self._eye_metrics(points_2d, RIGHT_EYE)

        if left is None or right is None:
            return None

        eye_x = (left["iris_x_norm"] + right["iris_x_norm"]) / 2.0
        eye_y = (left["iris_y_norm"] + right["iris_y_norm"]) / 2.0

        eye_dx = eye_x - 0.5
        eye_dy = eye_y - 0.5

        eye_dx, eye_dy = self._smooth(eye_dx, eye_dy)
        direction = self._classify_direction(eye_dx, eye_dy)

        left_open = left["eye_openness"]
        right_open = right["eye_openness"]
        blink_like = left_open < 0.10 or right_open < 0.10

        confidence = 1.0
        if blink_like:
            confidence *= 0.5

        return {
            "left_eye": left,
            "right_eye": right,
            "eye_x": float(eye_x),
            "eye_y": float(eye_y),
            "eye_dx": float(eye_dx),
            "eye_dy": float(eye_dy),
            "direction": direction,
            "confidence": float(confidence),
            "blink_like": bool(blink_like),
        }