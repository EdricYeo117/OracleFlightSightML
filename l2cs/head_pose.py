import cv2
import numpy as np
from math import atan2, sqrt, pi


# MediaPipe Face Mesh landmark indices commonly used for head pose
# nose tip, chin, left eye outer corner, right eye outer corner,
# left mouth corner, right mouth corner
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


class HeadPoseEstimator:
    def __init__(self):
        # Generic 3D face model points
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -63.6, -12.5),      # chin
                (-43.3, 32.7, -26.0),     # left eye outer
                (43.3, 32.7, -26.0),      # right eye outer
                (-28.9, -28.9, -24.1),    # left mouth
                (28.9, -28.9, -24.1),     # right mouth
            ],
            dtype=np.float64,
        )

    def _get_image_points(self, points_2d):
        return np.array(
            [
                points_2d[POSE_LANDMARKS["nose_tip"]],
                points_2d[POSE_LANDMARKS["chin"]],
                points_2d[POSE_LANDMARKS["left_eye_outer"]],
                points_2d[POSE_LANDMARKS["right_eye_outer"]],
                points_2d[POSE_LANDMARKS["left_mouth"]],
                points_2d[POSE_LANDMARKS["right_mouth"]],
            ],
            dtype=np.float64,
        )

    def estimate(self, frame_bgr, points_2d):
        h, w = frame_bgr.shape[:2]

        image_points = self._get_image_points(points_2d)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        pose_matrix = cv2.hconcat((rotation_matrix, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        nose_3d = np.array([(0.0, 0.0, 100.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(
            nose_3d, rvec, tvec, camera_matrix, dist_coeffs
        )

        nose_tip = tuple(image_points[0].astype(int))
        nose_end = tuple(nose_end_2d.reshape(-1, 2)[0].astype(int))

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "nose_tip": nose_tip,
            "nose_end": nose_end,
            "image_points": image_points,
        }