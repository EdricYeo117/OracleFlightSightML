import cv2
import numpy as np


POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


def _wrap_angle_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
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

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)

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

        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch_rad = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw_rad = np.arctan2(-rotation_matrix[2, 0], sy)
            roll_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch_rad = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw_rad = np.arctan2(-rotation_matrix[2, 0], sy)
            roll_rad = 0.0

        pitch = float(np.degrees(pitch_rad))
        yaw = float(np.degrees(yaw_rad))
        roll = float(np.degrees(roll_rad))

        pitch = _wrap_angle_deg(pitch)
        yaw = _wrap_angle_deg(yaw)
        roll = _wrap_angle_deg(roll)

        if pitch < -90.0:
            pitch += 180.0
        elif pitch > 90.0:
            pitch -= 180.0

        if yaw < -90.0:
            yaw += 180.0
        elif yaw > 90.0:
            yaw -= 180.0

        if roll < -90.0:
            roll += 180.0
        elif roll > 90.0:
            roll -= 180.0

        pitch = _wrap_angle_deg(pitch)
        yaw = _wrap_angle_deg(yaw)
        roll = _wrap_angle_deg(roll)

        nose_3d = np.array([(0.0, 0.0, 100.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(
            nose_3d, rvec, tvec, camera_matrix, dist_coeffs
        )

        nose_tip = tuple(image_points[0].astype(int))
        projected_end = nose_end_2d.reshape(-1, 2)[0]
        nose_end = tuple((2 * image_points[0] - projected_end).astype(int))

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