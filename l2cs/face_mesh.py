import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh


class FaceMeshDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        points_2d = []

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points_2d.append((x, y))

        return {
            "face_landmarks": face_landmarks,
            "points_2d": np.array(points_2d, dtype=np.float32),
            "image_width": w,
            "image_height": h,
        }