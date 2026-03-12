import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceMeshDetector:
    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ):
        if running_mode == "IMAGE":
            mp_running_mode = vision.RunningMode.IMAGE
        elif running_mode == "LIVE_STREAM":
            mp_running_mode = vision.RunningMode.LIVE_STREAM
        else:
            mp_running_mode = vision.RunningMode.VIDEO

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.running_mode = mp_running_mode

    def process(self, frame_bgr, timestamp_ms: int):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]

        points_2d = []
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points_2d.append((x, y))

        return {
            "face_landmarks": face_landmarks,
            "points_2d": np.array(points_2d, dtype=np.float32),
            "image_width": w,
            "image_height": h,
        }