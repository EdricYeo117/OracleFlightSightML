import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from calibration_utils import build_feature_vector, FEATURE_COLUMNS
from l2cs import Pipeline, FaceMeshDetector, HeadPoseEstimator


WINDOW_NAME = "Predicted Screen Gaze"


def extract_first_gaze(result):
    if result is None:
        return None, None

    if hasattr(result, "yaw") and hasattr(result, "pitch"):
        yaw = result.yaw
        pitch = result.pitch

        if hasattr(yaw, "__len__") and len(yaw) > 0:
            yaw = yaw[0]
        if hasattr(pitch, "__len__") and len(pitch) > 0:
            pitch = pitch[0]

        try:
            return float(yaw), float(pitch)
        except Exception:
            return None, None

    return None, None


def draw_landmarks(frame, points_2d, color=(0, 255, 0), radius=1):
    for (x, y) in points_2d:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def predict_screen_point(model_x, model_y, features):
    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    pred_x = float(model_x.predict(x)[0])
    pred_y = float(model_y.predict(x)[0])
    return pred_x, pred_y


def clamp_point(x, y, width, height):
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    return x, y


def draw_text(frame, lines, start=(30, 40), dy=35, color=(255, 255, 255)):
    x, y = start
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    gaze_model_path = project_root / "models" / "L2CSNet_gaze360.pkl"
    face_model_path = project_root / "models" / "face_landmarker.task"
    mapper_x_path = project_root / "model_x.pkl"
    mapper_y_path = project_root / "model_y.pkl"

    gaze_pipeline = Pipeline(
        weights=str(gaze_model_path),
        arch="ResNet50",
        device=device,
    )

    face_mesh = FaceMeshDetector(
        model_path=str(face_model_path),
        num_faces=1,
        running_mode="VIDEO",
    )

    head_pose_estimator = HeadPoseEstimator()

    model_x = joblib.load(mapper_x_path)
    model_y = joblib.load(mapper_y_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    temp = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, temp)
    cv2.waitKey(100)

    _, _, screen_w, screen_h = cv2.getWindowImageRect(WINDOW_NAME)
    if screen_w <= 0 or screen_h <= 0:
        screen_w, screen_h = 1920, 1080

    start_time = time.time()

    try:
        while True:
            ok, webcam_frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)

            gaze_result = gaze_pipeline.step(webcam_frame)
            mesh_result = face_mesh.process(webcam_frame, timestamp_ms)

            pose_result = None
            preview = webcam_frame.copy()

            if mesh_result is not None:
                pose_result = head_pose_estimator.estimate(webcam_frame, mesh_result["points_2d"])
                draw_landmarks(preview, mesh_result["points_2d"], radius=1)

            yaw, pitch = extract_first_gaze(gaze_result)

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            preview_h, preview_w = preview.shape[:2]
            inset_w = min(480, screen_w // 3)
            inset_h = int(preview_h * (inset_w / preview_w))
            preview_small = cv2.resize(preview, (inset_w, inset_h))
            canvas[20:20 + inset_h, 20:20 + inset_w] = preview_small

            if mesh_result is not None and pose_result is not None and yaw is not None and pitch is not None:
                features = build_feature_vector(
                    gaze_yaw=yaw,
                    gaze_pitch=pitch,
                    pose_result=pose_result,
                    points_2d=mesh_result["points_2d"],
                )

                pred_x, pred_y = predict_screen_point(model_x, model_y, features)
                pred_x, pred_y = clamp_point(pred_x, pred_y, screen_w, screen_h)

                cv2.circle(canvas, (pred_x, pred_y), 16, (0, 0, 255), -1)
                cv2.circle(canvas, (pred_x, pred_y), 24, (255, 255, 255), 2)

                draw_text(
                    canvas,
                    [
                        f"Predicted X: {pred_x}",
                        f"Predicted Y: {pred_y}",
                        f"Gaze Yaw: {np.degrees(yaw):.2f} deg",
                        f"Gaze Pitch: {np.degrees(pitch):.2f} deg",
                    ],
                    start=(30, inset_h + 70),
                )
            else:
                draw_text(
                    canvas,
                    [
                        "No valid prediction yet",
                        "Make sure face, gaze, and head pose are detected",
                    ],
                    start=(30, inset_h + 70),
                )

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting cleanly...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()