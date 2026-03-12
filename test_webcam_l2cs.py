import time
from pathlib import Path

import cv2
import numpy as np
import torch

from l2cs import Pipeline, FaceMeshDetector, HeadPoseEstimator


def draw_landmarks(frame, points_2d, color=(0, 255, 0), radius=1):
    for (x, y) in points_2d:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def draw_head_pose(frame, pose_result):
    if pose_result is None:
        return

    cv2.line(
        frame,
        pose_result["nose_tip"],
        pose_result["nose_end"],
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Head Yaw: {pose_result['yaw']:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Head Pitch: {pose_result['pitch']:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Head Roll: {pose_result['roll']:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


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


def draw_gaze_text(frame, result):
    yaw, pitch = extract_first_gaze(result)

    if yaw is not None:
        cv2.putText(
            frame,
            f"Gaze Yaw: {yaw:.2f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if pitch is not None:
        cv2.putText(
            frame,
            f"Gaze Pitch: {pitch:.2f}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def extract_face_box(points_2d):
    xs = points_2d[:, 0]
    ys = points_2d[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    return {
        "face_cx": (x_min + x_max) / 2.0,
        "face_cy": (y_min + y_max) / 2.0,
        "face_w": x_max - x_min,
        "face_h": y_max - y_min,
    }


def draw_feature_text(frame, features):
    if features is None:
        return

    y0 = 200
    dy = 24

    lines = [
        f"face_cx: {features['face_cx']:.1f}",
        f"face_cy: {features['face_cy']:.1f}",
        f"face_w: {features['face_w']:.1f}",
        f"face_h: {features['face_h']:.1f}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def draw_gaze_arrow(frame, origin, yaw, pitch, length=120, color=(255, 0, 0), thickness=2):
    dx = int(length * np.sin(yaw))
    dy = int(-length * np.sin(pitch))

    end_point = (int(origin[0] + dx), int(origin[1] + dy))
    cv2.line(frame, (int(origin[0]), int(origin[1])), end_point, color, thickness)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    face_model_path = project_root / "models" / "face_landmarker.task"
    gaze_model_path = project_root / "models" / "L2CSNet_gaze360.pkl"

    if not face_model_path.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {face_model_path}")

    if not gaze_model_path.exists():
        raise FileNotFoundError(f"L2CS model not found: {gaze_model_path}")

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    start_time = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame_idx += 1

            # DO NOT mirror if you want true left/right debugging
            # frame = cv2.flip(frame, 1)

            gaze_result = gaze_pipeline.step(frame)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mesh_result = face_mesh.process(frame, timestamp_ms)

            pose_result = None
            if mesh_result is not None:
                pose_result = head_pose_estimator.estimate(frame, mesh_result["points_2d"])
                draw_landmarks(frame, mesh_result["points_2d"], radius=1)

            yaw, pitch = extract_first_gaze(gaze_result)
            features = None

            if mesh_result is not None and pose_result is not None and yaw is not None and pitch is not None:
                face_box = extract_face_box(mesh_result["points_2d"])

                features = {
                    "gaze_yaw": yaw,
                    "gaze_pitch": pitch,
                    "head_yaw": pose_result["yaw"],
                    "head_pitch": pose_result["pitch"],
                    "head_roll": pose_result["roll"],
                    "face_cx": face_box["face_cx"],
                    "face_cy": face_box["face_cy"],
                    "face_w": face_box["face_w"],
                    "face_h": face_box["face_h"],
                }

                if frame_idx % 30 == 0:
                    print(features)

                eye_center = extract_eye_center(mesh_result["points_2d"])
                draw_gaze_arrow(
                    frame,
                    origin=eye_center,
                    yaw=yaw,
                    pitch=pitch,
                    length=140,
                    color=(255, 0, 0),
                    thickness=2,
                )

            draw_head_pose(frame, pose_result)
            draw_gaze_text(frame, gaze_result)
            draw_feature_text(frame, features)

            cv2.imshow("L2CS + FaceLandmarker + HeadPose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("Exiting on user request.")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting cleanly...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")


if __name__ == "__main__":
    main()