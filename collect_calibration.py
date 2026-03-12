import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from l2cs import Pipeline, FaceMeshDetector, HeadPoseEstimator
from calibration_utils import (
    build_feature_vector,
    average_feature_dicts,
    FEATURE_COLUMNS,
    generate_9_point_targets,
)


WINDOW_NAME = "Calibration"
CALIBRATION_CSV = "calibration_samples.csv"


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


def draw_dot(frame, point, radius=14, color=(0, 0, 255)):
    cv2.circle(frame, point, radius, color, -1)
    cv2.circle(frame, point, radius + 6, (255, 255, 255), 2)


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
    out_csv_path = project_root / CALIBRATION_CSV

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

    ok, webcam_frame = cap.read()
    if not ok:
        print("Error: could not read initial frame")
        cap.release()
        return

    # Create a fullscreen window first
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show a temporary frame so OpenCV creates the fullscreen surface
    temp = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, temp)
    cv2.waitKey(100)

    # Get actual fullscreen window size
    _, _, screen_w, screen_h = cv2.getWindowImageRect(WINDOW_NAME)

    if screen_w <= 0 or screen_h <= 0:
        screen_h, screen_w = 1080, 1920

    targets = generate_9_point_targets(screen_w, screen_h, margin=0.12)

    rows = []
    point_index = 0
    start_time = time.time()

    settle_seconds = 1.0
    collect_seconds = 1.2

    state = "wait_start"
    state_start = time.time()
    samples_for_point = []

    print("Calibration started.")
    print("Look at each red dot.")
    print("Press SPACE to begin, Q or ESC to quit.")

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

            # Fullscreen canvas
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            # Put webcam preview as inset
            preview_h, preview_w = preview.shape[:2]
            inset_w = min(480, screen_w // 3)
            inset_h = int(preview_h * (inset_w / preview_w))
            preview_small = cv2.resize(preview, (inset_w, inset_h))

            canvas[20:20 + inset_h, 20:20 + inset_w] = preview_small

            now = time.time()

            if state == "wait_start":
                draw_text(
                    canvas,
                    [
                        "9-Point Calibration",
                        "Press SPACE to start",
                        "Look directly at each red dot when collecting",
                    ],
                    start=(30, 40),
                )

            elif point_index < len(targets):
                target = targets[point_index]
                draw_dot(canvas, target)

                if state == "settle":
                    remaining = max(0.0, settle_seconds - (now - state_start))
                    draw_text(
                        canvas,
                        [
                            f"Point {point_index + 1} / {len(targets)}",
                            "Focus on the red dot",
                            f"Settling... {remaining:.1f}s",
                        ],
                        start=(30, 40),
                    )

                    if now - state_start >= settle_seconds:
                        state = "collect"
                        state_start = now
                        samples_for_point = []

                elif state == "collect":
                    remaining = max(0.0, collect_seconds - (now - state_start))
                    draw_text(
                        canvas,
                        [
                            f"Point {point_index + 1} / {len(targets)}",
                            "Keep looking at the red dot",
                            f"Collecting... {remaining:.1f}s",
                        ],
                        start=(30, 40),
                    )

                    if mesh_result is not None and pose_result is not None and yaw is not None and pitch is not None:
                        features = build_feature_vector(
                            gaze_yaw=yaw,
                            gaze_pitch=pitch,
                            pose_result=pose_result,
                            points_2d=mesh_result["points_2d"],
                        )
                        samples_for_point.append(features)

                    if now - state_start >= collect_seconds:
                        if samples_for_point:
                            avg_features = average_feature_dicts(samples_for_point)
                            row = {
                                **avg_features,
                                "target_x": float(target[0]),
                                "target_y": float(target[1]),
                                "point_index": point_index,
                            }
                            rows.append(row)
                            print(f"Saved point {point_index + 1}: {target}")
                        else:
                            print(f"No valid samples for point {point_index + 1}")

                        point_index += 1
                        state = "settle"
                        state_start = now

            else:
                draw_text(
                    canvas,
                    [
                        "Calibration complete",
                        f"Collected {len(rows)} averaged samples",
                        "Press S to save and exit",
                    ],
                    start=(30, 40),
                )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                print("Exiting without saving.")
                break

            if state == "wait_start" and key == ord(" "):
                state = "settle"
                state_start = time.time()

            if point_index >= len(targets) and key == ord("s"):
                if rows:
                    fieldnames = FEATURE_COLUMNS + ["target_x", "target_y", "point_index"]
                    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"Saved calibration CSV to: {out_csv_path}")
                else:
                    print("No rows to save.")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting cleanly...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()