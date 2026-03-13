import time
from pathlib import Path

import cv2
import numpy as np
import torch

from l2cs.pipeline import Pipeline
from l2cs.face_mesh_localmodel import FaceMeshDetector
from l2cs.head_pose import HeadPoseEstimator
from l2cs.temporal_filters import TemporalGazeTracker
from l2cs.eye_gaze_estimator import EyeGazeEstimator
from l2cs.lasergaze_adapter import LaserGazeAdapter
from l2cs.console_logger import log_snapshot
from l2cs.gaze_debug_utils import (
    build_side_panel,
    draw_arrow,
    draw_landmark_groups,
    extract_eye_center,
)


WINDOW_NAME = "Hybrid LaserGaze + L2CS"

LEFT_IRIS_IDXS = [469, 470, 471, 472]
RIGHT_IRIS_IDXS = [474, 475, 476, 477]


def extract_first_gaze(result):
    if result is None:
        return None, None

    if not hasattr(result, "yaw") or not hasattr(result, "pitch"):
        return None, None

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


def l2cs_angles_to_vec2(pitch, yaw, gain_x=1.0, gain_y=1.0, invert_x=False, invert_y=False):
    if yaw is None or pitch is None:
        return None

    dx = -np.sin(yaw) * gain_x
    dy = -np.sin(pitch) * gain_y

    if invert_x:
        dx = -dx
    if invert_y:
        dy = -dy

    return np.array(
        [
            float(np.clip(dx, -1.0, 1.0)),
            float(np.clip(dy, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def iris_center(points_2d, idxs):
    if points_2d is None:
        return None
    if len(points_2d) <= max(idxs):
        return None
    return points_2d[idxs].mean(axis=0).astype(int)


def choose_primary_iris_vec(laser_result, iris_result):
    laser_vec = laser_result.get("avg_vec_2d")
    iris_vec = None

    if iris_result is not None:
        iris_vec = np.array(
            [iris_result["eye_dx"], iris_result["eye_dy"]],
            dtype=np.float32,
        )

    left_ready = laser_result["left_detector"].center_detected
    right_ready = laser_result["right_detector"].center_detected

    if (left_ready or right_ready) and laser_vec is not None:
        return laser_vec

    return iris_vec


def should_degrade_laser(head_pose, laser_result, iris_result):
    left_conf = float(laser_result["left_detector"].current_confidence)
    right_conf = float(laser_result["right_detector"].current_confidence)
    max_conf = max(left_conf, right_conf)

    if max_conf < 0.995:
        return True

    if head_pose is not None and abs(head_pose["yaw"]) > 35.0:
        return True

    return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    gaze_model_path = project_root / "models" / "L2CSNet_gaze360.pkl"
    face_model_path = project_root / "models" / "face_landmarker.task"

    if not gaze_model_path.exists():
        raise FileNotFoundError(f"L2CS model not found: {gaze_model_path}")
    if not face_model_path.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {face_model_path}")

    gaze_pipeline = Pipeline(
        weights=gaze_model_path,
        arch="ResNet50",
        device=device,
        include_detector=True,
        confidence_threshold=0.6,
    )

    face_mesh = FaceMeshDetector(
        model_path=str(face_model_path),
        num_faces=1,
        running_mode="VIDEO",
    )

    head_pose_estimator = HeadPoseEstimator()
    eye_gaze_estimator = EyeGazeEstimator(
        center_tolerance_x=0.06,
        center_tolerance_y=0.08,
        smoothing=0.35,
        baseline_alpha=0.15,
    )

    # Lower smoothing here so LaserGaze feels fast like the dual-line demo.
    laser_adapter = LaserGazeAdapter(
        avg_gain_x=2.5,
        avg_gain_y=2.0,
        invert_x=False,
        invert_y=True,
        left_alpha=0.20,
        right_alpha=0.20,
        avg_alpha=0.20,
    )

    # Keep fusion lighter too.
    tracker = TemporalGazeTracker(
        iris_alpha=0.20,
        l2cs_alpha=0.35,
        head_alpha=0.50,
        final_alpha=0.10,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mirror_mode = False
    show_l2cs = True
    show_fast_laser = True
    frame_idx = 0

    try:
        with torch.no_grad():
            while True:
                t0 = time.time()

                ok, frame = cap.read()
                if not ok:
                    break

                if mirror_mode:
                    frame = cv2.flip(frame, 1)

                frame_idx += 1
                timestamp_ms = int(time.time() * 1000)

                # L2CS fallback branch
                gaze_result = gaze_pipeline.step(frame)
                yaw, pitch = extract_first_gaze(gaze_result)
                l2cs_vec = l2cs_angles_to_vec2(
                    pitch=pitch,
                    yaw=yaw,
                    gain_x=1.0,
                    gain_y=1.0,
                    invert_x=False,
                    invert_y=False,
                )

                # LaserGaze branch
                mesh_result = face_mesh.process(frame, timestamp_ms)
                laser_result = laser_adapter.process(mesh_result, timestamp_ms)

                points_2d = laser_result["points_2d"]
                if points_2d is not None:
                    draw_landmark_groups(frame, points_2d)

                head_pose = None
                iris_result = None

                if points_2d is not None:
                    try:
                        head_pose = head_pose_estimator.estimate(frame, points_2d)
                    except Exception:
                        head_pose = None

                    try:
                        iris_result = eye_gaze_estimator.estimate(points_2d)
                    except Exception:
                        iris_result = None

                if iris_result is not None and not iris_result["blink_like"]:
                    if abs(iris_result["eye_dx"]) < 0.03 and abs(iris_result["eye_dy"]) < 0.03:
                        eye_gaze_estimator.update_baseline(
                            iris_result["eye_x"],
                            iris_result["eye_y"],
                        )

                primary_iris_vec = choose_primary_iris_vec(laser_result, iris_result)

                blink_like = bool(iris_result.get("blink_like", False)) if iris_result is not None else False
                degrade_primary = should_degrade_laser(head_pose, laser_result, iris_result)

                left_conf = float(laser_result["left_detector"].current_confidence)
                right_conf = float(laser_result["right_detector"].current_confidence)
                primary_confidence = max(left_conf, right_conf)

                tracker_snapshot = tracker.update(
                    iris_vec=primary_iris_vec,
                    l2cs_vec=l2cs_vec,
                    head_pose=head_pose,
                    blink_like=blink_like,
                    degrade_primary=degrade_primary,
                    primary_confidence=primary_confidence,
                )

                eye_center = extract_eye_center(points_2d)
                left_origin = iris_center(points_2d, LEFT_IRIS_IDXS)
                right_origin = iris_center(points_2d, RIGHT_IRIS_IDXS)

                # Fast LaserGaze-style per-eye lines
                if show_fast_laser and laser_result["left_vec_2d"] is not None:
                    draw_arrow(
                        frame,
                        left_origin,
                        laser_result["left_vec_2d"],
                        length=105,
                        color=(255, 0, 255),
                        thickness=2,
                        label="L",
                    )

                if show_fast_laser and laser_result["right_vec_2d"] is not None:
                    draw_arrow(
                        frame,
                        right_origin,
                        laser_result["right_vec_2d"],
                        length=105,
                        color=(0, 165, 255),
                        thickness=2,
                        label="R",
                    )

                # Averaged LaserGaze preview
                if show_fast_laser and laser_result["avg_vec_2d"] is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        laser_result["avg_vec_2d"],
                        length=135,
                        color=(255, 255, 255),
                        thickness=2,
                        label="laser",
                    )

                # L2CS fallback arrow
                if show_l2cs and tracker_snapshot["l2cs_vec"] is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        tracker_snapshot["l2cs_vec"],
                        length=120,
                        color=(0, 255, 255),
                        thickness=2,
                        label="l2cs",
                    )

                # Final stable fused arrow
                if tracker_snapshot["final_vec"] is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        tracker_snapshot["final_vec"],
                        length=165,
                        color=(0, 255, 0),
                        thickness=2,
                        label="final",
                    )

                log_snapshot(
                    frame_idx=frame_idx,
                    left_gaze_vec=laser_result["left_gaze_vec"],
                    right_gaze_vec=laser_result["right_gaze_vec"],
                    tracker_snapshot=tracker_snapshot,
                    every_n=5,
                )

                fps = 1.0 / max(time.time() - t0, 1e-6)

                panel = build_side_panel(
                    frame.shape[0],
                    laser_result["left_gaze_vec"],
                    laser_result["right_gaze_vec"],
                    tracker_snapshot,
                    fps,
                    laser_result["left_detector"],
                    laser_result["right_detector"],
                    show_l2cs,
                    mirror_mode,
                )

                cv2.putText(
                    panel,
                    f"Laser ready: {laser_result['left_detector'].center_detected or laser_result['right_detector'].center_detected}",
                    (20, panel.shape[0] - 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 255, 200),
                    2,
                )
                cv2.putText(
                    panel,
                    f"Iris calibrated: {eye_gaze_estimator.is_calibrated}",
                    (20, panel.shape[0] - 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 255, 200),
                    2,
                )
                cv2.putText(
                    panel,
                    f"Fast laser shown: {show_fast_laser}",
                    (20, panel.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 255, 200),
                    2,
                )

                combined = cv2.hconcat([frame, panel])
                cv2.imshow(WINDOW_NAME, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    laser_adapter.reset()
                    tracker.reset()
                    eye_gaze_estimator.reset()
                if key == ord("m"):
                    mirror_mode = not mirror_mode
                if key == ord("l"):
                    show_l2cs = not show_l2cs
                if key == ord("f"):
                    show_fast_laser = not show_fast_laser

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()