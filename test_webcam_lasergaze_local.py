import time
from pathlib import Path
import cv2
import torch
from l2cs import Pipeline, FaceMeshDetector, HeadPoseEstimator
from l2cs.temporal_filters import TemporalGazeTracker
from l2cs.console_logger import log_snapshot
from l2cs.gaze_debug_utils import (
    build_side_panel,
    draw_arrow,
    draw_landmark_groups,
    extract_eye_center,
)
from l2cs.lasergaze_adapter import LaserGazeAdapter


WINDOW_NAME = "LaserGaze-style Webcam Debug"


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


def l2cs_to_vector(yaw, pitch):
    if yaw is None or pitch is None:
        return None

    dx = -torch.sin(torch.tensor(yaw)).item()
    dy = -torch.sin(torch.tensor(pitch)).item()
    return [dx, dy]


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

    laser_adapter = LaserGazeAdapter(
        avg_gain_x=2.5,
        avg_gain_y=2.0,
        invert_x=False,
        invert_y=True,
        left_alpha=0.60,
        right_alpha=0.60,
        avg_alpha=0.55,
    )
    
    tracker = TemporalGazeTracker(
        iris_alpha=0.55,
        l2cs_alpha=0.75,
        head_alpha=0.70,
        final_alpha=0.60,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mirror_mode = False
    show_l2cs = True
    frame_idx = 0

    try:
        with torch.no_grad():
            while True:
                loop_start = time.time()

                ok, frame = cap.read()
                if not ok:
                    break

                if mirror_mode:
                    frame = cv2.flip(frame, 1)

                frame_idx += 1
                timestamp_ms = int(time.time() * 1000)

                gaze_result = gaze_pipeline.step(frame)
                yaw, pitch = extract_first_gaze(gaze_result)
                l2cs_vec_raw = l2cs_to_vector(yaw, pitch)

                mesh_result = face_mesh.process(frame, timestamp_ms)

                laser_result = laser_adapter.process(mesh_result, timestamp_ms)

                points_2d = laser_result["points_2d"]
                if points_2d is not None:
                    draw_landmark_groups(frame, points_2d)

                head_pose = None
                if points_2d is not None:
                    try:
                        head_pose = head_pose_estimator.estimate(frame, points_2d)
                    except Exception:
                        head_pose = None

                tracker_snapshot = tracker.update(
                    iris_vec=laser_result["avg_vec_2d"],
                    l2cs_vec=l2cs_vec_raw,
                    head_pose=head_pose,
                    blink_like=False,
                )

                eye_center = extract_eye_center(points_2d)

                if tracker_snapshot["iris_vec"] is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        tracker_snapshot["iris_vec"],
                        length=145,
                        color=(255, 255, 255),
                        thickness=3,
                        label="laser",
                    )

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

                fps = 1.0 / max(time.time() - loop_start, 1e-6)

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

                combined = cv2.hconcat([frame, panel])
                cv2.imshow(WINDOW_NAME, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    laser_adapter.reset()
                    tracker.reset()
                if key == ord("m"):
                    mirror_mode = not mirror_mode
                if key == ord("l"):
                    show_l2cs = not show_l2cs

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()