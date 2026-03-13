import time
from pathlib import Path

import cv2
import numpy as np
import torch

from l2cs import Pipeline, FaceMeshDetector, EyeGazeEstimator, HeadPoseEstimator


PANEL_WIDTH = 430
WINDOW_NAME = "Iris + L2CS Direction Debug"

LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [263, 362, 386, 374]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]


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


def draw_point(frame, pt, color, radius=2):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), radius, color, -1)


def draw_eye_debug(frame, eye_data, color=(0, 255, 0)):
    iris = eye_data["iris_center"].astype(int)
    outer_corner = eye_data["outer_corner"].astype(int)
    inner_corner = eye_data["inner_corner"].astype(int)
    upper_lid = eye_data["upper_lid"].astype(int)
    lower_lid = eye_data["lower_lid"].astype(int)

    cv2.circle(frame, tuple(iris), 4, (0, 0, 255), -1)
    cv2.circle(frame, tuple(outer_corner), 3, color, -1)
    cv2.circle(frame, tuple(inner_corner), 3, color, -1)
    cv2.circle(frame, tuple(upper_lid), 3, color, -1)
    cv2.circle(frame, tuple(lower_lid), 3, color, -1)

    cv2.line(frame, tuple(outer_corner), tuple(inner_corner), color, 1)
    cv2.line(frame, tuple(upper_lid), tuple(lower_lid), color, 1)


def draw_mediapipe_eye_landmarks(frame, points_2d):
    if points_2d is None or len(points_2d) < 477:
        return

    for idx in LEFT_EYE_IDX:
        draw_point(frame, points_2d[idx], (0, 255, 0), radius=2)

    for idx in RIGHT_EYE_IDX:
        draw_point(frame, points_2d[idx], (255, 255, 0), radius=2)

    for idx in LEFT_IRIS_IDX:
        draw_point(frame, points_2d[idx], (255, 0, 255), radius=3)

    for idx in RIGHT_IRIS_IDX:
        draw_point(frame, points_2d[idx], (0, 165, 255), radius=3)

    left_iris_pts = np.array([points_2d[i] for i in LEFT_IRIS_IDX], dtype=np.int32)
    right_iris_pts = np.array([points_2d[i] for i in RIGHT_IRIS_IDX], dtype=np.int32)

    cv2.polylines(frame, [left_iris_pts], True, (255, 0, 255), 1)
    cv2.polylines(frame, [right_iris_pts], True, (0, 165, 255), 1)


def extract_eye_center(points_2d):
    if points_2d is None or len(points_2d) <= 263:
        return None

    left_eye = points_2d[33]
    right_eye = points_2d[263]
    return ((left_eye + right_eye) / 2.0).astype(int)


def draw_arrow(frame, origin, dx, dy, length=120, color=(0, 0, 255), thickness=2, label=None):
    if origin is None:
        return

    end_x = int(origin[0] + dx * length)
    end_y = int(origin[1] + dy * length)

    cv2.arrowedLine(
        frame,
        (int(origin[0]), int(origin[1])),
        (end_x, end_y),
        color,
        thickness,
        tipLength=0.2,
    )

    if label:
        cv2.putText(
            frame,
            label,
            (end_x + 6, end_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def l2cs_to_vector(yaw, pitch):
    if yaw is None or pitch is None:
        return None

    # Use screen-style direction:
    # left  = negative x
    # right = positive x
    # up    = negative y
    # down  = positive y
    dx = -np.sin(yaw)
    dy = -np.sin(pitch)
    return float(dx), float(dy)


def iris_to_vector(eye_dx, eye_dy, gain_x=6.0, gain_y=5.0, invert_x=True, invert_y=False):
    dx = eye_dx * gain_x
    dy = eye_dy * gain_y

    if invert_x:
        dx = -dx
    if invert_y:
        dy = -dy

    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dx, dy


def vector_to_dir(dx, dy, thresh=0.12):
    h = "center"
    v = "center"

    if dx <= -thresh:
        h = "left"
    elif dx >= thresh:
        h = "right"

    if dy <= -thresh:
        v = "up"
    elif dy >= thresh:
        v = "down"

    return f"{v}-{h}"


# Kept for later use if you want fusion again.
# def head_pose_strength(pose_result):
#     if pose_result is None:
#         return 0.0
#
#     yaw = abs(float(pose_result.get("yaw", 0.0)))
#     pitch = abs(float(pose_result.get("pitch", 0.0)))
#
#     yaw_strength = min(1.0, yaw / 20.0)
#     pitch_strength = min(1.0, pitch / 20.0)
#     return float(max(yaw_strength, pitch_strength))


# Kept for later use if you want fusion again.
# def fuse_vectors_dynamic(iris_vec, l2cs_vec, eye_result=None, pose_result=None):
#     if iris_vec is None and l2cs_vec is None:
#         return None
#     if iris_vec is None:
#         return {
#             "dx": float(l2cs_vec[0]),
#             "dy": float(l2cs_vec[1]),
#             "iris_weight": 0.0,
#             "l2cs_weight": 1.0,
#         }
#     if l2cs_vec is None:
#         return {
#             "dx": float(iris_vec[0]),
#             "dy": float(iris_vec[1]),
#             "iris_weight": 1.0,
#             "l2cs_weight": 0.0,
#         }
#
#     iris_strength = float(np.hypot(iris_vec[0], iris_vec[1]))
#     iris_strength = min(1.0, iris_strength)
#
#     head_strength = head_pose_strength(pose_result)
#
#     iris_weight = 0.70 + 0.20 * iris_strength + 0.08 * head_strength
#     iris_weight = float(np.clip(iris_weight, 0.70, 0.95))
#     l2cs_weight = 1.0 - iris_weight
#
#     if eye_result is not None and eye_result.get("blink_like", False):
#         iris_weight = 0.55
#         l2cs_weight = 0.45
#
#     dx = iris_weight * iris_vec[0] + l2cs_weight * l2cs_vec[0]
#     dy = iris_weight * iris_vec[1] + l2cs_weight * l2cs_vec[1]
#
#     return {
#         "dx": float(np.clip(dx, -1.0, 1.0)),
#         "dy": float(np.clip(dy, -1.0, 1.0)),
#         "iris_weight": float(iris_weight),
#         "l2cs_weight": float(l2cs_weight),
#     }


# Kept for later use if you want region logic again.
# def classify_6_region_from_vector(dx, dy, x_thresh=0.12, y_thresh=0.0):
#     if dx <= -x_thresh:
#         col = "left"
#     elif dx >= x_thresh:
#         col = "right"
#     else:
#         col = "middle"
#
#     if dy < y_thresh:
#         row = "top"
#     else:
#         row = "bottom"
#
#     return f"{row} {col}"


def build_side_panel(
    height,
    eye_result,
    yaw,
    pitch,
    pose_result,
    iris_vec,
    l2cs_vec,
    fps,
    calibration_state,
    mirror_mode,
    iris_invert_x,
    iris_invert_y,
):
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)

    y = 35
    dy = 28

    def put(text, color=(255, 255, 255), scale=0.62, thickness=2):
        nonlocal y
        cv2.putText(
            panel,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
        y += dy

    put("Iris + L2CS Direction Debug", color=(0, 255, 255), scale=0.82)
    put(f"FPS: {fps:.1f}", color=(220, 220, 220), scale=0.58)

    if calibration_state["active"]:
        put("Calibration: LOOK CENTER", color=(0, 255, 255), scale=0.72)
        put(f"Collecting: {calibration_state['remaining']:.1f}s", color=(0, 255, 255), scale=0.65)
    else:
        put("Calibration: DONE", color=(0, 255, 0), scale=0.68)

    put(f"Mirror mode: {mirror_mode}", color=(180, 180, 255), scale=0.58)
    put(f"Iris invert X: {iris_invert_x}", color=(180, 180, 255), scale=0.58)
    put(f"Iris invert Y: {iris_invert_y}", color=(180, 180, 255), scale=0.58)

    y += 6

    if eye_result is not None:
        put(f"Iris raw dir: {eye_result['direction']}", color=(0, 255, 0))
        put(f"eye_dx_raw: {eye_result.get('eye_dx_raw', eye_result['eye_dx']):.3f}", color=(0, 255, 0), scale=0.58)
        put(f"eye_dy_raw: {eye_result.get('eye_dy_raw', eye_result['eye_dy']):.3f}", color=(0, 255, 0), scale=0.58)
        put(f"eye_dx: {eye_result['eye_dx']:.3f}", color=(0, 255, 0), scale=0.58)
        put(f"eye_dy: {eye_result['eye_dy']:.3f}", color=(0, 255, 0), scale=0.58)

        baseline_eye_x = eye_result.get("baseline_eye_x", 0.5)
        baseline_eye_y = eye_result.get("baseline_eye_y", 0.5)
        put(
            f"baseline: ({baseline_eye_x:.3f}, {baseline_eye_y:.3f})",
            color=(180, 255, 180),
            scale=0.56,
        )
        put(
            f"conf: {eye_result['confidence']:.2f} blink: {eye_result['blink_like']}",
            color=(255, 255, 255),
            scale=0.56,
        )
    else:
        put("Iris: none", color=(0, 0, 255))

    y += 6

    if yaw is not None and pitch is not None:
        put(f"L2CS yaw: {np.degrees(yaw):.2f} deg", color=(255, 255, 0), scale=0.58)
        put(f"L2CS pitch: {np.degrees(pitch):.2f} deg", color=(255, 255, 0), scale=0.58)
    else:
        put("L2CS: none", color=(0, 0, 255))

    if pose_result is not None:
        put(f"Head yaw: {pose_result['yaw']:.2f}", color=(255, 200, 120), scale=0.58)
        put(f"Head pitch: {pose_result['pitch']:.2f}", color=(255, 200, 120), scale=0.58)
        put(f"Head roll: {pose_result['roll']:.2f}", color=(255, 200, 120), scale=0.58)

    y += 6

    if iris_vec is not None:
        put(f"Iris vec: ({iris_vec[0]:.3f}, {iris_vec[1]:.3f})", color=(255, 0, 255), scale=0.56)
        put(f"Iris dir: {vector_to_dir(iris_vec[0], iris_vec[1])}", color=(255, 0, 255), scale=0.56)

    if l2cs_vec is not None:
        put(f"L2CS vec: ({l2cs_vec[0]:.3f}, {l2cs_vec[1]:.3f})", color=(0, 255, 255), scale=0.56)
        put(f"L2CS dir: {vector_to_dir(l2cs_vec[0], l2cs_vec[1])}", color=(0, 255, 255), scale=0.56)

    y += 10
    put("Keys:", color=(255, 255, 0), scale=0.68)
    put("Q / ESC : Quit", color=(220, 220, 220), scale=0.54)
    put("R : Reset calibration", color=(220, 220, 220), scale=0.54)
    put("M : Toggle mirror", color=(220, 220, 220), scale=0.54)
    put("X : Toggle iris X invert", color=(220, 220, 220), scale=0.54)
    put("Y : Toggle iris Y invert", color=(220, 220, 220), scale=0.54)

    return panel


def draw_calibration_target(frame, active, remaining):
    if not active:
        return

    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    cv2.circle(frame, center, 18, (0, 255, 255), 2)
    cv2.circle(frame, center, 3, (0, 255, 255), -1)

    cv2.putText(
        frame,
        f"LOOK HERE {remaining:.1f}s",
        (center[0] - 110, center[1] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


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

    eye_estimator = EyeGazeEstimator(
        center_tolerance_x=0.06,
        center_tolerance_y=0.08,
        smoothing=0.35,
        baseline_alpha=0.15,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    start_time = time.time()

    calibration_duration_s = 2.5
    calibration_samples_x = []
    calibration_samples_y = []

    mirror_mode = False
    iris_invert_x = True
    iris_invert_y = False

    try:
        with torch.no_grad():
            while True:
                loop_start = time.time()

                ok, frame = cap.read()
                if not ok:
                    break

                if mirror_mode:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.time() - start_time) * 1000)
                elapsed_s = time.time() - start_time

                gaze_result = gaze_pipeline.step(frame)
                yaw, pitch = extract_first_gaze(gaze_result)

                mesh_result = face_mesh.process(frame, timestamp_ms)

                points_2d = None
                eye_result = None
                pose_result = None

                if mesh_result is not None and "points_2d" in mesh_result:
                    points_2d = mesh_result["points_2d"]
                    draw_mediapipe_eye_landmarks(frame, points_2d)

                    try:
                        pose_result = head_pose_estimator.estimate(frame, points_2d)
                    except Exception:
                        pose_result = None

                    eye_result = eye_estimator.estimate(points_2d)
                    if eye_result is not None:
                        draw_eye_debug(frame, eye_result["left_eye"], color=(0, 255, 0))
                        draw_eye_debug(frame, eye_result["right_eye"], color=(255, 255, 0))

                        if elapsed_s <= calibration_duration_s and not eye_result["blink_like"]:
                            calibration_samples_x.append(eye_result["eye_x"])
                            calibration_samples_y.append(eye_result["eye_y"])

                calibration_active = elapsed_s <= calibration_duration_s
                calibration_remaining = max(0.0, calibration_duration_s - elapsed_s)

                if not calibration_active and not getattr(eye_estimator, "is_calibrated", False):
                    if calibration_samples_x and calibration_samples_y:
                        if hasattr(eye_estimator, "set_baseline"):
                            eye_estimator.set_baseline(
                                float(np.mean(calibration_samples_x)),
                                float(np.mean(calibration_samples_y)),
                            )
                    else:
                        if hasattr(eye_estimator, "set_baseline"):
                            eye_estimator.set_baseline(0.5, 0.5)

                draw_calibration_target(frame, calibration_active, calibration_remaining)

                eye_center = extract_eye_center(points_2d)

                iris_vec = None
                if eye_result is not None:
                    iris_vec = iris_to_vector(
                        eye_result["eye_dx"],
                        eye_result["eye_dy"],
                        gain_x=6.0,
                        gain_y=5.0,
                        invert_x=iris_invert_x,
                        invert_y=iris_invert_y,
                    )

                l2cs_vec = l2cs_to_vector(yaw, pitch)

                if iris_vec is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        iris_vec[0],
                        iris_vec[1],
                        length=95,
                        color=(255, 0, 255),
                        thickness=2,
                        label="iris",
                    )

                if l2cs_vec is not None:
                    draw_arrow(
                        frame,
                        eye_center,
                        l2cs_vec[0],
                        l2cs_vec[1],
                        length=120,
                        color=(0, 255, 255),
                        thickness=2,
                        label="l2cs",
                    )

                # Disabled for now so you can debug raw direction first.
                # fused_info = fuse_vectors_dynamic(
                #     iris_vec,
                #     l2cs_vec,
                #     eye_result=eye_result,
                #     pose_result=pose_result,
                # )
                #
                # if fused_info is not None:
                #     draw_arrow(
                #         frame,
                #         eye_center,
                #         fused_info["dx"],
                #         fused_info["dy"],
                #         length=150,
                #         color=(255, 255, 255),
                #         thickness=3,
                #         label="fused",
                #     )

                fps = 1.0 / max(time.time() - loop_start, 1e-6)
                panel = build_side_panel(
                    frame.shape[0],
                    eye_result,
                    yaw,
                    pitch,
                    pose_result,
                    iris_vec,
                    l2cs_vec,
                    fps,
                    {
                        "active": calibration_active,
                        "remaining": calibration_remaining,
                    },
                    mirror_mode,
                    iris_invert_x,
                    iris_invert_y,
                )

                combined = np.hstack([frame, panel])
                cv2.imshow(WINDOW_NAME, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    if hasattr(eye_estimator, "reset"):
                        eye_estimator.reset()
                    start_time = time.time()
                    calibration_samples_x.clear()
                    calibration_samples_y.clear()
                if key == ord("m"):
                    mirror_mode = not mirror_mode
                if key == ord("x"):
                    iris_invert_x = not iris_invert_x
                if key == ord("y"):
                    iris_invert_y = not iris_invert_y

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()