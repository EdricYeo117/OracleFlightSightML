import cv2
import numpy as np

from .console_logger import vector_to_dir


LEFT_IRIS = [469, 470, 471, 472]
LEFT_PUPIL = 468
RIGHT_IRIS = [474, 475, 476, 477]
RIGHT_PUPIL = 473
ADJACENT_LEFT_EYELID_PART = [160, 159, 158, 163, 144, 145, 153]
ADJACENT_RIGHT_EYELID_PART = [387, 386, 385, 390, 373, 374, 380]
OUTER_HEAD_POINTS = [162, 389]
NOSE_BRIDGE = 6
NOSE_TIP = 4


def draw_point(frame, pt, color, radius=2):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), radius, color, -1)


def draw_landmark_groups(frame, points_2d):
    if points_2d is None or len(points_2d) < 478:
        return

    for idx in LEFT_IRIS:
        draw_point(frame, points_2d[idx], (255, 0, 255), 3)
    draw_point(frame, points_2d[LEFT_PUPIL], (0, 0, 255), 4)

    for idx in RIGHT_IRIS:
        draw_point(frame, points_2d[idx], (0, 165, 255), 3)
    draw_point(frame, points_2d[RIGHT_PUPIL], (0, 0, 255), 4)

    for idx in ADJACENT_LEFT_EYELID_PART:
        draw_point(frame, points_2d[idx], (0, 255, 0), 2)

    for idx in ADJACENT_RIGHT_EYELID_PART:
        draw_point(frame, points_2d[idx], (255, 255, 0), 2)

    for idx in [NOSE_BRIDGE, NOSE_TIP] + OUTER_HEAD_POINTS:
        draw_point(frame, points_2d[idx], (255, 255, 255), 2)


def extract_eye_center(points_2d):
    if points_2d is None or len(points_2d) <= 263:
        return None

    left_eye = points_2d[33]
    right_eye = points_2d[263]
    return ((left_eye + right_eye) / 2.0).astype(int)


def draw_arrow(frame, origin_xy, vec_xy, length=140, color=(255, 255, 255), thickness=2, label=None):
    if origin_xy is None or vec_xy is None:
        return

    end_x = int(origin_xy[0] + vec_xy[0] * length)
    end_y = int(origin_xy[1] + vec_xy[1] * length)

    cv2.arrowedLine(
        frame,
        (int(origin_xy[0]), int(origin_xy[1])),
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


def build_side_panel(
    height,
    left_gaze_vec,
    right_gaze_vec,
    tracker_snapshot,
    fps,
    left_detector,
    right_detector,
    show_l2cs,
    mirror_mode,
):
    panel = np.zeros((height, 480, 3), dtype=np.uint8)

    y = 35
    dy = 24

    def put(text, color=(255, 255, 255), scale=0.56, thickness=2):
        nonlocal y
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += dy

    put("LaserGaze-style Debug", color=(0, 255, 255), scale=0.78)
    put(f"FPS: {fps:.1f}", color=(220, 220, 220), scale=0.52)
    put(f"Mirror mode: {mirror_mode}", color=(180, 180, 255), scale=0.52)
    put(f"L2CS shown: {show_l2cs}", color=(180, 180, 255), scale=0.52)

    y += 4

    if left_gaze_vec is not None:
        put(f"left3d=({left_gaze_vec[0]:+.3f},{left_gaze_vec[1]:+.3f},{left_gaze_vec[2]:+.3f})", color=(255, 0, 255), scale=0.46)
    else:
        put("left3d=None", color=(255, 0, 255), scale=0.46)

    if right_gaze_vec is not None:
        put(f"right3d=({right_gaze_vec[0]:+.3f},{right_gaze_vec[1]:+.3f},{right_gaze_vec[2]:+.3f})", color=(0, 165, 255), scale=0.46)
    else:
        put("right3d=None", color=(0, 165, 255), scale=0.46)

    y += 4

    iris_vec = tracker_snapshot["iris_vec"]
    l2cs_vec = tracker_snapshot["l2cs_vec"]
    final_vec = tracker_snapshot["final_vec"]
    head_pose = tracker_snapshot["head_pose"]

    if iris_vec is not None:
        put(f"iris2d=({iris_vec[0]:+.3f},{iris_vec[1]:+.3f})", color=(255, 0, 255), scale=0.50)
        put(f"iris dir: {vector_to_dir(iris_vec[0], iris_vec[1])}", color=(255, 0, 255), scale=0.54)

    if l2cs_vec is not None:
        put(f"l2cs2d=({l2cs_vec[0]:+.3f},{l2cs_vec[1]:+.3f})", color=(0, 255, 255), scale=0.50)
        put(f"l2cs dir: {vector_to_dir(l2cs_vec[0], l2cs_vec[1])}", color=(0, 255, 255), scale=0.54)

    if final_vec is not None:
        put(f"final2d=({final_vec[0]:+.3f},{final_vec[1]:+.3f})", color=(255, 255, 255), scale=0.50)
        put(f"final dir: {vector_to_dir(final_vec[0], final_vec[1])}", color=(255, 255, 255), scale=0.54)

    if tracker_snapshot["iris_weight"] is not None:
        put(
            f"weights iris/l2cs: {tracker_snapshot['iris_weight']:.2f}/{tracker_snapshot['l2cs_weight']:.2f}",
            color=(255, 255, 255),
            scale=0.50,
        )

    if head_pose is not None:
        y += 4
        put(f"head yaw: {head_pose['yaw']:+.2f}", color=(255, 200, 120), scale=0.52)
        put(f"head pitch: {head_pose['pitch']:+.2f}", color=(255, 200, 120), scale=0.52)
        put(f"head roll: {head_pose['roll']:+.2f}", color=(255, 200, 120), scale=0.52)

    y += 4
    put(
        f"left eye center found: {left_detector.center_detected} conf={left_detector.current_confidence:.4f}",
        color=(255, 0, 255),
        scale=0.48,
    )
    put(
        f"right eye center found: {right_detector.center_detected} conf={right_detector.current_confidence:.4f}",
        color=(0, 165, 255),
        scale=0.48,
    )

    y += 8
    put("Keys:", color=(255, 255, 0), scale=0.62)
    put("Q / ESC : Quit", color=(220, 220, 220), scale=0.48)
    put("R : Reset detectors/tracker", color=(220, 220, 220), scale=0.48)
    put("M : Toggle mirror", color=(220, 220, 220), scale=0.48)
    put("L : Toggle L2CS arrow", color=(220, 220, 220), scale=0.48)

    return panel