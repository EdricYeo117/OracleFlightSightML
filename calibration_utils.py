import numpy as np


FEATURE_COLUMNS = [
    "gaze_yaw",
    "gaze_pitch",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_cx",
    "face_cy",
    "face_w",
    "face_h",
]


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


def build_feature_vector(gaze_yaw, gaze_pitch, pose_result, points_2d):
    face_box = extract_face_box(points_2d)

    return {
        "gaze_yaw": float(gaze_yaw),
        "gaze_pitch": float(gaze_pitch),
        "head_yaw": float(pose_result["yaw"]),
        "head_pitch": float(pose_result["pitch"]),
        "head_roll": float(pose_result["roll"]),
        "face_cx": float(face_box["face_cx"]),
        "face_cy": float(face_box["face_cy"]),
        "face_w": float(face_box["face_w"]),
        "face_h": float(face_box["face_h"]),
    }


def feature_dict_to_list(features):
    return [features[col] for col in FEATURE_COLUMNS]


def average_feature_dicts(feature_dicts):
    if not feature_dicts:
        return None

    keys = feature_dicts[0].keys()
    avg = {}
    for key in keys:
        avg[key] = sum(d[key] for d in feature_dicts) / len(feature_dicts)
    return avg


def generate_9_point_targets(screen_w, screen_h, margin=0.1):
    xs = [
        int(screen_w * margin),
        int(screen_w * 0.5),
        int(screen_w * (1.0 - margin)),
    ]
    ys = [
        int(screen_h * margin),
        int(screen_h * 0.5),
        int(screen_h * (1.0 - margin)),
    ]

    targets = []
    for y in ys:
        for x in xs:
            targets.append((x, y))
    return targets