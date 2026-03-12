import cv2
import numpy as np
from .results import GazeResultContainer


def draw_gaze(a, b, c, d, image_in, pitchyaw, thickness=2, color=(255, 255, 0), scale=2.0):
    image_out = image_in
    pitch, yaw = pitchyaw

    length = c * scale
    pos = (int(a + c / 2.0), int(b + d / 2.0))

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    dx = -length * np.sin(yaw)
    dy = -length * np.sin(pitch)

    cv2.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(np.int32)),
        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    return image_out


def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    x_min = int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min = int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max = int(bbox[2])
    y_max = int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    return frame


def render(frame: np.ndarray, results: GazeResultContainer):
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    for i in range(results.pitch.shape[0]):
        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]

        x_min = int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min = int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(
            x_min,
            y_min,
            bbox_width,
            bbox_height,
            frame,
            (pitch, yaw),
            color=(0, 0, 255),
        )

    return frame