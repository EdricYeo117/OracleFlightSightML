from pathlib import Path
import argparse
import time
import math

import cv2
import torch

from l2cs import Pipeline, render


def draw_status(frame, fps, text):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (430, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def result_to_text(results):
    if results is None:
        return "No face"
    try:
        if len(results) == 0:
            return "No face"
        first = results[0]
    except Exception:
        first = results

    yaw = None
    pitch = None

    if isinstance(first, dict):
        yaw = first.get("yaw")
        pitch = first.get("pitch")
    else:
        yaw = getattr(first, "yaw", None)
        pitch = getattr(first, "pitch", None)

    def to_float(v):
        if v is None:
            return None
        try:
            if hasattr(v, "item"):
                return float(v.item())
            return float(v)
        except Exception:
            return None

    yaw = to_float(yaw)
    pitch = to_float(pitch)

    if yaw is None or pitch is None:
        return "Face detected"

    return f"Yaw: {math.degrees(yaw):+.1f} deg | Pitch: {math.degrees(pitch):+.1f} deg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/L2CSNet_gaze360.pkl")
    parser.add_argument(
        "--arch",
        default="ResNet50",
        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"],
    )
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing weights file: {weights}\n"
            "Place L2CSNet_gaze360.pkl inside the models folder."
        )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"[INFO] device={device}")
    print(f"[INFO] weights={weights}")

    pipeline = Pipeline(
        weights=weights,
        arch=args.arch,
        device=device,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}")

    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        results = pipeline.step(frame)
        output = render(frame, results)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        draw_status(output, fps, result_to_text(results))
        cv2.imshow("L2CS-Net Webcam Test", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()