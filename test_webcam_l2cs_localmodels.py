import time
import cv2
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaze_pipeline = Pipeline(
        weights="models/L2CSNet_gaze360.pkl",
        arch="ResNet50",
        device=device,
    )

    face_mesh = FaceMeshDetector(
        model_path="models/face_landmarker.task",
        num_faces=1,
        running_mode="VIDEO",
    )

    head_pose_estimator = HeadPoseEstimator()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        gaze_result = gaze_pipeline.step(frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        mesh_result = face_mesh.process(frame, timestamp_ms)

        pose_result = None
        if mesh_result is not None:
            pose_result = head_pose_estimator.estimate(frame, mesh_result["points_2d"])
            draw_landmarks(frame, mesh_result["points_2d"], radius=1)

        draw_head_pose(frame, pose_result)
        draw_gaze_text(frame, gaze_result)

        cv2.imshow("L2CS + FaceLandmarker + HeadPose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()