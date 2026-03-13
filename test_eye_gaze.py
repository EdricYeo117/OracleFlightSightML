import cv2

from l2cs import FaceMeshDetector, EyeGazeEstimator


def draw_eye_debug(frame, eye_data, color=(0, 255, 0)):
    iris = eye_data["iris_center"].astype(int)
    outer_corner = eye_data["outer_corner"].astype(int)
    inner_corner = eye_data["inner_corner"].astype(int)
    upper_lid = eye_data["upper_lid"].astype(int)
    lower_lid = eye_data["lower_lid"].astype(int)

    cv2.circle(frame, tuple(iris), 3, (0, 0, 255), -1)
    cv2.circle(frame, tuple(outer_corner), 2, color, -1)
    cv2.circle(frame, tuple(inner_corner), 2, color, -1)
    cv2.circle(frame, tuple(upper_lid), 2, color, -1)
    cv2.circle(frame, tuple(lower_lid), 2, color, -1)

    cv2.line(frame, tuple(outer_corner), tuple(inner_corner), color, 1)
    cv2.line(frame, tuple(upper_lid), tuple(lower_lid), color, 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    eye_estimator = EyeGazeEstimator()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = face_mesh.process(frame)
        if result is not None:
            points_2d = result["points_2d"]
            eye_result = eye_estimator.estimate(points_2d)

            if eye_result is not None:
                draw_eye_debug(frame, eye_result["left_eye"], color=(0, 255, 0))
                draw_eye_debug(frame, eye_result["right_eye"], color=(255, 255, 0))

                cv2.putText(
                    frame,
                    f"Eye dir: {eye_result['direction']}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"eye_dx={eye_result['eye_dx']:.3f} eye_dy={eye_result['eye_dy']:.3f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"conf={eye_result['confidence']:.2f} blink_like={eye_result['blink_like']}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Eye Gaze Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()