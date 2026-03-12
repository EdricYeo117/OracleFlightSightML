import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer


class Pipeline:
    def __init__(
        self,
        weights: pathlib.Path,
        arch: str,
        device: str = "cpu",
        include_detector: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        if self.include_detector:
            if device.type == "cpu":
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None:
                for box, landmark, score in faces:
                    if score < self.confidence_threshold:
                        continue

                    x_min = max(int(box[0]), 0)
                    y_min = max(int(box[1]), 0)
                    x_max = min(int(box[2]), frame.shape[1])
                    y_max = min(int(box[3]), frame.shape[0])

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    img = frame[y_min:y_max, x_min:x_max]
                    if img is None or img.size == 0:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

            if len(face_imgs) == 0:
                return GazeResultContainer(
                    pitch=np.empty((0,), dtype=np.float32),
                    yaw=np.empty((0,), dtype=np.float32),
                    bboxes=np.empty((0, 4), dtype=np.float32),
                    landmarks=np.empty((0, 5, 2), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                )

            pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            return GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.asarray(bboxes, dtype=np.float32),
                landmarks=np.asarray(landmarks, dtype=np.float32),
                scores=np.asarray(scores, dtype=np.float32),
            )

        pitch, yaw = self.predict_gaze(frame)

        return GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.empty((0, 4), dtype=np.float32),
            landmarks=np.empty((0, 5, 2), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")

        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)

        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted