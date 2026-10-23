"""Face tracking without MediaPipe's runtime: the same two models, run through ONNX Runtime.

This is the reference for the C++ plugin. Every step here gets ported one to one, and
`compare_tracker.py` checks it against MediaPipe's own results.

1. When there's no face yet, the detector (BlazeFace, 128x128) finds one in the whole frame.
2. The face's region is cut out, rotated upright using the eyes, and fed to the landmark
   model (256x256), which returns 478 points.
3. On the next frame the region comes from those points instead, so the detector only runs
   again when the face is lost.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

DETECTOR_SIZE = 128
LANDMARK_SIZE = 256
MIN_DETECTION_SCORE = 0.5
MIN_FACE_PRESENCE = 0.5
# The region fed to the landmark model is this much larger than the face.
ROI_SCALE = 1.5
# Detector keypoints 0 and 1 are the eyes; mesh points 33 and 263 are the outer eye corners.
DETECTOR_RIGHT_EYE, DETECTOR_LEFT_EYE = 0, 1
MESH_RIGHT_EYE, MESH_LEFT_EYE = 33, 263


@dataclass
class Region:
    """A rotated square in image pixels."""
    center_x: float
    center_y: float
    size: float
    rotation: float  # radians


def detector_anchors() -> np.ndarray:
    """Anchor centers for the short-range BlazeFace model: 16x16 cells with 2 anchors each,
    then 8x8 cells with 6 each, 896 in all."""
    anchors = []
    for grid, per_cell in ((16, 2), (8, 6)):
        for y in range(grid):
            for x in range(grid):
                anchors += [((x + 0.5) / grid, (y + 0.5) / grid)] * per_cell
    return np.array(anchors, np.float32)


ANCHORS = detector_anchors()


class FaceTracker:
    def __init__(self, models_dir: str):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.detector = ort.InferenceSession(f"{models_dir}/face_detector.onnx", options)
        self.landmarker = ort.InferenceSession(f"{models_dir}/face_landmarks_detector.onnx", options)
        self.region: Region | None = None

    def track(self, rgb: np.ndarray) -> np.ndarray | None:
        """478 (x, y) points in image pixels, or None when there's no face."""
        if self.region is None:
            self.region = self.detect(rgb)
            if self.region is None:
                return None
        points = self.landmarks(rgb, self.region)
        self.region = None if points is None else region_from_landmarks(points)
        return points

    def detect(self, rgb: np.ndarray) -> Region | None:
        height, width = rgb.shape[:2]
        # Letterbox into a square, keeping the aspect ratio.
        scale = DETECTOR_SIZE / max(width, height)
        pad_x = (DETECTOR_SIZE - width * scale) / 2
        pad_y = (DETECTOR_SIZE - height * scale) / 2
        transform = np.array([[scale, 0, pad_x], [0, scale, pad_y]], np.float32)
        square = cv2.warpAffine(rgb, transform, (DETECTOR_SIZE, DETECTOR_SIZE), flags=cv2.INTER_LINEAR)
        tensor = (square.astype(np.float32) / 127.5 - 1.0)[None]
        regressors, scores = self.detector.run(None, {"input": tensor})
        scores = 1 / (1 + np.exp(-np.clip(scores[0, :, 0], -100, 100)))
        best = int(np.argmax(scores))
        if scores[best] < MIN_DETECTION_SCORE:
            return None

        raw = regressors[0, best]
        anchor = ANCHORS[best]

        def to_image(x, y):
            # Detector output is in 128-pixel units relative to the anchor; undo the letterbox.
            square_x = (x / DETECTOR_SIZE + anchor[0]) * DETECTOR_SIZE
            square_y = (y / DETECTOR_SIZE + anchor[1]) * DETECTOR_SIZE
            return (square_x - pad_x) / scale, (square_y - pad_y) / scale

        center_x, center_y = to_image(raw[0], raw[1])
        box_width, box_height = raw[2] / scale, raw[3] / scale
        right_eye = np.array(to_image(raw[4 + 2 * DETECTOR_RIGHT_EYE], raw[5 + 2 * DETECTOR_RIGHT_EYE]))
        left_eye = np.array(to_image(raw[4 + 2 * DETECTOR_LEFT_EYE], raw[5 + 2 * DETECTOR_LEFT_EYE]))
        return Region(center_x, center_y, max(box_width, box_height) * ROI_SCALE, eye_rotation(right_eye, left_eye))

    def landmarks(self, rgb: np.ndarray, region: Region) -> np.ndarray | None:
        to_image = region_to_image(region, LANDMARK_SIZE)
        to_crop = cv2.invertAffineTransform(to_image)
        crop = cv2.warpAffine(rgb, to_crop, (LANDMARK_SIZE, LANDMARK_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        tensor = (crop.astype(np.float32) / 255.0)[None]
        landmarks, presence, _ = self.landmarker.run(None, {"input_12": tensor})
        if 1 / (1 + math.exp(-float(presence.ravel()[0]))) < MIN_FACE_PRESENCE:
            return None
        crop_points = landmarks.reshape(-1, 3)[:, :2]
        return crop_points @ to_image[:, :2].T + to_image[:, 2]


def eye_rotation(right_eye: np.ndarray, left_eye: np.ndarray) -> float:
    """How far the line between the eyes is tilted from horizontal."""
    return math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])


def region_from_landmarks(points: np.ndarray) -> Region:
    rotation = eye_rotation(points[MESH_RIGHT_EYE], points[MESH_LEFT_EYE])
    # The face's extent measured along the rotated axes.
    cos, sin = math.cos(rotation), math.sin(rotation)
    upright = points @ np.array([[cos, -sin], [sin, cos]])
    low, high = upright.min(axis=0), upright.max(axis=0)
    middle = (low + high) / 2
    center = middle @ np.array([[cos, sin], [-sin, cos]])
    size = float(max(high - low)) * ROI_SCALE
    return Region(float(center[0]), float(center[1]), size, rotation)


def region_to_image(region: Region, crop_size: int) -> np.ndarray:
    """The 2x3 transform from crop pixels to image pixels."""
    scale = region.size / crop_size
    cos, sin = math.cos(region.rotation) * scale, math.sin(region.rotation) * scale
    half = crop_size / 2
    return np.array([
        [cos, -sin, region.center_x - cos * half + sin * half],
        [sin, cos, region.center_y - sin * half - cos * half],
    ], np.float32)
