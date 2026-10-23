"""Checks tracker.py against MediaPipe's own Face Landmarker on a video.

    uv run compare_tracker.py VIDEO.mp4
"""

import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from eyeliner import LEFT_UPPER_LID, MODEL, RIGHT_UPPER_LID
from tracker import FaceTracker

LIDS = RIGHT_UPPER_LID + LEFT_UPPER_LID

video = cv2.VideoCapture(sys.argv[1])
fps = video.get(cv2.CAP_PROP_FPS)
reference = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=1))
ours = FaceTracker("../data/models")
errors, missing, index = [], 0, 0
while True:
    ok, frame = video.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    expected = reference.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 1000 / fps))
    actual = ours.track(rgb)
    index += 1
    if not expected.face_landmarks:
        continue
    if actual is None:
        missing += 1
        continue
    want = np.array([(p.x * width, p.y * height) for p in expected.face_landmarks[0]])
    eye_width = np.linalg.norm(want[33] - want[133])
    errors.append(np.linalg.norm(actual[LIDS] - want[LIDS], axis=1).max() / eye_width)
errors = np.array(errors)
print(f"{index} frames, {missing} missed; lid error in eye widths: median {np.median(errors):.4f}, 95th {np.percentile(errors, 95):.4f}, max {errors.max():.4f}")
