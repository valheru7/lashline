"""Prototype: draws a basic top-lid eyeliner on a video, to tune the look before the OBS plugin.

    curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    uv run eyeliner.py INPUT.mp4 OUTPUT.mp4 [--compare] [--wing 0.12] [--thickness 0.07]

The face tracker (MediaPipe Face Landmarker) gives the upper-lid contour of each eye. The liner
is a band that sits on the lid just above that contour: thin at the inner corner, thicker at the
outer corner, with an optional small wing. It's drawn into a soft mask and blended into the
frame, so it reads as makeup rather than a sticker on top.

There's no smoothing on purpose. The tracker's own frame-to-frame wobble on her facecam was
about 0.003 eye widths (invisible), while every smoothing filter tried added visible lag when
she turned her head.
"""

import argparse
import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "models/face_landmarker.task"

# Upper-lid contour points of the face mesh, from the inner corner to the outer corner.
# "Right" and "left" are the person's own right and left eye.
RIGHT_UPPER_LID = [133, 173, 157, 158, 159, 160, 161, 246, 33]
LEFT_UPPER_LID = [362, 398, 384, 385, 386, 387, 388, 466, 263]


@dataclass
class Look:
    color: tuple[int, int, int] = (18, 16, 20)  # BGR, a soft black
    opacity: float = 0.85
    # Thickness as a fraction of the eye's width, at the inner and outer corner.
    inner_thickness: float = 0.015
    outer_thickness: float = 0.07
    # How far the wing extends past the outer corner, as a fraction of the eye's width (0 = none).
    wing_length: float = 0.12
    # How far the wing tips up from the lid's line, in degrees.
    wing_lift_degrees: float = 18.0
    # Edge softness as a fraction of the eye's width.
    softness: float = 0.012
    lashes: bool = False


def catmull_rom(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    """A smooth curve through `points`, so the liner doesn't show the mesh's corners."""
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    curve = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            curve.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    curve.append(points[-1])
    return np.array(curve)


def liner_polygon(lid: np.ndarray, look: Look) -> np.ndarray:
    """The outline of the liner band for one eye. `lid` runs from the inner to the outer corner."""
    curve = catmull_rom(lid)
    eye_width = np.linalg.norm(lid[-1] - lid[0])

    # Normals pointing up, away from the eyeball.
    tangents = np.gradient(curve, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    # The upper lid arches above the line between the corners; that's the "up" side.
    arch = lid.mean(axis=0) - (lid[0] + lid[-1]) / 2
    if np.dot(normals[len(normals) // 2], arch) < 0:
        normals = -normals

    # Thin at the inner corner, growing toward the outer corner; it eases in so the inner
    # third stays fine.
    t = np.linspace(0, 1, len(curve))
    thickness = (look.inner_thickness + (look.outer_thickness - look.inner_thickness) * t**1.6) * eye_width
    # Start a hair below the contour so there's no gap of skin between the lashes and the liner.
    lash_edge = curve - normals * 0.008 * eye_width
    top_edge = curve + normals * thickness[:, None]

    outline = list(lash_edge)
    if look.wing_length > 0:
        # The wing continues the lid's outer direction, tipped up, and tapers to a point.
        direction = tangents[-1]
        angle = math.radians(look.wing_lift_degrees)
        up = normals[-1]
        wing_direction = direction * math.cos(angle) + up * math.sin(angle)
        tip = curve[-1] + wing_direction * look.wing_length * eye_width
        outline.append(tip)
    outline += list(top_edge[::-1])
    return np.array(outline)


def draw_liner(frame: np.ndarray, lids: list[np.ndarray], look: Look) -> np.ndarray:
    height, width = frame.shape[:2]
    supersample = 4
    mask = np.zeros((height * supersample, width * supersample), np.uint8)
    softness_px = 0.0
    for lid in lids:
        polygon = liner_polygon(lid, look)
        cv2.fillPoly(mask, [np.round(polygon * supersample).astype(np.int32)], 255, cv2.LINE_AA)
        softness_px = max(softness_px, look.softness * np.linalg.norm(lid[-1] - lid[0]))
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if softness_px > 0.3:
        alpha = cv2.GaussianBlur(alpha, (0, 0), softness_px)
    alpha = (alpha * look.opacity)[:, :, None]
    color = np.array(look.color, np.float32)
    blended = frame.astype(np.float32) * (1 - alpha) + color * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--compare", action="store_true", help="Before and after side by side")
    parser.add_argument("--wing", type=float, default=Look.wing_length)
    parser.add_argument("--thickness", type=float, default=Look.outer_thickness)
    parser.add_argument("--stills", type=str, default="", help="Comma-separated seconds to also save as PNG")
    args = parser.parse_args()
    look = Look(wing_length=args.wing, outer_thickness=args.thickness)

    video = cv2.VideoCapture(args.input)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width * 2 if args.compare else width
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, height))
    still_frames = {round(float(s) * fps) for s in args.stills.split(",") if s}

    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    ))
    index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        timestamp_ms = int(index * 1000 / fps)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, timestamp_ms)
        made_up = frame
        if result.face_landmarks:
            points = np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]])
            lids = [points[RIGHT_UPPER_LID], points[LEFT_UPPER_LID]]
            made_up = draw_liner(frame, lids, look)
        shown = np.hstack([frame, made_up]) if args.compare else made_up
        writer.write(shown)
        if index in still_frames:
            cv2.imwrite(f"{args.output.rsplit('.', 1)[0]}_{index / fps:.1f}s.png", shown)
        index += 1
    writer.release()
    print(f"{index} frames")


if __name__ == "__main__":
    main()
