# Lashline

Eyeliner for OBS Studio.


A filter that draws a little winged eyeliner, with optional lashes, on your webcam, live in
OBS Studio. It tracks your face so the liner stays on your lash line as you move, blink and
turn your head.

It's a replacement for Snap Camera's eyeliner lens. It runs entirely on your computer, and
needs no account and no extra app.

## Install

Download the latest release for your system from the Releases page and install it (Windows:
run the installer or unzip into `C:\ProgramData\obs-studio\plugins`). Then in OBS:

1. Right-click your webcam source → **Filters**.
2. Click **+** under Effect Filters → **Lashline**.
3. Put it **above** any Chroma Key filter.

## Settings

| Setting | What it does |
|---|---|
| Color | Liner color. A soft black by default; try a dark brown. |
| Opacity | How strong the liner is. |
| Thickness | Thickness at the outer corner, as a percentage of your eye's width. |
| Wing length / angle | The flick past the outer corner. Length 0 means no wing. |
| Edge softness | How soft the edges are. |
| Lashes | A few thin lashes on the outer half of each lid. |

Sizes are relative to your eyes, so the look stays the same as you lean in or out.

## How it works

1. Every frame, a downscaled copy of the webcam picture goes to a background thread.
2. There, Google's MediaPipe face models find 478 points on the face. The models are run
   through ONNX Runtime and converted to ONNX in `data/models/`.
3. The upper-lid points become a smooth curve, and the liner is a band along it: thin at the
   inner corner, thicker at the outer corner, with a wing. It's drawn on the GPU with soft,
   faded edges.

The face tracking takes about 3 ms per frame on a laptop CPU, and the filter never makes
OBS's render thread wait for it.

## Build

The plugin is based on [obs-plugintemplate](https://github.com/obsproject/obs-plugintemplate),
so its build instructions apply:

```sh
cmake --preset macos        # or windows-x64, ubuntu-x86_64
cmake --build --preset macos
```

The ONNX Runtime release is downloaded during configuration.

### Checks that don't need OBS

`tests/` builds two small tools that run on video files:

- `track-video` prints the tracked points for each frame. Compare them with
  `prototype/compare_tracker.py`.
- `render-video` draws the liner with a software renderer.

`prototype/` is the Python version that was used to design the look.

## Credits

- Face models: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  by Google, Apache License 2.0.
- [ONNX Runtime](https://onnxruntime.ai) by Microsoft, MIT License.

## License

GPL-2.0-or-later, like OBS Studio.
