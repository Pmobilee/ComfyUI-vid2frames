# ComfyUI-vid2frames

A small ComfyUI custom node package for selecting useful still frames from a decoded video frame batch.

## Node

### `AIA_SelectVideoFrame`

Inputs:

- `image`: ComfyUI `IMAGE` batch, usually decoded video frames.
- `selector`: text command describing the target frame type.

Output:

- A single-frame ComfyUI `IMAGE` batch.

Supported selector modes:

- Empty or unknown text: best still frame.
- `no face`: lowest face score while still preferring usable frames.
- `blurry`: softest reasonable frame.
- `moving`: strongest motion frame with blur and lighting penalties.
- `static`: low-motion sharp frame.
- `flicker` or `lighting`: strongest lighting fluctuation.

Ranked variants are supported. For example, `moving rank 2` or `lighting #3` returns another high-scoring frame in the same category. Ranked picks apply temporal spacing so nearby frames from the same moment do not crowd out more useful alternatives.

## How It Scores Frames

The node computes:

- Sharpness from Laplacian variance.
- Motion from neighbor-frame pixel deltas, smoothed over a short window.
- Lighting fluctuation from luminance deltas, smoothed over a short window.
- Face score from MediaPipe face detection, with OpenCV Haar cascades as a no-download fallback.

The first and last 8% of frames are lightly penalized for normal-length videos, because endpoints are often less representative.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
git clone https://github.com/Pmobilee/ComfyUI-vid2frames.git ComfyUI-vid2frames
pip install -r ComfyUI-vid2frames/requirements.txt
```

## Tests

```bash
python -m pytest tests/test_frame_selector.py -q
```
