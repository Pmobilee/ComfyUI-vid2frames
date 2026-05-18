# ComfyUI-vid2frames

ComfyUI custom node for selecting one useful still frame from a decoded video frame batch.

## Node

- `AIA_SelectVideoFrame` / **AIA Select Video Frame**

The node accepts a batched ComfyUI `IMAGE` tensor and a selector string, then returns exactly one `IMAGE` frame.

Supported selector targets:

- empty / unknown / `best still` -> best still-frame scoring
- `no face` -> lowest face score while still preferring usable frames
- `blurry` -> blurriest reasonable frame
- `moving` -> highest-motion frame with blur/light penalties
- `static` -> lowest-motion sharp frame
- `flicker` / `lighting` -> strongest lighting fluctuation

## Metrics

- Sharpness: Laplacian variance
- Motion: neighbor frame pixel delta, smoothed over a 5-frame window
- Lighting fluctuation: luminance delta, smoothed over a 5-frame window
- Face score: MediaPipe face detection first, OpenCV Haar fallback
- Edge penalty: first/last 8% of frames are penalized for longer videos

## Install

Clone into ComfyUI custom nodes:

```bash
cd /home/ComfyUI/custom_nodes
git clone https://github.com/Pmobilee/ComfyUI-vid2frames.git
pip install -r ComfyUI-vid2frames/requirements.txt
```

Restart ComfyUI after installation.

## Workflow Use

Wire decoded video frames into `AIA_SelectVideoFrame`, pass selector text such as `moving` or `no face`, then connect the output to `SaveImage`.
