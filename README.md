# ComfyUI-AIA-BatchVideoDegrade

Small AIA custom node pack for applying cheap-phone/video-upload degradation to a
batched ComfyUI `IMAGE` tensor.

Repository: https://github.com/Pmobilee/ComfyUI-vid2frames

## Nodes

- `AIA_BatchDequality`: applies per-frame JPEG recompression, sensor-like noise,
  and small brightness/color/contrast jitter while preserving batch shape.
- `AIA_SelectVideoFrame`: scores a decoded video frame batch and returns one
  selected frame. The selector string supports `no face`, `blurry`, `moving`,
  `static`, `flicker`, and `lighting`; empty or unknown selectors use best-still
  scoring. Add an explicit rank such as `moving rank 2` or `lighting #3` to get
  the next high-scoring frame for that category, with temporal spacing applied
  so ranked picks do not cluster around the same moment.
