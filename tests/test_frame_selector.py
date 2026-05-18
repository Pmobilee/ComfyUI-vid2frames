import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "frame_selector.py"
SPEC = importlib.util.spec_from_file_location("frame_selector", MODULE_PATH)
frame_selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frame_selector)

AIA_SelectVideoFrame = frame_selector.AIA_SelectVideoFrame
_normalize_selector = frame_selector._normalize_selector


def _checker(size=64, block=8):
    y, x = np.indices((size, size))
    pattern = ((x // block + y // block) % 2) * 255
    return np.stack([pattern, pattern, pattern], axis=-1).astype(np.uint8)


def _to_tensor(frames):
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def _disable_face_detection(monkeypatch, count):
    monkeypatch.setattr(
        AIA_SelectVideoFrame,
        "_detect_face_scores",
        staticmethod(lambda _frames: np.zeros(count, dtype=np.float32)),
    )


def test_selector_aliases():
    assert _normalize_selector("") == "best"
    assert _normalize_selector("best still") == "best"
    assert _normalize_selector("no face please") == "no_face"
    assert _normalize_selector("very blurry") == "blurry"
    assert _normalize_selector("moving") == "moving"
    assert _normalize_selector("static pose") == "static"
    assert _normalize_selector("strong lighting flicker") == "lighting"
    assert _normalize_selector("whatever") == "best"


def test_blurry_mode_selects_soft_frame(monkeypatch):
    sharp = _checker()
    soft = cv2.GaussianBlur(sharp, (17, 17), 0)
    frames = [sharp, soft, sharp]
    _disable_face_detection(monkeypatch, len(frames))

    selected = AIA_SelectVideoFrame().select(_to_tensor(frames), "blurry")[0]

    assert torch.allclose(selected[0], _to_tensor([soft])[0])


def test_moving_and_static_modes_split_on_motion(monkeypatch):
    base = _checker()
    shifted = np.roll(base, 18, axis=1)
    frames = [base, base, base, shifted, base, base, base]
    _disable_face_detection(monkeypatch, len(frames))
    node = AIA_SelectVideoFrame()

    moving = node.select(_to_tensor(frames), "moving")[0]
    static = node.select(_to_tensor(frames), "static")[0]

    assert torch.allclose(moving[0], _to_tensor([shifted])[0])
    assert torch.allclose(static[0], _to_tensor([base])[0])


def test_lighting_mode_selects_brightness_jump(monkeypatch):
    dark = np.full((64, 64, 3), 40, dtype=np.uint8)
    pre_jump = np.full((64, 64, 3), 80, dtype=np.uint8)
    bright = np.full((64, 64, 3), 220, dtype=np.uint8)
    post_jump = np.full((64, 64, 3), 90, dtype=np.uint8)
    frames = [dark.copy() for _ in range(31)]
    frames[15] = pre_jump
    frames[16] = bright
    frames[17] = post_jump
    _disable_face_detection(monkeypatch, len(frames))

    selected = AIA_SelectVideoFrame().select(_to_tensor(frames), "flicker")[0]

    selected_mean = int(round(float(selected[0].mean() * 255)))
    assert selected_mean in {80, 90, 220}


def test_no_face_mode_uses_face_scores(monkeypatch):
    frames = [_checker() for _ in range(5)]
    face_scores = np.asarray([1.0, 0.8, 0.0, 0.7, 1.0], dtype=np.float32)
    monkeypatch.setattr(
        AIA_SelectVideoFrame,
        "_detect_face_scores",
        staticmethod(lambda _frames: face_scores),
    )

    selected = AIA_SelectVideoFrame().select(_to_tensor(frames), "no face")[0]

    assert torch.allclose(selected[0], _to_tensor([frames[2]])[0])


def test_smoke_241_frame_batch(monkeypatch):
    frames = []
    base = _checker()
    for idx in range(241):
        frame = np.roll(base, idx % 32, axis=1)
        frames.append(frame)
    _disable_face_detection(monkeypatch, len(frames))

    selected = AIA_SelectVideoFrame().select(_to_tensor(frames), "")[0]

    assert selected.shape == (1, 64, 64, 3)
