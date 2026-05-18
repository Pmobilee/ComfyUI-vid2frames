import re

import numpy as np
import torch


def _optional_cv2():
    try:
        import cv2
    except Exception:
        return None
    return cv2


def _normalize_selector(selector):
    text = re.sub(r"\s+", " ", str(selector or "").strip().lower())
    if not text:
        return "best"
    if "best" in text:
        return "best"
    if any(term in text for term in ("no face", "faceless", "face out", "without face")):
        return "no_face"
    if any(term in text for term in ("blurry", "blurred", "blur", "out of focus", "soft focus")):
        return "blurry"
    if any(term in text for term in ("moving", "motion", "movement", "dynamic", "action")):
        return "moving"
    if any(term in text for term in ("static", "still", "steady", "posed", "pose")):
        return "static"
    if any(term in text for term in ("flicker", "lighting", "light fluctuation", "flash")):
        return "lighting"
    return "best"


def _normalized(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _smooth(values, radius=2):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return values
    out = np.empty_like(values)
    for idx in range(values.size):
        lo = max(0, idx - radius)
        hi = min(values.size, idx + radius + 1)
        out[idx] = float(np.mean(values[lo:hi]))
    return out


def _edge_weights(count):
    weights = np.ones(count, dtype=np.float32)
    if count < 25:
        return weights
    margin = max(1, int(round(count * 0.08)))
    weights[:margin] = 0.65
    weights[-margin:] = 0.65
    return weights


def _center_tiebreak(count):
    if count <= 1:
        return np.zeros(count, dtype=np.float32)
    midpoint = (count - 1) / 2.0
    distances = np.abs(np.arange(count, dtype=np.float32) - midpoint)
    return (1.0 - (distances / max(midpoint, 1.0))) * 1e-4


def _gray_frame(rgb_frame):
    frame = rgb_frame.astype(np.float32)
    return frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114


def _blur_scores(gray_frames):
    cv2 = _optional_cv2()
    scores = []
    for gray in gray_frames:
        if cv2 is not None:
            scores.append(float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()))
            continue
        gy, gx = np.gradient(gray.astype(np.float32))
        scores.append(float(np.var(gx) + np.var(gy)))
    return np.asarray(scores, dtype=np.float32)


def _motion_and_lighting(gray_frames):
    count = len(gray_frames)
    motion = np.zeros(count, dtype=np.float32)
    lighting = np.zeros(count, dtype=np.float32)
    means = np.asarray([float(np.mean(gray)) for gray in gray_frames], dtype=np.float32)

    for idx, gray in enumerate(gray_frames):
        diffs = []
        light_diffs = []
        if idx > 0:
            diffs.append(float(np.mean(np.abs(gray - gray_frames[idx - 1]))))
            light_diffs.append(abs(float(means[idx] - means[idx - 1])))
        if idx + 1 < count:
            diffs.append(float(np.mean(np.abs(gray - gray_frames[idx + 1]))))
            light_diffs.append(abs(float(means[idx] - means[idx + 1])))
        motion[idx] = float(np.mean(diffs)) if diffs else 0.0
        lighting[idx] = float(np.mean(light_diffs)) if light_diffs else 0.0

    return _smooth(motion), _smooth(lighting)


def _mediapipe_face_scores(rgb_frames):
    try:
        import mediapipe as mp
    except Exception:
        return None

    scores = []
    try:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.25,
        )
    except Exception:
        return None

    try:
        for frame in rgb_frames:
            result = detector.process(frame)
            best = 0.0
            for detection in result.detections or []:
                confidence = float(detection.score[0]) if detection.score else 0.0
                box = detection.location_data.relative_bounding_box
                area = max(0.0, float(box.width)) * max(0.0, float(box.height))
                best = max(best, confidence * (1.0 + min(area, 0.25)))
            scores.append(best)
    except Exception:
        return None
    finally:
        detector.close()

    return np.asarray(scores, dtype=np.float32)


def _opencv_face_scores(rgb_frames):
    cv2 = _optional_cv2()
    if cv2 is None:
        return np.zeros(len(rgb_frames), dtype=np.float32)

    cascade_names = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    )
    cascades = []
    for name in cascade_names:
        path = cv2.data.haarcascades + name
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            cascades.append(cascade)
    if not cascades:
        return np.zeros(len(rgb_frames), dtype=np.float32)

    scores = []
    for frame in rgb_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        min_size = (max(20, width // 12), max(20, height // 12))
        best_area = 0.0
        for cascade in cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=min_size,
            )
            for _, _, face_w, face_h in faces:
                best_area = max(best_area, float(face_w * face_h) / float(width * height))
        scores.append(min(1.0, best_area * 12.0))
    return np.asarray(scores, dtype=np.float32)


class AIA_SelectVideoFrame:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"
    CATEGORY = "AIA/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selector": ("STRING", {"default": "", "multiline": False}),
            }
        }

    @staticmethod
    def _detect_face_scores(rgb_frames):
        scores = _mediapipe_face_scores(rgb_frames)
        if scores is not None:
            return scores
        return _opencv_face_scores(rgb_frames)

    @classmethod
    def _metrics(cls, frames):
        rgb_frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        gray_frames = [_gray_frame(frame) for frame in rgb_frames]
        blur = _blur_scores(gray_frames)
        motion, lighting = _motion_and_lighting(gray_frames)
        face = cls._detect_face_scores(rgb_frames)
        return {
            "blur": blur,
            "motion": motion,
            "lighting": lighting,
            "face": np.asarray(face, dtype=np.float32),
        }

    @staticmethod
    def _choose_index(metrics, mode):
        count = len(metrics["blur"])
        sharp = _normalized(metrics["blur"])
        motion = _normalized(metrics["motion"])
        lighting = _normalized(metrics["lighting"])
        face = _normalized(metrics["face"])

        stable = 1.0 - lighting
        low_motion = 1.0 - motion
        no_face = 1.0 - face
        blurry = 1.0 - sharp
        moderate_motion = np.clip(1.0 - np.abs(motion - 0.55) * 2.0, 0.0, 1.0)

        if mode == "no_face":
            score = no_face * 0.55 + sharp * 0.20 + stable * 0.15 + low_motion * 0.10
        elif mode == "blurry":
            score = blurry * 0.70 + stable * 0.15 + face * 0.10 + low_motion * 0.05
        elif mode == "moving":
            score = motion * 0.55 + sharp * 0.25 + stable * 0.15 + face * 0.05
        elif mode == "static":
            score = low_motion * 0.45 + sharp * 0.35 + stable * 0.15 + face * 0.05
        elif mode == "lighting":
            score = lighting * 0.65 + sharp * 0.20 + motion * 0.10 + face * 0.05
        else:
            score = sharp * 0.50 + stable * 0.20 + moderate_motion * 0.15 + face * 0.15

        score = score * _edge_weights(count) + _center_tiebreak(count)
        return int(np.argmax(score))

    def select(self, image, selector):
        if image is None or image.shape[0] == 0:
            raise ValueError("AIA_SelectVideoFrame received an empty image batch")

        frames = image.detach().cpu().numpy()
        mode = _normalize_selector(selector)
        metrics = self._metrics(frames)
        index = self._choose_index(metrics, mode)
        print(f"AIA_SelectVideoFrame: selector={selector!r} mode={mode} index={index}/{image.shape[0]}")
        return (image[index : index + 1].to(device=image.device, dtype=image.dtype),)


NODE_CLASS_MAPPINGS = {
    "AIA_SelectVideoFrame": AIA_SelectVideoFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIA_SelectVideoFrame": "AIA Select Video Frame",
}
