"""
Deployment: photo -> face box -> crops -> real-face score -> verdict.
Runs ONNX models with onnxruntime (no PyTorch needed) or .pth models with PyTorch.
Data prep imports crop_face from here, so training and deployment crop faces identically.
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from fas import REAL, parse_name

ROOT = Path(__file__).resolve().parents[1]


def crop_face(img, bbox, scale, w=80, h=80):
    """Expand face box [x, y, w, h] by `scale` around its centre (shifted to stay inside the image), resize.
    Same rule as Minivision's preprocessing, so crops match what the pretrained models saw."""
    H, W = img.shape[:2]
    x, y, bw, bh = bbox
    scale = min((H - 1) / bh, (W - 1) / bw, scale)
    cx, cy, nw, nh = x + bw / 2, y + bh / 2, bw * scale, bh * scale
    x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if x1 < 0:
        x1, x2 = 0, x2 - x1
    if y1 < 0:
        y1, y2 = 0, y2 - y1
    if x2 > W - 1:
        x1, x2 = x1 - (x2 - W + 1), W - 1
    if y2 > H - 1:
        y1, y2 = y1 - (y2 - H + 1), H - 1
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (w, h))


class FaceDetector:
    """RetinaFace (Caffe) via OpenCV DNN, as in the original project, plus a confidence floor."""

    def __init__(self, model_dir=ROOT / "pretrained" / "detector", min_conf=0.6):
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise ImportError("OpenCV 5 removed the Caffe loader; install OpenCV 4: pip install 'opencv-python<5'")
        d = Path(model_dir)
        self.net = cv2.dnn.readNetFromCaffe(str(d / "deploy.prototxt"), str(d / "Widerface-RetinaFace.caffemodel"))
        self.min_conf = min_conf

    def detect(self, img):
        """-> ([x, y, w, h] or None, confidence)"""
        h, w = img.shape[:2]
        small = img
        if w * h >= 192 * 192:
            ar = w / h
            small = cv2.resize(img, (int(192 * math.sqrt(ar)), int(192 / math.sqrt(ar))))
        self.net.setInput(cv2.dnn.blobFromImage(small, 1, mean=(104, 117, 123)), "data")
        out = self.net.forward("detection_out").reshape(-1, 7)
        best = out[int(np.argmax(out[:, 2]))]
        l, t = max(0, best[3] * w), max(0, best[4] * h)
        r, b = min(w - 1, best[5] * w), min(h - 1, best[6] * h)
        if best[2] < self.min_conf or r - l < 10 or b - t < 10:
            return None, float(best[2])
        return [int(l), int(t), int(r - l + 1), int(b - t + 1)], float(best[2])


def _runner(path):
    """Model file -> function(batch float32 N,3,H,W) -> class probabilities."""
    if path.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return lambda x: sess.run(None, {"input": x})[0]
    import torch

    from fas.model import load

    model, _ = load(path)

    def run(x):
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()

    return run


class LivenessEngine:
    """
    model_dir:  folder with the 2.7x and 4x models (.onnx preferred, else .pth)
    int8:       prefer *_int8.onnx when present
    early_exit: run the second model only if the first score is between band and 1-band (0 = always both)
    threshold:  real if score >= threshold; default from threshold.json in model_dir, else 0.5
    """

    def __init__(self, model_dir, int8=False, early_exit=0.1, threshold=None, detector=None):
        d = Path(model_dir)
        onnx = sorted(d.glob("*.onnx"))
        pick = [p for p in onnx if p.name.endswith("_int8.onnx") == int8] or [p for p in onnx if "_int8" not in p.name]
        files = pick or sorted(d.glob("*.pth"))
        if not files:
            raise FileNotFoundError(f"No .onnx or .pth models in {d}")
        self.models = [(_runner(p), parse_name(p.name), p.name) for p in files]
        self.backend = "onnx" if pick else "pytorch"
        tfile = d / "threshold.json"
        self.threshold = threshold if threshold is not None else (
            json.loads(tfile.read_text())["threshold"] if tfile.exists() else 0.5)
        self.early_exit = early_exit
        self.detector = detector or FaceDetector()
        self.name = d.name

    def score(self, img, bbox):
        scores, used = [], []
        for i, (run, info, name) in enumerate(self.models):
            crop = crop_face(img, bbox, info["scale"], info["w"], info["h"])
            p = float(run(crop.transpose(2, 0, 1)[None].astype(np.float32))[0, REAL])  # raw 0-255 BGR
            scores.append(p)
            used.append(name)
            if i == 0 and self.early_exit and not self.early_exit < p < 1 - self.early_exit:
                break
        return float(np.mean(scores)), used

    def predict(self, img):
        t0 = time.perf_counter()
        bbox, conf = self.detector.detect(img)
        t1 = time.perf_counter()
        if bbox is None:
            return {"verdict": "no face", "face_confidence": round(conf, 3), "ms_total": round((t1 - t0) * 1000, 2)}
        score, used = self.score(img, bbox)
        t2 = time.perf_counter()
        return {"verdict": "real" if score >= self.threshold else "fake", "real_score": round(score, 4),
                "threshold": round(self.threshold, 4), "bbox": bbox, "face_confidence": round(conf, 3),
                "models_run": len(used), "backend": self.backend,
                "ms_detect": round((t1 - t0) * 1000, 2), "ms_liveness": round((t2 - t1) * 1000, 2),
                "ms_total": round((t2 - t0) * 1000, 2)}

    @staticmethod
    def draw(img, r, label=""):
        if "bbox" not in r:
            cv2.putText(img, "No face found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return img
        x, y, w, h = r["bbox"]
        color = (0, 180, 0) if r["verdict"] == "real" else (0, 0, 230)
        thick, fs = max(2, img.shape[1] // 300), max(0.5, img.shape[1] / 900)
        text = f"{label + ': ' if label else ''}{r['verdict'].upper()} {r['real_score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        y0 = max(th + 6, y - 6)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        cv2.rectangle(img, (x, y0 - th - 6), (x + tw + 6, y0 + 4), color, -1)
        cv2.putText(img, text, (x + 3, y0), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)
        return img
