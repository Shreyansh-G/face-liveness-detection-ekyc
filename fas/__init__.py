"""
fas: face anti-spoofing for eKYC fraud detection, built on Minivision's MiniFASNet.

Modules
  network   MiniFASNet architecture (pretrained weights load unchanged)
  model     load models, training wrapper with FFT head, live-vs-attack loss
  augment   standard + fraud-aware transforms, FFT target
  data      CelebA-Spoof download -> face crops, PyTorch dataset
  metrics   bank metrics (APCER / BPCER / ACER), calibrated threshold, early exit
  engine    deployment: face detection -> crops -> score -> verdict (ONNX or PyTorch)
"""
from pathlib import Path

REAL = 1  # class index of a real face

# Minivision's pretrained models, keyed by crop setting ("<scale>_<h>x<w>")
PRETRAINED = {
    "2.7_80x80": "2.7_80x80_MiniFASNetV2.pth",
    "4_80x80": "4_0_0_80x80_MiniFASNetV1SE.pth",
}


def parse_name(name):
    """'2.7_80x80_MiniFASNetV2.pth' -> dict(h=80, w=80, arch='MiniFASNetV2', scale=2.7, patch='2.7_80x80')."""
    stem = Path(name).name.replace(".pth", "").replace(".onnx", "").replace("_int8", "")
    parts = stem.split("_")
    h, w = (int(v) for v in parts[-2].split("x"))
    scale = float(parts[0])
    return {"h": h, "w": w, "arch": parts[-1], "scale": scale, "patch": f"{scale:g}_{h}x{w}"}
