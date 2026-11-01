"""Load models, training wrapper with the Fourier-spectrum (FFT) head, and the loss."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fas import REAL, parse_name
from fas.network import build


def load_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()})
    return model


def load(path):
    """Model file -> (eval-mode model, info dict from its name)."""
    info = parse_name(path)
    return load_weights(build(info["arch"], info["h"], info["w"]), path).eval(), info


class FASModel(nn.Module):
    """Backbone + FFT head that predicts the image's Fourier spectrum (Minivision's auxiliary task).
    The head only exists during training; the saved/deployed model is the backbone alone."""

    def __init__(self, arch, use_fft=True):
        super().__init__()
        self.backbone = build(arch)
        self.ft = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1), nn.BatchNorm2d(1), nn.ReLU(True),
        ) if use_fft else None

    def forward(self, x):
        logits, feat = self.backbone(x, return_feat=True)
        return logits, (self.ft(feat) if self.training and self.ft is not None else None)


def live_vs_attack_loss(logits, labels):
    """Real = class 1, attack = class 0 OR 2. The pretrained head splits attacks into two classes;
    this keeps that split instead of forcing every attack into class 0."""
    logp = F.log_softmax(logits.float(), dim=1)
    attack = torch.logsumexp(logp[:, [0, 2]], dim=1)
    y = labels.float()
    return -(y * logp[:, REAL] + (1 - y) * attack).mean()


def to_tensor(imgs):
    """BGR uint8 crops -> float tensor N,3,H,W in raw 0-255 (what these models expect, no /255)."""
    return torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))).float()
