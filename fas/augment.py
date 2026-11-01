"""
Training transforms on BGR uint8 face crops.

  standard_augment()  the original Minivision recipe (baseline)
  fft_target()        Fourier-spectrum target for the training-only FFT head
  FraudAwareAugment   fraud-aware transforms (this project's contribution), two groups:

1. Capture degradations -> applied to REAL and FAKE faces alike.
   Real eKYC selfies arrive compressed, dark, blurry or from cheap cameras.
   Applying these to both classes stops the model from using "image quality"
   as a shortcut for "fake".

2. Synthetic attacks -> applied to REAL faces only, which are then labelled FAKE.
   They imitate the traces a screen or a printout leaves on a re-captured face:
   moire interference, screen sub-pixel grid, glare, print halftone and colour loss.
   This adds attack variety without collecting new attack data.
"""
import math
import random

import cv2
import numpy as np


# ---------------------------------------------------------------- helpers
def _clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def _grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return yy, xx


# ---------------------------------------------------------------- 1. capture degradations
def jpeg(img, q=None):
    q = q or random.randint(25, 75)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def sensor_noise(img):
    sigma = random.uniform(2, 10)
    return _clip(img.astype(np.float32) + np.random.normal(0, sigma, img.shape))


def low_light(img):
    gain = random.uniform(0.35, 0.7)
    gamma = random.uniform(1.1, 1.6)
    x = (img.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    return _clip(x + np.random.normal(0, random.uniform(2, 6), img.shape))


def blur(img):
    if random.random() < 0.5:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    k = random.choice([3, 5, 7])  # motion blur, random direction
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def low_res(img):
    h, w = img.shape[:2]
    f = random.uniform(0.4, 0.7)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def white_balance(img):
    x = img.astype(np.float32)
    x[..., 0] *= random.uniform(0.85, 1.15)  # blue
    x[..., 2] *= random.uniform(0.85, 1.15)  # red
    return _clip(x)


DEGRADATIONS = {
    "jpeg": jpeg,
    "noise": sensor_noise,
    "low_light": low_light,
    "blur": blur,
    "low_res": low_res,
    "white_balance": white_balance,
}


# ---------------------------------------------------------------- 2. synthetic attacks
def moire(img):
    """Two slightly different gratings beat against each other, like a camera filming a screen."""
    h, w = img.shape[:2]
    yy, xx = _grid(h, w)
    theta = random.uniform(0, np.pi)
    f1 = random.uniform(0.18, 0.45)
    f2 = f1 * random.uniform(0.9, 1.1)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = xx * np.cos(theta + random.uniform(0.02, 0.15)) + yy * np.sin(theta + 0.05)
    pattern = np.sin(2 * np.pi * f1 * u) * np.sin(2 * np.pi * f2 * v)
    amp = random.uniform(5, 16)
    x = img.astype(np.float32)
    for c, phase in enumerate(np.random.uniform(0, np.pi, 3)):  # coloured fringes
        x[..., c] += amp * pattern * np.cos(phase)
    return _clip(x)


def screen_grid(img):
    """RGB sub-pixel stripes plus the slightly blue, low-contrast look of a phone/laptop screen."""
    h, w = img.shape[:2]
    period = random.choice([2, 3, 4])
    x = img.astype(np.float32)
    strength = random.uniform(0.06, 0.16)
    cols = np.arange(w) % period
    for c in range(3):
        mask = (cols == (c % period)).astype(np.float32)[None, :]
        x[..., c] *= 1 - strength + 2 * strength * mask
    x[..., 0] *= random.uniform(1.0, 1.1)  # bluish tint
    x = x * random.uniform(0.85, 0.95) + random.uniform(5, 20)  # lifted blacks
    return _clip(x)


def glare(img):
    """A bright reflection spot, as on glossy photo paper or a screen."""
    h, w = img.shape[:2]
    yy, xx = _grid(h, w)
    cx, cy = random.uniform(0, w), random.uniform(0, h)
    sx, sy = random.uniform(0.1, 0.35) * w, random.uniform(0.1, 0.35) * h
    spot = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    x = img.astype(np.float32) + random.uniform(50, 130) * spot[..., None]
    return _clip(x)


def halftone_print(img):
    """Printed photo: fewer colours, paper tint, dot pattern, softer contrast."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * random.uniform(0.5, 0.85)  # weaker colours
    x = x * random.uniform(0.75, 0.9) + random.uniform(15, 35)  # paper can't do pure black
    x[..., 0] *= random.uniform(0.88, 0.97)  # warm paper tint (less blue)
    x = cv2.GaussianBlur(x, (3, 3), 0.6)  # ink spread, before the dots so they stay visible
    yy, xx = _grid(h, w)
    period = random.uniform(2.5, 4.5)
    dots = np.cos(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period)
    x += random.uniform(4, 10) * dots[..., None]
    return _clip(x)


SYNTHETIC_ATTACKS = {
    "moire": moire,
    "screen_grid": screen_grid,
    "glare": glare,
    "halftone_print": halftone_print,
}


# ---------------------------------------------------------------- pipeline
class FraudAwareAugment:
    """
    Call with (img, label) -> (img, label). label: 1 = real, 0 = fake.

    p_synthetic: chance a REAL face becomes a synthetic attack (label flips to 0)
    p_degrade:   chance each capture degradation is applied (to any face)
    disable:     names to switch off, for the ablation study
    """

    def __init__(self, p_synthetic=0.2, p_degrade=0.15, disable=()):
        self.p_synthetic = p_synthetic
        self.p_degrade = p_degrade
        self.attacks = {k: v for k, v in SYNTHETIC_ATTACKS.items() if k not in disable}
        self.degradations = {k: v for k, v in DEGRADATIONS.items() if k not in disable}

    def __call__(self, img, label):
        if label == 1 and self.attacks and random.random() < self.p_synthetic:
            names = random.sample(list(self.attacks), k=random.choice([1, 1, 2]))
            for n in names:
                img = self.attacks[n](img)
            label = 0
        for fn in self.degradations.values():
            if random.random() < self.p_degrade:
                img = fn(img)
        return img, label


def standard_augment(img):
    """The original recipe (Minivision's dataset_loader.py), re-implemented with OpenCV:
    random resized crop (area 0.9-1.1), colour jitter (0.4, 0.4, 0.4, hue 0.1),
    rotation +-10 degrees, horizontal flip."""
    h, w = img.shape[:2]
    area = min(random.uniform(0.9, 1.1), 1.0) * h * w
    ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
    cw = min(w, int(round(math.sqrt(area * ratio))))
    ch = min(h, int(round(math.sqrt(area / ratio))))
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    x = img.astype(np.float32)
    ops = ["brightness", "contrast", "saturation", "hue"]
    random.shuffle(ops)
    for op in ops:
        if op == "brightness":
            x *= random.uniform(0.6, 1.4)
        elif op == "contrast":
            mean = cv2.cvtColor(_clip(x), cv2.COLOR_BGR2GRAY).mean()
            x = mean + (x - mean) * random.uniform(0.6, 1.4)
        elif op == "saturation":
            gray = cv2.cvtColor(_clip(x), cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
            x = gray + (x - gray) * random.uniform(0.6, 1.4)
        else:
            hsv = cv2.cvtColor(_clip(x), cv2.COLOR_BGR2HSV)
            shift = int(round(random.uniform(-0.1, 0.1) * 180))
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
            x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    img = _clip(x)

    rot = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-10, 10), 1.0)
    img = cv2.warpAffine(img, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    return img


def fft_target(img_bgr, out_h, out_w):
    """Log-magnitude Fourier spectrum, normalised like the original project (src/data_io/dataset_folder.py)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)  # 2x faster than np.fft, same result
    m = np.log(np.fft.fftshift(cv2.magnitude(f[..., 0], f[..., 1])) + 1)
    m = (m - m.min() + 1) / (m.max() - m.min() + 1)
    return cv2.resize(m, (out_w, out_h)).astype(np.float32)
