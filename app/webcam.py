"""
Real-time liveness from the webcam (press q to quit). The score is smoothed over recent frames
so the verdict doesn't flicker, like an eKYC capture screen deciding over a short burst.

  python app/webcam.py
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from fas.engine import LivenessEngine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=str(ROOT / "app" / "models" / "fine-tuned"))
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--int8", action="store_true", help="INT8 ONNX: 2.6x smaller, less accurate")
ap.add_argument("--smooth", type=float, default=0.3, help="weight of the newest frame (1 = no smoothing)")
a = ap.parse_args()

engine = LivenessEngine(a.models, int8=a.int8)
cap = cv2.VideoCapture(a.camera)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {a.camera}")
avg, last = None, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = engine.predict(frame)
    if "real_score" in r:
        avg = r["real_score"] if avg is None else a.smooth * r["real_score"] + (1 - a.smooth) * avg
        r.update(real_score=avg, verdict="real" if avg >= engine.threshold else "fake")
    else:
        avg = None
    now = time.perf_counter()
    fps, last = 1 / max(now - last, 1e-6), now
    engine.draw(frame, r)
    cv2.putText(frame, f"{engine.name} | {fps:.0f} FPS | {r['ms_total']:.1f} ms", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Liveness (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
