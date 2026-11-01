"""
Showcase: upload a photo or use the webcam; every model set in app/models/ judges it side by side
(pretrained vs fine-tuned). Falls back to the pretrained models if app/models/ is empty.

  python app/app.py      -> http://127.0.0.1:7860
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402

from fas.engine import FaceDetector, LivenessEngine  # noqa: E402

SETS = {p.name: p for p in sorted((ROOT / "app" / "models").glob("*")) if any(p.glob("*.onnx"))} \
    or {"original": ROOT / "pretrained" / "anti_spoof"}
DETECTOR = FaceDetector()
ENGINES = {}


def check(image, int8, early_exit):
    if image is None:
        return [], []
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    shots, rows = [], []
    for name, path in SETS.items():
        key = (name, int8, early_exit)
        if key not in ENGINES:
            ENGINES[key] = LivenessEngine(path, int8=int8, early_exit=0.1 if early_exit else 0, detector=DETECTOR)
        e = ENGINES[key]
        r = e.predict(bgr)
        shots.append((cv2.cvtColor(e.draw(bgr.copy(), r, name), cv2.COLOR_BGR2RGB), name))
        rows.append([name, r["verdict"], r.get("real_score"), r.get("threshold"), r.get("models_run"), r["ms_total"], e.backend])
    return shots, rows


with gr.Blocks(title="Face Liveness Check") as demo:
    gr.Markdown("# Face Liveness Detection for eKYC\nIs this a real person or a fake (printed photo, screen replay, mask)? "
                "The pretrained and the fine-tuned model check the same photo.")
    with gr.Row():
        image = gr.Image(sources=["upload", "webcam"], type="numpy", label="Photo or webcam")
        gallery = gr.Gallery(label="Result per model", columns=2, height=420)
    with gr.Row():
        int8 = gr.Checkbox(False, label="INT8 models (smaller, less accurate)")
        early = gr.Checkbox(True, label="Early exit (skip 2nd model when the 1st is confident)")
        button = gr.Button("Check", variant="primary")
    table = gr.Dataframe(headers=["model", "verdict", "real score", "threshold", "models run", "ms", "backend"])
    button.click(check, [image, int8, early], [gallery, table])

if __name__ == "__main__":
    demo.launch()
