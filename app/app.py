"""
Web demo: take a selfie or upload a photo; every model set in app/models/ (pretrained and fine-tuned)
checks the same face side by side.

  streamlit run app/app.py      -> http://localhost:8501
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from fas.engine import FaceDetector, LivenessEngine  # noqa: E402


@st.cache_resource
def load_engines(early_exit):
    detector = FaceDetector()
    sets = sorted((p for p in (ROOT / "app" / "models").glob("*") if any(p.glob("*.onnx"))), reverse=True)  # pretrained first
    return [LivenessEngine(p, early_exit=0.1 if early_exit else 0, detector=detector) for p in sets]


st.set_page_config(page_title="Face Liveness Detection for eKYC", page_icon="🛡️", layout="wide")
st.title("Face Liveness Detection for eKYC")
st.write("Is this a real person or a fake (printed photo, screen replay, mask)? "
         "The pretrained and the fine-tuned model check the same photo.")
early =st.sidebar.checkbox("Early exit (skip the 2nd model when the 1st is confident)", True)
st.sidebar.markdown("0.87M parameters, 3.5 MB, about 9 ms per face on CPU. "
                    "[Code and results](https://github.com/Shreyansh-G/face-liveness-detection-ekyc)")

source = st.radio("Photo source", ["Camera", "Upload"], horizontal=True)
photo = st.camera_input("Take a selfie") if source == "Camera" else \
    st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if photo:
    img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read this image.")
        st.stop()
    engines = load_engines(early)
    rows = []
    for col, e in zip(st.columns(len(engines)), engines):
        r = e.predict(img)
        col.image(e.draw(img.copy(), r, e.name), channels="BGR", caption=e.name, width="stretch")
        rows.append({"model": e.name, "verdict": r["verdict"], "real score": r.get("real_score"),
                     "threshold": r.get("threshold"), "models run": r.get("models_run"), "ms": r["ms_total"],
                     "backend": e.backend})
    st.dataframe(rows, hide_index=True)
