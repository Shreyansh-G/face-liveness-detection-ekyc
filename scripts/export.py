"""
Export a model pair for the app -> app/models/<name>/ (ONNX float32 + INT8 + threshold.json),
then compare PyTorch vs ONNX vs INT8 on speed, size and accuracy -> results/deploy/<name>.json

  python scripts/export.py --models pretrained/anti_spoof --name original
  python scripts/export.py --models results/runs/ours/models --name ours --metrics results/eval/ours/metrics.json
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fas import REAL  # noqa: E402
from fas.data import list_images  # noqa: E402
from fas.metrics import at_threshold, threshold_free  # noqa: E402
from fas.model import load, to_tensor  # noqa: E402


def to_onnx(model, info, path):
    net = torch.nn.Sequential(model, torch.nn.Softmax(dim=1)).eval()
    x = torch.rand(1, 3, info["h"], info["w"]) * 255
    kw = dict(input_names=["input"], output_names=["probs"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(net, x, str(path), dynamo=False, **kw)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(net, x, str(path), **kw)


def to_int8(fp32, int8, imgs):
    """Static INT8 quantization calibrated on real face crops."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter({"input": to_tensor(imgs[i:i + 16]).numpy()} for i in range(0, len(imgs), 16))

        def get_next(self):
            return next(self.it, None)

    pre = fp32.with_name(fp32.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()


def runners(kind, files):
    if kind == "pytorch":
        ms = [load(p)[0] for p in files]
        return [lambda x, m=m: torch.softmax(m(torch.from_numpy(x)), 1).numpy() for m in ms]
    import onnxruntime as ort

    ss = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in files]
    return [lambda x, s=s: s.run(None, {"input": x})[0] for s in ss]


ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="folder with the two .pth files")
ap.add_argument("--name", required=True, help="model set name in the app, e.g. original / ours")
ap.add_argument("--out", default="app/models")
ap.add_argument("--calib", default="data/val", help="crops for INT8 calibration")
ap.add_argument("--test", default="data/test", help="crops for the accuracy check")
ap.add_argument("--metrics", help="metrics.json whose bank threshold the app should use")
ap.add_argument("--report", default="results/deploy", help="where the speed/size/accuracy comparison is saved")
a = ap.parse_args()
torch.set_grad_enabled(False)  # inference only

out = Path(a.out) / a.name
out.mkdir(parents=True, exist_ok=True)
pths = sorted(Path(a.models).glob("*.pth"))
files = {"pytorch": pths, "onnx_fp32": [], "onnx_int8": []}
for p in pths:
    model, info = load(p)
    fp32 = out / f"{p.stem}.onnx"
    to_onnx(model, info, fp32)
    files["onnx_fp32"].append(fp32)
    calib = list_images(Path(a.calib) / info["patch"], limit=300)
    if calib:
        to_int8(fp32, out / f"{p.stem}_int8.onnx", [cv2.imread(f) for f, _ in calib])
        files["onnx_int8"].append(out / f"{p.stem}_int8.onnx")
threshold = json.loads(Path(a.metrics).read_text()).get("bank_threshold", {}).get("threshold", 0.5) if a.metrics else 0.5
(out / "threshold.json").write_text(json.dumps({"threshold": threshold}))
print(f"saved {out} (threshold {threshold:.3f}{', INT8 skipped: no calibration crops' if not files['onnx_int8'] else ''})")

infos = [load(p)[1] for p in pths]
test = {i["patch"]: list_images(Path(a.test) / i["patch"]) for i in infos}
report = {}
for kind, fs in files.items():
    if len(fs) != len(pths):
        continue
    run = runners(kind, fs)
    xs = [(np.random.rand(1, 3, i["h"], i["w"]) * 255).astype(np.float32) for i in infos]
    for _ in range(30):
        [r(x) for r, x in zip(run, xs)]
    t = []
    for _ in range(300):
        t0 = time.perf_counter()
        [r(x) for r, x in zip(run, xs)]
        t.append((time.perf_counter() - t0) * 1000)
    rep = {"ms_median": float(np.median(t)), "ms_p95": float(np.percentile(t, 95)),
           "size_mb": round(sum(f.stat().st_size for f in fs) / 1e6, 2)}
    if all(test.values()):
        per = [np.concatenate([r(to_tensor([cv2.imread(f) for f, _ in test[i["patch"]][j:j + 256]]).numpy())[:, REAL]
                               for j in range(0, len(test[i["patch"]]), 256)]) for r, i in zip(run, infos)]
        y = np.array([l for _, l in test[infos[0]["patch"]]])
        s = np.mean(per, 0)
        rep.update(ACER=at_threshold(s, y)["ACER"], AUC=threshold_free(s, y)["AUC"])
    report[kind] = rep
    acc = f" | ACER {rep['ACER']:.2%} AUC {rep['AUC']:.4f}" if "AUC" in rep else ""
    print(f"{kind:9s}: {rep['ms_median']:.2f} ms/face (p95 {rep['ms_p95']:.2f}) | {rep['size_mb']:.2f} MB{acc}")
Path(a.report).mkdir(parents=True, exist_ok=True)
(Path(a.report) / f"{a.name}.json").write_text(json.dumps(report, indent=2))
