"""
Publish the demo to a free Hugging Face Space. Uploads only what the app needs (fas/, app/app.py,
app/models/, the face detector) plus a Space card and slim requirements, in one commit.

  hf auth login                                          # once, with a "write" token
  python scripts/deploy_space.py --repo <hf-user>/ekyc-liveness --github <repo url>
"""
import argparse
from pathlib import Path

import gradio
from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = "numpy\nopencv-python-headless>=4.8,<5\nonnxruntime>=1.17\n"  # gradio comes with the Space


def card(github):
    return f"""---
title: Face Liveness Detection eKYC
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.11"
app_file: app/app.py
license: apache-2.0
short_description: Face liveness (anti-spoofing) check for banking eKYC
---

Upload a selfie or use the webcam: the pretrained and the fine-tuned MiniFASNet judge the same face side by side.
0.87M parameters, 3.5 MB, about 9 ms per face on CPU.{f" Code and results: {github}" if github else ""}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--github", default="", help="GitHub link shown on the Space page")
    ap.add_argument("--dry_run", action="store_true", help="list the files without uploading")
    a = ap.parse_args()

    files = [*sorted(ROOT.glob("fas/*.py")), ROOT / "app" / "app.py", *sorted((ROOT / "app" / "models").rglob("*.*")),
             *sorted((ROOT / "pretrained" / "detector").glob("*")), ROOT / "LICENSE", ROOT / "NOTICE"]
    ops = [CommitOperationAdd(f.relative_to(ROOT).as_posix(), str(f)) for f in files if f.is_file()]
    ops += [CommitOperationAdd("README.md", card(a.github).encode()),
            CommitOperationAdd("requirements.txt", REQUIREMENTS.encode())]
    print("\n".join(op.path_in_repo for op in ops))
    if a.dry_run:
        raise SystemExit
    api = HfApi()
    api.create_repo(a.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.create_commit(a.repo, ops, commit_message="Deploy eKYC liveness demo", repo_type="space")
    print(f"Building at https://huggingface.co/spaces/{a.repo} (takes 2-3 minutes)")
