"""
Zip code + pretrained weights for upload as a Kaggle Dataset -> dist/fas_project.zip

  python scripts/pack_kaggle.py
"""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
zpath = root / "dist" / "fas_project.zip"
zpath.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for item in ["fas", "scripts", "pretrained", "notebooks", "requirements.txt", "requirements-train.txt", "README.md", "LICENSE", "NOTICE"]:
        p = root / item
        for f in [p] if p.is_file() else sorted(p.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(root))
print(f"{zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
