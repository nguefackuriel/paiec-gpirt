"""Build dist/<name>.zip from submission/ (runtime files only, at the zip root)."""
import sys, zipfile
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "paiec_gpirt"
src = Path(__file__).resolve().parent.parent / "submission"
out = src.parent / "dist" / f"{name}.zip"
out.parent.mkdir(exist_ok=True)
files = ["model.py", "gpirt.py", "embed.py", "tokens.py", "qwenrep.py", "qwen_prior.npz", "config.json", "models.txt", "labeling.py", "prediction_config.json"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if (src / f).exists():
            z.write(src / f, f)
print(out, [i.filename for i in zipfile.ZipFile(out).infolist()])
