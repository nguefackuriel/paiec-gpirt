"""Pre-download the HuggingFace repos listed in submission/models.txt (runtime is offline)."""
from pathlib import Path
from huggingface_hub import snapshot_download

for repo in Path(__file__).resolve().parent.parent.joinpath("submission/models.txt").read_text().split():
    print(repo, snapshot_download(repo), flush=True)
