"""Embed all training items with paiec.embed (same code path as runtime)."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from paiec.embed import Embedder, item_text
from huggingface_hub import snapshot_download

repo = sys.argv[1]; out = sys.argv[2]
x = pd.read_parquet("data/proc/code_items.parquet").sort_values("i")
texts = [item_text(c, f) for c, f in zip(x.content, x.item_features)]
path = snapshot_download(repo)
emb = Embedder(path, device="cuda")
t = time.time()
E = emb.encode(texts, batch_size=128)
np.save(out, E.astype(np.float16))
print(E.shape, time.time() - t)
