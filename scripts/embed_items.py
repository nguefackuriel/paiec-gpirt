"""Embed item text (content + features) with a small multilingual encoder."""
import sys, time
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/proc/emb_e5small.npy"


def item_text(content, features, max_chars=2000):
    t = str(content)[:max_chars]
    if features:
        t = t + "\n" + str(features)[:300]
    return "query: " + t


if __name__ == "__main__":
    x = pd.read_parquet("data/proc/code_items.parquet").sort_values("i")
    assert (x.i.to_numpy() == np.arange(len(x))).all()
    texts = [item_text(c, f) for c, f in zip(x.content, x.item_features)]
    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 512
    t = time.time()
    order = np.argsort([len(s) for s in texts])  # length bucketing
    emb = model.encode([texts[j] for j in order], batch_size=128, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    out = np.empty_like(emb); out[order] = emb
    np.save(OUT, out.astype(np.float16))
    print(out.shape, time.time() - t)
