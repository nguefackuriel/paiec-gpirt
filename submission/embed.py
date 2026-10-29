"""Item text embeddings with plain transformers (mean pooling + L2 norm).

Identical code path offline (training tables) and at runtime (submission).
"""
from __future__ import annotations

import numpy as np

MAX_CHARS = 2000
MAX_FEAT_CHARS = 300
MAX_TOKENS = 512


def item_text(content: str, features: str) -> str:
    t = str(content or "")[:MAX_CHARS]
    if features:
        t = t + "\n" + str(features)[:MAX_FEAT_CHARS]
    return "query: " + t


class Embedder:
    def __init__(self, path: str, device: str | None = None, threads: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()

    def encode(self, texts, batch_size=64) -> np.ndarray:
        torch = self.torch
        out = []
        order = np.argsort([len(t) for t in texts])
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                batch = [texts[j] for j in order[s:s + batch_size]]
                enc = self.tok(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                v = torch.nn.functional.normalize(v.float(), dim=-1)
                out.append(v.cpu().numpy())
        emb = np.concatenate(out) if out else np.zeros((0, 384), np.float32)
        res = np.empty_like(emb); res[order] = emb
        return res
