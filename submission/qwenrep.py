"""Qwen2.5-7B-Instruct item representation and cross-benchmark accuracy prior.

h = mean-pooled hidden_states[layer] over "Task:\n<content>\n<features>" (<= 512 tokens)
x = L2-normalize((h - raw_mean) / raw_std)        -> item vector for the GP kernel
z = c + (x - mu) @ beta                           -> prior logit (ridge trained on the
                                                     measurement-db training pool)
Same code path as scripts/llm_embed.py + scripts/build_qwen_prior.py.
"""
from __future__ import annotations

import numpy as np

MAXTOK = 512


def qwen_text(content: str, features: str) -> str:
    return "Task:\n" + str(content or "")[:4000] + ("\n" + str(features)[:300] if features else "")


class QwenRep:
    def __init__(self, path: str, prior_npz: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        P = np.load(prior_npz)
        self.layer = int(P["layer"])
        self.raw_mean, self.raw_std = P["raw_mean"].astype(np.float32), P["raw_std"].astype(np.float32)
        self.mu, self.beta, self.c = P["mu"].astype(np.float32), P["beta"].astype(np.float32), float(P["c"])

    def encode(self, texts, batch_size=8):
        """Returns (x: n x d normalized vectors, z: n prior logits)."""
        torch = self.torch
        H = np.zeros((len(texts), self.raw_mean.shape[0]), np.float32)
        order = np.argsort([len(t) for t in texts])
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                               max_length=MAXTOK).to(self.device)
                hs = self.model.model(**enc, output_hidden_states=True).hidden_states[self.layer]
                m = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
                v = (hs.float() * m).sum(1) / m.sum(1)
                # match the offline pipeline, which stored hidden states as float16
                H[idx] = v.cpu().numpy().astype(np.float16).astype(np.float32)
        X = (H - self.raw_mean) / self.raw_std
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        # offline vectors were stored as float16 too
        X = X.astype(np.float16).astype(np.float32)
        z = self.c + (X - self.mu) @ self.beta
        return X, z
