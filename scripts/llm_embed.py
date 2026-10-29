"""Mean-pooled hidden states of a causal LLM for every item (several layers).

Usage: llm_embed.py <hf_repo> <tag> [layers comma-separated as fractions, e.g. 0.5,0.75,1.0]
Writes data/proc/emb_<tag>_L<layer>.npy (float16, one row per item code).
"""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

repo, tag = sys.argv[1], sys.argv[2]
fracs = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.5,0.75,1.0").split(",")]
MAXTOK = 512
x = pd.read_parquet("data/proc/code_items.parquet").sort_values("i")
texts = [("Task:\n" + str(c)[:4000] + ("\n" + str(f)[:300] if f else "")) for c, f in zip(x.content, x.item_features)]
path = snapshot_download(repo)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
nl = model.config.num_hidden_layers
layers = sorted({max(1, min(nl, int(round(f * nl)))) for f in fracs})
print("layers", layers, "of", nl, flush=True)
H = model.config.hidden_size
out = {l: np.zeros((len(texts), H), np.float16) for l in layers}
order = np.argsort([len(t) for t in texts])
t0 = time.time(); bs = 32
with torch.inference_mode():
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXTOK).to("cuda")
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        for l in layers:
            v = (hs[l].float() * m).sum(1) / m.sum(1)
            out[l][idx] = v.cpu().numpy().astype(np.float16)
        if (s // bs) % 200 == 0:
            print(s, f"{time.time() - t0:.0f}s", flush=True)
for l in layers:
    np.save(f"data/proc/emb_{tag}_L{l}.npy", out[l])
print("done", time.time() - t0, flush=True)
