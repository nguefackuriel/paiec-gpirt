"""Zero-shot LLM difficulty ratings vs measured within-benchmark item difficulty."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/proc/llm_diff_sample.parquet"
PROMPT = ("Below is one task from an AI evaluation benchmark. Estimate how likely a typical current AI system "
          "is to get this task right. Rate its difficulty from 1 (almost every system gets it right) to 9 "
          "(almost every system fails). Reply with a single digit.\n\nTask:\n{content}\n{features}")

item = pd.read_parquet('data/proc/item_nn.parquet')
items = pd.read_parquet('data/proc/code_items.parquet').set_index('i')
sel = item[item.ng >= 5]
sel = sel.sample(frac=1, random_state=0); sel = sel[sel.groupby('b').cumcount() < 40]; sel = sel[sel.b.isin(sel.b.drop_duplicates().sample(min(30, sel.b.nunique()), random_state=1))]
print(len(sel), 'items', sel.b.nunique(), 'benches', flush=True)
path = snapshot_download(MODEL)
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to('cuda').eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]
texts = []
for i in sel.index:
    c = str(items.at[i, 'content'])[:2000]; f = str(items.at[i, 'item_features'])[:300]
    msgs = [{"role": "user", "content": PROMPT.format(content=c, features=f)}]
    texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
order = np.argsort([len(t) for t in texts])
scores = np.zeros(len(texts))
t0 = time.time()
with torch.inference_mode():
    for s in range(0, len(texts), 8):
        idx = order[s:s + 8]
        enc = tok([texts[j] for j in idx], return_tensors='pt', padding=True, truncation=True, max_length=768).to('cuda')
        logits = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(logits[:, digit_ids], -1).cpu().numpy()
        scores[idx] = pr @ np.arange(1, 10)
        if s % 200 == 0: print(s, time.time() - t0, flush=True)
print('time', time.time() - t0, flush=True)
sel = sel.assign(llm=scores)
sel.to_parquet(OUT)
from scipy.stats import spearmanr
cors = pd.Series({b: spearmanr(d.llm, d.res).correlation for b, d in sel.groupby('b') if d.llm.std() > 0 and d.res.std() > 0})
print('within-bench spearman(llm, -res): mean', -cors.mean(), 'median', -cors.median(), 'n', cors.notna().sum())
print('frac benches with corr>0.1 (correct sign):', (-cors > 0.1).mean())
