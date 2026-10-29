"""Download measurement-db tables (no traces/raw) pinned to one revision."""
import concurrent.futures as cf, json, os, sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/mdb"
TABLES = ("benchmarks", "subjects", "items", "responses", "response")
api = HfApi()
rev = api.dataset_info(REPO).sha
dirs = sorted({e.path for e in api.list_repo_tree(REPO, repo_type="dataset", revision=rev) if "." not in e.path})
print(rev, len(dirs), "dirs", flush=True)

def one(d):
    got = []
    try:
        names = {e.path.split("/")[-1] for e in api.list_repo_tree(REPO, path_in_repo=f"{d}/formatted_tables", repo_type="dataset", revision=rev)}
        sub = "formatted_tables/"
    except Exception:
        names = {e.path.split("/")[-1] for e in api.list_repo_tree(REPO, path_in_repo=d, repo_type="dataset", revision=rev)}
        sub = ""
    for t in TABLES:
        if f"{t}.parquet" in names:
            hf_hub_download(REPO, f"{d}/{sub}{t}.parquet", repo_type="dataset", revision=rev, local_dir=OUT); got.append(t)
    for f in ("metadata.yaml",):
        try: hf_hub_download(REPO, f"{d}/{f}", repo_type="dataset", revision=rev, local_dir=OUT)
        except Exception: pass
    return d, got

res = {}
with cf.ThreadPoolExecutor(16) as ex:
    for i, (d, got) in enumerate(ex.map(one, dirs), 1):
        res[d] = got
        if i % 20 == 0: print(i, flush=True)
json.dump({"revision": rev, "dirs": res}, open(os.path.join(OUT, "download_manifest.json"), "w"), indent=1)
print("DONE")
