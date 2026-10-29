"""PAIEC submission: GP-IRT with per-benchmark Bayesian model averaging.

For each benchmark (anonymous benchmark_id), the revealed labels of *all*
subjects are pooled into a hierarchical logistic model

    logit P(correct) = m + a_subject + d_item

with a Gaussian-process prior on item effects whose kernel comes from text
embeddings of the items (intfloat/multilingual-e5-small, mean pooled). The
model is fit by Laplace approximation for a small grid of prior scales and the
predictions are averaged with weights proportional to the Laplace marginal
likelihood.

v4: an ensemble of two such models whose item kernels use (a) e5-small text
embeddings and (b) Qwen2.5-7B hidden states; both use a cross-benchmark prior
logit offset T * z_item from a ridge regression on Qwen hidden states trained
on the public training pool. If Qwen (or a GPU) is unavailable the predictor
falls back to the single e5 model without prior (v3 behaviour).

Every call is a pure function of (input, labeled); fits are memoized per
evidence state and benchmark purely for speed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import gpirt  # noqa: E402
import tokens as TK  # noqa: E402
from embed import Embedder, item_text  # noqa: E402

CFG = json.loads((_HERE / "config.json").read_text())
SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                "reasoning_effort", "harness_version", "subject_features_extra")


# ---------------------------------------------------------------------------
# Embedding model (pre-downloaded through models.txt; hub is offline)
# ---------------------------------------------------------------------------
def _declared_models() -> list:
    lines = (_HERE / "models.txt").read_text().splitlines()
    return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]


def _declared_model() -> str:
    return [m for m in _declared_models() if "e5" in m.lower()][0]


def _local_path(repo):
    from huggingface_hub import snapshot_download
    for cand in (os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(_HERE / ".hf_cache")):
        if cand and Path(cand).exists():
            try:
                return snapshot_download(repo, cache_dir=cand, local_files_only=True)
            except Exception:
                pass
    return snapshot_download(repo, local_files_only=True)


def _load_embedder():
    if os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "").strip().lower() in {"1", "true", "yes", "on"} \
            and os.environ.get("PAIEC_FORCE_EMBEDDER", "") != "1":
        return None
    try:
        from huggingface_hub import snapshot_download
        cache_dir = None
        for cand in (os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(_HERE / ".hf_cache")):
            if cand and Path(cand).exists():
                cache_dir = cand
                break
        try:
            path = snapshot_download(_declared_model(), cache_dir=cache_dir, local_files_only=True)
        except Exception:
            path = snapshot_download(_declared_model(), local_files_only=True)
        threads = max(1, (os.cpu_count() or 4) // 4)
        return Embedder(path, threads=threads)
    except Exception as exc:  # fall back to identity-only item kernel
        print(f"[paiec] embedder unavailable ({type(exc).__name__}: {exc}); using identity kernel", flush=True)
        return None


def _load_qwen():
    if EMBEDDER is None or not CFG.get("use_qwen", False):
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            print("[paiec] no CUDA: Qwen member disabled", flush=True)
            return None
        from qwenrep import QwenRep
        repo = [m for m in _declared_models() if "qwen" in m.lower()][0]
        return QwenRep(_local_path(repo), str(_HERE / "qwen_prior.npz"))
    except Exception as exc:
        print(f"[paiec] Qwen unavailable ({type(exc).__name__}: {exc}); using e5 only", flush=True)
        return None


EMBEDDER = _load_embedder()
QWEN = _load_qwen()
_EMB_CACHE: dict = {}
_QW_CACHE: dict = {}
_STATE: dict = {"key": None, "by_bench": None, "fits": {}}


def _subject_key(subject: dict) -> tuple:
    return tuple(str(subject.get(k, "") or "") for k in SUBJECT_KEYS)


def _item_key(item: dict) -> tuple:
    return (str(item.get("benchmark_id", "") or ""), str(item.get("item_content", "") or ""),
            str(item.get("item_features", "") or ""))


def _embed(keys: list) -> np.ndarray:
    missing = [k for k in keys if k not in _EMB_CACHE]
    if missing and EMBEDDER is not None:
        vecs = EMBEDDER.encode([item_text(k[1], k[2]) for k in missing], batch_size=32)
        for k, v in zip(missing, vecs):
            _EMB_CACHE[k] = v.astype(np.float32)
    if EMBEDDER is None:
        return None
    return np.stack([_EMB_CACHE[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)


def _qwen(keys: list):
    """(vectors, prior logits) from Qwen for item keys, memoized."""
    from qwenrep import qwen_text
    missing = [k for k in keys if k not in _QW_CACHE]
    if missing:
        X, z = QWEN.encode([qwen_text(k[1], k[2]) for k in missing])
        for k, x, zz in zip(missing, X, z):
            _QW_CACHE[k] = (x, float(zz))
    if not keys:
        return np.zeros((0, 1), np.float32), np.zeros(0)
    return np.stack([_QW_CACHE[k][0] for k in keys]), np.array([_QW_CACHE[k][1] for k in keys])


def _kernels(E_L, toks_L, E_t, toks_t, same_tL):
    """Item kernel: rho * embedding RBF-on-cosine + rho_f * feature-token cosine + rest identity."""
    rho, rho_f, ell = CFG["rho"], CFG.get("rho_f", 0.0), CFG["ell"]
    if E_L is None or E_t is None:
        rho = 0.0
    nL, nt = len(toks_L), len(toks_t)
    eyeL = np.eye(nL)
    K_LL = (1 - rho - rho_f) * eyeL
    K_tL = (1 - rho - rho_f) * same_tL
    k_tt = np.full(nt, 1.0 - rho_f)
    if rho > 0:
        K_LL = K_LL + rho * np.exp((E_L @ E_L.T - 1.0) / ell)
        K_tL = K_tL + rho * np.exp((E_t @ E_L.T - 1.0) / ell)
    if rho_f > 0:
        XL, XT = TK.token_matrix(toks_L, toks_t)
        if XL.shape[1]:
            K_LL = K_LL + rho_f * TK.cosine_kernel(XL, XL)
            K_tL = K_tL + rho_f * TK.cosine_kernel(XT, XL)
            has_t = XT.sum(1) > 0
            k_tt = k_tt + rho_f * has_t
            noL = XL.sum(1) == 0
            K_LL[noL, noL] += rho_f
            K_tL = K_tL + rho_f * same_tL * noL[None, :]
            k_tt = np.where(has_t, k_tt, k_tt + rho_f)
        else:
            K_LL = K_LL + rho_f * eyeL
            K_tL = K_tL + rho_f * same_tL
            k_tt = k_tt + rho_f
    rho_le = CFG.get("rho_le", 0.0)
    if rho_le > 0 and E_L is not None and E_t is not None and nL >= 3:
        # Bayesian linear regression of item effects on benchmark-centered embeddings
        mu = E_L.mean(0)
        ZL, ZT = E_L - mu, E_t - mu
        q = max(1e-9, float((ZL ** 2).sum(1).mean()))
        s = 1.0 - rho_le
        K_LL = s * K_LL + rho_le * (ZL @ ZL.T) / q
        K_tL = s * K_tL + rho_le * (ZT @ ZL.T) / q
        k_tt = s * k_tt + rho_le * (ZT ** 2).sum(1) / q
    return K_LL, K_tL, k_tt


def _evidence_state(labeled):
    key = (len(labeled), repr(labeled[-1]) if labeled else "")
    if _STATE["key"] != key:
        by_bench: dict = {}
        for entry in labeled:
            (subject, item), label = entry[0], entry[1]
            by_bench.setdefault(str(item.get("benchmark_id", "") or ""), []).append(
                (_subject_key(subject), _item_key(item), float(label)))
        _STATE.update(key=key, by_bench=by_bench, fits={})
    return _STATE


class _BenchFit:
    """All grid fits for one benchmark under one evidence state, for one ensemble member.

    source: "e5" or "qwen" (item vectors for the kernel); use_prior adds T * z offsets.
    """

    def __init__(self, rows, source="e5", use_prior=False):
        self.source, self.use_prior = source, use_prior
        subj = sorted({r[0] for r in rows})
        items = sorted({r[1] for r in rows})
        self.sidx = {s: j for j, s in enumerate(subj)}
        self.iidx = {it: j for j, it in enumerate(items)}
        self.items = items
        self.n_s = len(subj) + 1  # last index = unseen subject (prior only)
        ls = np.array([self.sidx[r[0]] for r in rows]); ll = np.array([self.iidx[r[1]] for r in rows])
        y = np.array([r[2] for r in rows])
        self.E, zL = self._vectors(items)
        self.toks = [TK.feature_tokens(it[2]) for it in items]
        K, _, _ = _kernels(self.E, self.toks, self.E[:0] if self.E is not None else None, [], np.zeros((0, len(items))))
        base = CFG.get("T", 0.0) * zL[ll] if use_prior else np.zeros(len(y))
        self.nlab = np.bincount(ls, minlength=self.n_s)
        # BMA over the components of the benchmark-shift prior (centre, scale, weight),
        # subject and item scales; the component centre is added to the logit offset
        fits, lps, mus = [], [], []
        for mu, sm, lw in _m_mix():
            for sa in CFG["s_a_grid"]:
                for sd in CFG["s_d_grid"]:
                    fits.append(gpirt.fit(ls, ll, y, base + mu, K, self.n_s, sm, sa, sd))
                    lps.append(lw); mus.append(mu)
        w = gpirt.bma_weights(fits, np.array(lps))
        keep = w >= 1e-4
        self.fits = [f for f, k in zip(fits, keep) if k]
        self.mus = [m for m, k in zip(mus, keep) if k]
        self.w = w[keep] / w[keep].sum()

    def _vectors(self, keys):
        if self.source == "qwen":
            X, z = _qwen(keys)
            return X, z
        E = _embed(keys)
        z = _qwen(keys)[1] if self.use_prior else np.zeros(len(keys))
        return E, z

    def predict(self, skey, ikey) -> float:
        s = self.sidx.get(skey, self.n_s - 1)
        Et, zt = self._vectors([ikey])
        same = np.array([[1.0 if ikey == it else 0.0 for it in self.items]])
        _, K_tL, k_tt = _kernels(self.E, self.toks, Et, [TK.feature_tokens(ikey[2])], same)
        tb = CFG.get("T", 0.0) * zt if self.use_prior else np.zeros(1)
        # more confident predictions once the target subject has more than n_lo labels on this benchmark
        vs = _var_scale(self.nlab[s] if s < len(self.nlab) else 0)
        p = 0.0
        for w, f, m0 in zip(self.w, self.fits, self.mus):
            mu, var = gpirt.predict_moments(f, np.array([s]), K_tL, k_tt, tb + m0)
            p += w * float(gpirt.predictive(mu, var, vs)[0])
        return p


def _m_mix():
    """Prior on the benchmark shift as (centre, scale, log weight) components.
    Without "m_mix" in the config: the v7 scale grid centred at zero, equal weights."""
    if CFG.get("m_mix"):
        pis = np.array([c[2] for c in CFG["m_mix"]], float)
        return [(float(c[0]), float(c[1]), float(np.log(p / pis.sum()))) for c, p in zip(CFG["m_mix"], pis)]
    return [(0.0, float(sm), 0.0) for sm in CFG.get("s_m_grid", [CFG["s_m"]])]


def _var_scale(n_labels: int) -> float:
    lo = CFG.get("var_scale_lo", CFG.get("var_scale", 1.0))
    hi = CFG.get("var_scale_hi", CFG.get("var_scale", 1.0))
    return lo if n_labels <= CFG.get("n_lo", 3) else hi


def _members():
    if QWEN is None:
        return [("e5", False)]
    return [("e5", True), ("qwen", True)]


def _predict(input, labeled) -> float:
    subject, item = input
    labeled = labeled or []
    st = _evidence_state(labeled)
    bench = str(item.get("benchmark_id", "") or "")
    rows = st["by_bench"].get(bench)
    members = _members()
    if not rows:
        if QWEN is None and not CFG.get("m_mix"):
            return 0.5
        # no labels for this benchmark: prior only (as in the simulator)
        zt = _qwen([_item_key(item)])[1] if QWEN is not None else np.zeros(1)
        if not CFG.get("m_mix"):
            return float(gpirt.predictive(CFG.get("T", 0.0) * zt, np.array([CFG["s_m"] ** 2 + 1.0]),
                                          _var_scale(0))[0])
        return float(sum(np.exp(lw) * gpirt.predictive(CFG.get("T", 0.0) * zt + mu, np.array([sm ** 2 + 1.0]),
                                                       _var_scale(0))[0] for mu, sm, lw in _m_mix()))
    ps = []
    for source, use_prior in members:
        key = (bench, source)
        fit = st["fits"].get(key)
        if fit is None:
            fit = st["fits"][key] = _BenchFit(rows, source, use_prior)
        ps.append(fit.predict(_subject_key(subject), _item_key(item)))
    return float(np.mean(ps))


def _predict_safe(input, labeled) -> float:
    global QWEN
    try:
        return _predict(input, labeled)
    except Exception as exc:
        if QWEN is None:
            raise
        # Qwen failed at runtime (e.g. GPU OOM): disable it and continue with the e5-only model
        print(f"[paiec] Qwen member disabled after error ({type(exc).__name__}: {exc})", flush=True)
        QWEN = None
        _STATE.update(key=None, by_bench=None, fits={})
        return _predict(input, labeled)


def predict(input: list, labeled: list | None = None) -> float:
    try:
        p = float(_predict_safe(input, labeled))
        if not np.isfinite(p):
            return 0.5
        return min(1.0, max(0.0, p))
    except Exception as exc:  # never fail the run
        print(f"[paiec] predict fallback ({type(exc).__name__}: {exc})", flush=True)
        return 0.5
