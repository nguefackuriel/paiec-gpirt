"""Categorical tokens from 'key=value;key=value' feature strings."""
from __future__ import annotations

import numpy as np

MAX_VALUE_CHARS = 60


def feature_tokens(s: str) -> frozenset:
    out = set()
    for part in str(s or "").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip().lower(), v.strip().lower()
        if not k or not v or len(v) > MAX_VALUE_CHARS:
            continue
        out.add(k + "=" + v)
    return frozenset(out)


def token_matrix(token_sets_L, token_sets_T=None, min_count=2):
    """Binary indicator matrices over tokens seen >= min_count times in L and not in all of L."""
    from collections import Counter
    cnt = Counter(t for ts in token_sets_L for t in ts)
    nL = len(token_sets_L)
    vocab = sorted(t for t, c in cnt.items() if c >= min_count and c < nL)
    idx = {t: j for j, t in enumerate(vocab)}

    def mat(sets):
        X = np.zeros((len(sets), len(vocab)), np.float32)
        for r, ts in enumerate(sets):
            for t in ts:
                j = idx.get(t)
                if j is not None:
                    X[r, j] = 1.0
        return X
    XL = mat(token_sets_L)
    XT = mat(token_sets_T) if token_sets_T is not None else None
    return XL, XT


def cosine_kernel(XA, XB):
    na = np.sqrt(XA.sum(1)); nb = np.sqrt(XB.sum(1))
    K = XA @ XB.T
    den = np.outer(na, nb)
    return np.divide(K, den, out=np.zeros_like(K), where=den > 0)


SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version")


def subject_tokens(subject: dict) -> frozenset:
    out = set()
    for k in SUBJECT_FIELDS:
        v = str(subject.get(k, "") or "").strip().lower()
        if v and len(v) <= MAX_VALUE_CHARS:
            out.add("@" + k + "=" + v)
    return frozenset(out | feature_tokens(subject.get("subject_features_extra", "")))


def numeric_features(s: str) -> dict:
    """Numeric values of 'key=value' metadata (e.g. level=3, problem_id=42), keyed by lower-case key."""
    out = {}
    for part in str(s or "").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        try:
            x = float(v)
        except ValueError:
            continue
        if np.isfinite(x):
            out[k] = x
    return out


def numeric_matrix(dicts_L, dicts_T, min_frac=0.5):
    """Standardized (on L) matrix of numeric keys present in >= min_frac of L with non-constant values."""
    from collections import Counter
    nL = len(dicts_L)
    cnt = Counter(k for d in dicts_L for k in d)
    keys = []
    for k, c in cnt.items():
        if c >= max(2, min_frac * nL):
            vals = np.array([d[k] for d in dicts_L if k in d])
            if vals.std() > 0:
                keys.append(k)
    keys.sort()

    def mat(ds):
        X = np.zeros((len(ds), len(keys)))
        for r, d in enumerate(ds):
            for j, k in enumerate(keys):
                X[r, j] = d.get(k, np.nan)
        return X
    XL, XT = mat(dicts_L), mat(dicts_T)
    if not keys:
        return XL, XT
    mu = np.nanmean(XL, 0); sd = np.nanstd(XL, 0) + 1e-9
    XL = np.nan_to_num((XL - mu) / sd); XT = np.clip(np.nan_to_num((XT - mu) / sd), -5, 5)
    # also a rank-like transform via log for heavy-tailed ids
    return XL, XT
