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
