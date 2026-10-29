"""Runtime-safe feature extraction from the subject / item dicts.

Only the fields available at test time are used: the eight subject fields and
item_content, item_features, interactors. Benchmark IDs are never used as
features.
"""
from __future__ import annotations

import math
import re

import numpy as np

PROVIDERS = ["anthropic", "openai", "google", "meta", "alibaba", "deepseek", "mistral", "unknown",
             "(agent scaffold)", "opengvlab", "zhipu", "microsoft", "xai", "moonshot", "cohere", "amazon"]
EFFORTS = ["", "none", "minimal", "low", "medium", "high", "xhigh", "extended"]


def _date_num(s: str) -> float:
    m = re.match(r"(\d{4})(?:-(\d{1,2}))?", s or "")
    if not m:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) - 1) / 12 if m.group(2) else int(m.group(1)) + 0.5


def _size_b(name: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]\b", name or "")
    return math.log10(float(m[-1])) if m else np.nan


def subject_vector(subj: dict, ability: float = np.nan) -> np.ndarray:
    name = subj.get("normalized_name", "") or ""
    prov = (subj.get("provider", "") or "").strip().lower()
    eff = (subj.get("reasoning_effort", "") or "").strip().lower()
    extra = subj.get("subject_features_extra", "") or ""
    low = name.lower()
    v = [
        float(name != ""), float(prov != ""),
        _date_num(subj.get("release_date", "")), _date_num(subj.get("access_date", "")),
        float(subj.get("harness", "") != ""), float(subj.get("harness_version", "") != ""),
        EFFORTS.index(eff) if eff in EFFORTS else len(EFFORTS),
        _size_b(name),
        float(any(t in low for t in ("mini", "small", "nano", "lite", "flash", "haiku", "tiny"))),
        float(any(t in low for t in ("pro", "opus", "large", "ultra", "max"))),
        float(any(t in low for t in ("o1", "o3", "o4", "r1", "reason", "think", "gpt-5", "gpt 5"))),
        float("human" in extra.lower() or "participant" in extra.lower()),
        float("fine" in extra.lower() and "tun" in extra.lower()),
        len(extra) / 100.0,
        ability,
    ]
    v += [float(prov == p) for p in PROVIDERS]
    return np.array(v, np.float32)


SUBJ_DIM = 15 + len(PROVIDERS)


def item_stats(content: str, features: str, interactors: str) -> np.ndarray:
    c = content or ""
    n = len(c)
    letters = sum(ch.isalpha() for ch in c[:5000])
    ascii_ = sum(ord(ch) < 128 for ch in c[:5000])
    head = c[:5000]
    return np.array([
        math.log1p(n), math.log1p(len(features or "")), float(bool(interactors)),
        ascii_ / max(1, len(head)), letters / max(1, len(head)),
        float(c.lstrip().startswith("{")), float('"options"' in head or "choices" in head.lower()),
        head.count("\n") / max(1, len(head)) * 100, sum(ch.isdigit() for ch in head) / max(1, len(head)),
        float("```" in head or "def " in head), float("http" in head),
    ], np.float32)


ITEM_STATS_DIM = 11


def item_stats_v2(content: str, features: str) -> np.ndarray:
    """Richer generic, benchmark-agnostic text statistics (standardized later within a benchmark)."""
    import re as _re
    c = content or ""
    head = c[:20000]
    words = head.split()
    nw = len(words)
    lines = head.count("\n") + 1
    nums = _re.findall(r"\d+(?:\.\d+)?", head)
    upper = sum(ch.isupper() for ch in head)
    punct = sum(ch in "?!.,;:" for ch in head)
    maths = sum(ch in "$\\^_{}=+-*/<>" for ch in head)
    return np.array([
        math.log1p(len(c)), math.log1p(nw), math.log1p(lines), math.log1p(len(nums)),
        np.mean([len(w) for w in words]) if words else 0.0,
        len(set(w.lower() for w in words)) / max(1, nw),
        upper / max(1, len(head)), punct / max(1, len(head)), maths / max(1, len(head)),
        math.log1p(max([len(x) for x in nums], default=0)),
        float("?" in head[-300:]), head.count("(") / max(1, len(head)) * 100,
        float(bool(_re.search(r"\b[A-D][\).:]", head))), math.log1p(head.count('","')),
        sum(ord(ch) > 127 for ch in head) / max(1, len(head)),
        math.log1p(len(features or "")), math.log1p(len((features or "").split(";"))),
        float("```" in head or "def " in head or "class " in head), float("http" in head),
        math.log1p(head.count("{")),
    ], np.float32)
