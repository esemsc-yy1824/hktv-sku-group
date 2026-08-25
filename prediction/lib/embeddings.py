# -*- coding: utf-8 -*-
"""Offline semantic embeddings with a content-addressed local cache.

The API key is read from ``OPENAI_API_KEY`` or the repository ``.env`` file and
is never written to logs or cache metadata.  Embeddings are optional: callers
can fall back to the lexical method when the cache/key/network is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np

from core.norm import clean_title, nfkc
from core.variant import _residual_keep_variant


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 256
API_URL = "https://api.openai.com/v1/embeddings"


def _load_env_value(path, key):
    """Read one dotenv value without importing or echoing the whole file."""
    p = Path(path)
    if not p.exists():
        return None
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, value = line.split("=", 1)
        if k.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def api_key(env_path=".env"):
    return os.environ.get("OPENAI_API_KEY") or _load_env_value(env_path, "OPENAI_API_KEY")


def semantic_text(attr, summary_chars=220):
    """Make one pack-count-invariant text record for semantic comparison."""
    raw = attr["raw"]
    title = _residual_keep_variant(raw.get("sku_name_chi") or "")
    summary = clean_title(raw.get("summary_chi") or "")[:summary_chars]
    brand = nfkc(raw.get("brand_name_chi") or "")
    unit = "%g%s" % (attr["unit_value"], attr["unit_uom"]) \
        if attr.get("unit_value") is not None else "未知"
    return "商品：%s\n品牌：%s\n形态：%s\n单品规格：%s\n摘要：%s" % (
        title, brand, attr.get("form") or "UNKNOWN", unit, summary)


def _fingerprint(texts, model, dimensions):
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(str(dimensions).encode())
    for text in texts:
        h.update(b"\0")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def _request_embeddings(texts, key, model, dimensions, batch_size=512, timeout=120):
    vectors = []
    usage = {"prompt_tokens": 0, "total_tokens": 0, "requests": 0}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = json.dumps({
            "model": model,
            "input": batch,
            "dimensions": dimensions,
            "encoding_format": "float",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # API error bodies never need the request or API key to be useful.
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise RuntimeError("Embedding API HTTP %s: %s" % (exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Embedding API network error: %s" % exc.reason) from exc
        data = sorted(body["data"], key=lambda x: x["index"])
        if len(data) != len(batch):
            raise RuntimeError("Embedding API returned %d vectors for %d inputs" % (len(data), len(batch)))
        vectors.extend(x["embedding"] for x in data)
        u = body.get("usage") or {}
        usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        usage["total_tokens"] += int(u.get("total_tokens", 0))
        usage["requests"] += 1
    arr = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-12), usage


def load_or_create(rows, attrs, cache_dir=".cache/embeddings", env_path=".env",
                   model=DEFAULT_MODEL, dimensions=DEFAULT_DIMENSIONS,
                   allow_api=True):
    """Return ``(matrix, metadata)`` in row order, using cache when possible."""
    texts = [semantic_text(attrs[r["sku_id"]]) for r in rows]
    digest = _fingerprint(texts, model, dimensions)
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    npy = base / ("embeddings_%s.npy" % digest[:16])
    meta_path = base / ("embeddings_%s.json" % digest[:16])
    if npy.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(npy)
        if arr.shape == (len(rows), dimensions) and meta.get("fingerprint") == digest:
            meta["cache_hit"] = True
            return arr, meta
    if not allow_api:
        raise FileNotFoundError("No matching embedding cache and API calls are disabled")
    key = api_key(env_path)
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment and %s" % env_path)
    arr, usage = _request_embeddings(texts, key, model, dimensions)
    np.save(npy, arr)
    meta = {
        "fingerprint": digest,
        "model": model,
        "dimensions": dimensions,
        "n_rows": len(rows),
        "usage": usage,
        "cache_hit": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return arr, meta
