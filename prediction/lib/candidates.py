# -*- coding: utf-8 -*-
"""Multi-pass candidate generation for entity resolution.

Candidate generation is a union of independent recall channels.  Unlike a
single blocking key, one missing or noisy attribute cannot permanently hide a
pair from the scorer.
"""
from __future__ import annotations

import collections
import itertools

import numpy as np

from core.extract import origin_compatible, unit_compatible
from .features import packfree_text
from core.norm import fold


def _pair(a, b):
    return (a, b) if a < b else (b, a)


def form_compatible(a, b):
    fa, fb = a.get("form"), b.get("form")
    return fa == "UNKNOWN" or fb == "UNKNOWN" or fa == fb


def unit_candidate_compatible(a, b, allow_one_unknown=True):
    va, vb = a.get("unit_value"), b.get("unit_value")
    if va is None or vb is None:
        return allow_one_unknown or (va is None and vb is None)
    return unit_compatible((va, a.get("unit_uom")), (vb, b.get("unit_uom")))


def broad_compatible(a, b):
    """Cheap filters safe enough for recall; strict veto happens after scoring."""
    return (form_compatible(a, b)
            and unit_candidate_compatible(a, b, allow_one_unknown=True)
            and origin_compatible(a.get("origin", frozenset()), b.get("origin", frozenset())))


def packfree_key(a):
    return fold(packfree_text(a["raw"].get("sku_name_chi") or ""))


def generate(attrs, embeddings=None, embedding_top_k=20, embedding_min_cos=0.40,
             include_ean=True, include_legacy_hint=True, semantic_neighbors=None,
             embedding_batch_size=512, image_pairs=None):
    """Return ``pair -> {sources, embedding_cosine}``.

    Passes:
      * exact EAN (production-only strong evidence; disabled for honest silver eval)
      * legacy GS1/text identity + exact unit (kept only as a recall hint)
      * normalized brand + tolerance-aware unit
      * exact pack-count-free title across brand spellings
      * semantic embedding top-k under broad hard-attribute compatibility
      * near-duplicate product photos (``image_pairs``), an optional channel for
        listings whose text is too sparse for any other pass.  Measured as zero
        gain on the soap sample -- every new pair it surfaced was dropped by the
        hard-attribute check -- so ``multipass`` does not pass it.  See
        docs/research/image_signal.txt section 3.
    """
    info = {}

    def add(x, y, source, cosine=None):
        if x == y:
            return
        k = _pair(x, y)
        rec = info.setdefault(k, {"sources": set(), "embedding_cosine": None})
        rec["sources"].add(source)
        if cosine is not None:
            old = rec["embedding_cosine"]
            rec["embedding_cosine"] = float(cosine if old is None else max(old, cosine))

    values = list(attrs.values())

    if include_ean:
        by_ean = collections.defaultdict(list)
        for a in values:
            for e in a.get("ean", ()):
                by_ean[e].append(a["sku_id"])
        for xs in by_ean.values():
            for x, y in itertools.combinations(sorted(set(xs)), 2):
                add(x, y, "exact_ean")

    # Preserve the old block as one recall channel, but never use the inferred
    # fixed-length prefix as a hard company identity.
    if include_legacy_hint:
        legacy = collections.defaultdict(list)
        for a in values:
            legacy[(a.get("brand_id") or "<NB>", a.get("unit_value"), a.get("unit_uom"))].append(a)
        for xs in legacy.values():
            for a, b in itertools.combinations(xs, 2):
                if broad_compatible(a, b):
                    add(a["sku_id"], b["sku_id"], "legacy_block")

    by_brand = collections.defaultdict(list)
    for a in values:
        if a.get("brand_key"):
            by_brand[a["brand_key"]].append(a)
    for xs in by_brand.values():
        for a, b in itertools.combinations(xs, 2):
            if broad_compatible(a, b):
                add(a["sku_id"], b["sku_id"], "brand_unit")

    by_title = collections.defaultdict(list)
    for a in values:
        key = packfree_key(a)
        if key:
            by_title[key].append(a)
    for xs in by_title.values():
        for a, b in itertools.combinations(xs, 2):
            if broad_compatible(a, b):
                add(a["sku_id"], b["sku_id"], "packfree_exact")

    if semantic_neighbors is not None:
        # Production hook: feed top-k results from FAISS/HNSW/a vector service.
        # Keeping ANN outside this module avoids an all-pairs scan at 3M scale.
        for sid, neighbours in semantic_neighbors.items():
            a = attrs[sid]
            for other, score in neighbours:
                b = attrs[other]
                if score >= embedding_min_cos and broad_compatible(a, b):
                    add(sid, other, "semantic_topk", score)
    elif embeddings is not None and len(values) > 1:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape[0] != len(values):
            raise ValueError("embedding rows do not match attributes")
        k = min(int(embedding_top_k), len(values) - 1)
        # POC exact search in row batches: O(n²) compute, but only O(batch*n)
        # memory. At production scale use semantic_neighbors from an ANN index.
        for start in range(0, len(values), embedding_batch_size):
            stop = min(len(values), start + embedding_batch_size)
            sim = matrix[start:stop] @ matrix.T
            for local_i, a in enumerate(values[start:stop]):
                i = start + local_i
                sim[local_i, i] = -1.0
                idx = np.argpartition(sim[local_i], -k)[-k:]
                idx = sorted(idx, key=lambda j: (-float(sim[local_i, j]),
                                                  values[j]["sku_id"]))
                for j in idx:
                    score = float(sim[local_i, j])
                    if score < embedding_min_cos:
                        continue
                    b = values[j]
                    if broad_compatible(a, b):
                        add(a["sku_id"], b["sku_id"], "semantic_topk", score)

    if image_pairs:
        # The same manufacturer photo reused by different merchants -- the only channel
        # left when the text fields are missing.
        for x, y in image_pairs:
            a, b = attrs.get(x), attrs.get(y)
            if a is not None and b is not None and broad_compatible(a, b):
                add(x, y, "image_near_dup")

    return info


def source_counts(candidate_info):
    counts = collections.Counter()
    for rec in candidate_info.values():
        counts.update(rec["sources"])
    return counts
