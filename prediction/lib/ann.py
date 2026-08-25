# -*- coding: utf-8 -*-
"""IVF approximate nearest-neighbour search, an optional alternative to exact top-k.

**Off by default**: on 1,472 products exact search is actually faster (the fixed cost of
building the k-means index outweighs the brute-force scan), and exact search loses no
recall. This module exists for scale -- exact search is O(n²), which will not run on
3 million rows.

Measured (docs/research/ablation_form_ann.txt): switching to IVF left candidate recall,
completeness, false-merge rate, accuracy and pack-count invariance **unchanged to the last
digit**, even though it retrieves only 92.4% of the exact top-20. The neighbours it misses
would have been dropped by the hard-attribute check or fallen below threshold anyway.

Rows compared per query (nprobe × n/nlist, nlist=√n):

    n = 1,472      exact 1,471 rows    IVF ~   309 rows    79% saved
    n = 100,000    exact 99,999 rows   IVF ~ 2,531 rows    97% saved
    n = 3,000,000  exact 3M rows       IVF ~ 13,856 rows   99.5% saved

Production should swap in FAISS / HNSW or a hosted vector search service; the hand-rolled
numpy here avoids a dependency while proving that the semantic_neighbors interface of
candidates.generate is sufficient.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TOP_K = 20
DEFAULT_NPROBE = 8
KMEANS_ITERS = 12


def top_k(matrix, ids, k=DEFAULT_TOP_K, nprobe=DEFAULT_NPROBE, nlist=None, seed=0):
    """Return ``(sku_id -> [(neighbour sku_id, cosine similarity), ...], meta)``.

    Row vectors must already be L2-normalised so that the dot product equals cosine
    similarity -- both the embeddings and local_vectors modules guarantee this.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    n = matrix.shape[0]
    if n < 2:
        return {sid: [] for sid in ids}, {'nlist': 0, 'nprobe': nprobe,
                                          'avg_compared': 0.0,
                                          'exhaustive_compared': max(0, n - 1)}
    nlist = int(nlist or max(1, int(np.sqrt(n))))
    nlist = min(nlist, n)

    # ---- Build the index: spherical k-means (Lloyd iterations) ----
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(n, nlist, replace=False)].copy()
    for _ in range(KMEANS_ITERS):
        assign = np.argmax(matrix @ centroids.T, axis=1)
        for c in range(nlist):
            members = matrix[assign == c]
            if len(members):
                centre = members.sum(0)
                norm = float(np.linalg.norm(centre))
                if norm > 1e-9:
                    centroids[c] = centre / norm
    assign = np.argmax(matrix @ centroids.T, axis=1)
    buckets = [np.where(assign == c)[0] for c in range(nlist)]

    # ---- Query: probe only the nprobe nearest buckets ----
    sim_to_centroid = matrix @ centroids.T
    out, probed = {}, 0
    for i in range(n):
        probe = np.argsort(-sim_to_centroid[i])[:nprobe]
        candidates = np.concatenate([buckets[c] for c in probe])
        candidates = candidates[candidates != i]
        probed += candidates.size
        if candidates.size == 0:
            out[ids[i]] = []
            continue
        scores = matrix[candidates] @ matrix[i]
        kk = min(k, candidates.size)
        top = (np.argpartition(-scores, kk - 1)[:kk]
               if candidates.size > kk else np.arange(candidates.size))
        ranked = sorted(((float(scores[j]), ids[candidates[j]]) for j in top),
                        key=lambda t: (-t[0], t[1]))
        out[ids[i]] = [(sid, score) for score, sid in ranked]

    meta = {
        'kind': 'ivf_flat_numpy',
        'nlist': nlist,
        'nprobe': nprobe,
        'top_k': k,
        'avg_compared': round(probed / n, 1),
        'exhaustive_compared': n - 1,
        'saved_fraction': round(1 - (probed / n) / max(1, n - 1), 4),
    }
    return out, meta


def exact_top_k(matrix, ids, k=DEFAULT_TOP_K):
    """Exact top-k. Used only to measure the ANN's recall loss, not on the production path."""
    matrix = np.asarray(matrix, dtype=np.float32)
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -1.0)
    out = {}
    for i in range(len(ids)):
        idx = np.argpartition(sim[i], -k)[-k:]
        ranked = sorted(((float(sim[i, j]), ids[j]) for j in idx),
                        key=lambda t: (-t[0], t[1]))
        out[ids[i]] = [(sid, score) for score, sid in ranked]
    return out


def recall_against_exact(approx, exact):
    """Fraction of the exact top-k that the ANN actually retrieves."""
    hit = total = 0
    for sid, gold in exact.items():
        want = {other for other, _ in gold}
        got = {other for other, _ in approx.get(sid, [])}
        hit += len(want & got)
        total += len(want)
    return hit / max(1, total)
