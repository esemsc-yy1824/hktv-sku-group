# -*- coding: utf-8 -*-
"""Dependency-free local TF-IDF vectors for private-data retrieval.

This is not a semantic foundation-model embedding.  It is a deterministic,
hashed character/word TF-IDF representation used as the privacy-preserving
retrieval baseline when external API use is not authorised.
"""
from __future__ import annotations

import collections
import hashlib
import math
import re

import numpy as np

from .embeddings import semantic_text
from core.norm import fold


def _tokens(text):
    text = text.lower()
    out = []
    for word in re.findall(r'[a-z][a-z0-9]{1,}', text):
        out.append('w:' + word)
    compact = fold(text)
    for n in (2, 3, 4):
        out.extend('c%d:%s' % (n, compact[i:i + n])
                   for i in range(max(0, len(compact) - n + 1)))
    return out


def _bucket(token, dimensions):
    raw = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
    value = int.from_bytes(raw, 'little')
    return value % dimensions, 1.0 if (value >> 63) == 0 else -1.0


def build(rows, attrs, dimensions=4096):
    docs = [_tokens(semantic_text(attrs[r['sku_id']])) for r in rows]
    bucket_docs = []
    df = collections.Counter()
    for toks in docs:
        counts = collections.Counter(_bucket(t, dimensions)[0] for t in toks)
        bucket_docs.append((toks, counts))
        df.update(counts)
    n = len(docs)
    matrix = np.zeros((n, dimensions), dtype=np.float32)
    for i, (toks, counts) in enumerate(bucket_docs):
        signed = collections.Counter()
        for token in toks:
            bucket, sign = _bucket(token, dimensions)
            signed[bucket] += sign
        for bucket, value in signed.items():
            tf = math.copysign(math.log1p(abs(value)), value)
            idf = math.log((n + 1) / (df[bucket] + 1)) + 1.0
            matrix[i, bucket] = tf * idf
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12), {
        'kind': 'local_hashed_char_word_tfidf',
        'dimensions': dimensions, 'n_rows': n,
    }
