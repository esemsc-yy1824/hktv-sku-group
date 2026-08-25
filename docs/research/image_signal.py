# -*- coding: utf-8 -*-
"""Product-photo signal probe: descriptor comparison + component ablation of method 4.

Two questions:
  1. Which image descriptor can tell "same product" from "not the same product"?
     (Answer: naive similarity cannot, near-duplicate detection can.)
  2. Does method 4's gain over method 3 come from the image features or from the
     image recall channel?

Usage (from the repo root):
    python3 docs/research/image_signal.py > docs/research/image_signal.txt
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core.dataset import load                                           # noqa: E402
from core.silver import silver_negatives, silver_pairs                  # noqa: E402
from paths import (CACHE_DIR, IMAGE_CACHE_DIR, PROJECT_ROOT,            # noqa: E402
                   RAW_DATASET_PATH)
from prediction.engine import multipass as M                            # noqa: E402
from prediction.engine.training import build_context                    # noqa: E402
from prediction.lib import candidates as C                              # noqa: E402
from prediction.lib import embeddings as E                              # noqa: E402
from prediction.lib import images as IMG                                # noqa: E402


def hamming(a, b):
    return int(np.count_nonzero(a['dhash'].astype(bool) != b['dhash'].astype(bool)))


def auc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(labels)[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    return float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))


def main():
    rows = load(RAW_DATASET_PATH)
    ctx = build_context(rows)
    attrs = ctx['attrs']
    positives = set(silver_pairs(rows))
    negatives = set(silver_negatives(rows, positives, attrs))
    by_sku, image_meta = IMG.build(rows, IMAGE_CACHE_DIR, allow_download=False)

    print('=' * 78)
    print('product-photo signal probe')
    print('=' * 78)
    print('  SKUs with descriptor  %4d / %d   (download failed %d)'
          % (image_meta['n_with_descriptor'], len(rows), image_meta['n_failed']))
    both = lambda ps: [(a, b) for a, b in sorted(ps) if a in by_sku and b in by_sku]
    P, N = both(positives), both(negatives)
    print('  silver positive pairs %4d  (both sides have images %d)' % (len(positives), len(P)))
    print('  silver negative pairs %4d  (both sides have images %d)' % (len(negatives), len(N)))
    print()
    print('⚠️ a negative pair means same brand, same unit size, same pack count, disjoint barcodes --')
    print('   in practice, different scents of one product line that share a single package design.')

    print()
    print('--- 1. marginal discriminative power of three continuous descriptors ---')
    print('  %-24s %8s %9s %9s' % ('descriptor', 'AUC', 'pos mean', 'neg mean'))
    for name, key in (('mean-sub 32x32 grey', 'structure'),
                      ('hue histogram 32 bins', 'hue')):
        cp = [float(by_sku[a][key] @ by_sku[b][key]) for a, b in P]
        cn = [float(by_sku[a][key] @ by_sku[b][key]) for a, b in N]
        print('  %-24s %8.3f %9.3f %9.3f'
              % (name, auc(cp + cn, [1] * len(cp) + [0] * len(cn)),
                 float(np.mean(cp)), float(np.mean(cn))))
    dp = [1.0 - hamming(by_sku[a], by_sku[b]) / 64.0 for a, b in P]
    dn = [1.0 - hamming(by_sku[a], by_sku[b]) / 64.0 for a, b in N]
    print('  %-24s %8.3f %9.3f %9.3f'
          % ('dhash normalised sim', auc(dp + dn, [1] * len(dp) + [0] * len(dn)),
             float(np.mean(dp)), float(np.mean(dn))))
    print()
    print('  Conclusion: structural AUC is below 0.5 -- confusable pairs are structurally more')
    print('  alike than genuine same-product pairs, so "similar image => same product" is wrong here.')

    print()
    print('--- 2. dhash near-duplicates: precision by Hamming radius ---')
    print('  %-8s %18s %18s %10s' % ('radius', 'positive hits', 'negative FPs', 'prec ratio'))
    for radius in (0, 2, 4, 6, 8):
        hp = sum(1 for a, b in P if hamming(by_sku[a], by_sku[b]) <= radius)
        hn = sum(1 for a, b in N if hamming(by_sku[a], by_sku[b]) <= radius)
        ratio = (hp / len(P)) / max(1e-9, hn / len(N)) if hn else float('inf')
        print('  <=%-6d %8d (%5.1f%%) %8d (%5.1f%%) %9s'
              % (radius, hp, hp / len(P) * 100, hn, hn / len(N) * 100,
                 '%.1fx' % ratio if np.isfinite(ratio) else 'no FPs'))
    print()
    print('  We use radius 2: +4pp recall over radius 0 for 2/645 FPs; the ratio collapses from radius 4 on.')

    print()
    print('--- 3. component ablation of method 4 (honest track, threshold 0.98) ---')
    embeddings, _ = E.load_or_create(rows, attrs, cache_dir=CACHE_DIR,
                                     env_path=PROJECT_ROOT / '.env', allow_api=False)
    image_pairs = IMG.near_duplicate_pairs(by_sku)
    print('  image near-duplicate pairs: %d (before the hard-attribute check)' % len(image_pairs))
    print()
    print('  %-34s %8s %9s %9s %9s' % ('config', 'cand rec', 'silv rec', 'mismerge', 'precision'))
    for label, use_features, use_channel in (
            ('method 3 baseline (no images)', False, False),
            ('+ image recall channel only', False, True),
            ('+ image features only', True, False),
            ('+ both (method 4)', True, True)):
        img_by_sku = by_sku if use_features else None
        pairs = image_pairs if use_channel else None
        full, model_for, _ = M.train(rows, ctx, M._embedding_map(rows, embeddings), img_by_sku)
        info = C.generate(attrs, embeddings, include_ean=False, include_legacy_hint=False,
                          image_pairs=pairs)
        scores = M.score_candidates(info, attrs, ctx, model_for,
                                    M._embedding_map(rows, embeddings),
                                    use_ean_must_link=False, image_by_sku=img_by_sku)
        must = {p for p, v in scores.items() if v == 1.0}
        groups = M.constrained_clusters(attrs, scores, 0.98, ctx['sigs'], ctx['veto'], must)
        m = M.metrics(groups, rows, attrs, ctx['sigs'], info)
        print('  %-34s %7.1f%% %8.1f%% %8.1f%% %8.1f%%'
              % (label, m['candidate_recall'] * 100, m['silver_recall'] * 100,
                 m['silver_negative_merge_rate'] * 100, m['case_control_precision'] * 100))
    print()
    print('  How to read: the recall channel alone does not move candidate recall -- every new pair it')
    print('  finds is dropped by broad_compatible. The whole gain is the three image features in scoring.')


if __name__ == '__main__':
    main()
