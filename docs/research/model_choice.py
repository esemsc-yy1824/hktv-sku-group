# -*- coding: utf-8 -*-
"""Scorer selection probe: why GBDT, and why 180 trees / depth 3.

The "model vs out-of-fold AUC" table in the deck comes from this script. The setup
matches `multipass.train()` exactly, so the numbers line up with the production model:

  · the training matrix uses the 27 method_5_hybrid_fusion features (22 lexical +
    semantic cosine + local TF-IDF cosine + 3 image features)
  · only pairs with `fold_of(a) == fold_of(b)` are kept -- the same keep mask train() uses
  · out-of-fold prediction: train on the other four folds, predict this fold, then compute
    AUC once over all out-of-fold predictions
  · positives are weighted (n-npos)/npos, exactly as in train()

⚠️ Depth 2 scores slightly higher out-of-fold than the depth 3 we run, but AUC is not the
   release criterion -- shipping is decided by the four `evaluate_all` metrics plus the
   adversarial regression. This script therefore prints both: whether the end-to-end
   metrics rise or fall on a depth-2 pipeline is answered by the lower table.

Usage (from the repo root):
    python3 docs/research/model_choice.py > docs/research/model_choice.txt
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core.dataset import load                                          # noqa: E402
from core.silver import (packcount_invariance_pairs, silver_negatives,  # noqa: E402
                         silver_pairs)
from paths import CACHE_DIR, IMAGE_CACHE_DIR, PROCESSED_DATASET_PATH, PROJECT_ROOT  # noqa: E402
from prediction import regression_checks as RC                         # noqa: E402
from prediction.engine import multipass as M                           # noqa: E402
from prediction.engine.training import N_FOLDS, build_context, fold_of, make_labels  # noqa: E402
from prediction.lib import candidates as C                             # noqa: E402
from prediction.lib import embeddings as E                             # noqa: E402
from prediction.lib import images as IMG                               # noqa: E402
from prediction.lib import local_vectors as LV                         # noqa: E402
from prediction.lib.model import GBDT, LogReg, auc                     # noqa: E402

THRESHOLD = 0.98            # method 5's production operating point


def build_matrix():
    """Reproduce the training matrix of multipass.train() (method 5 representation)."""
    rows = load(str(PROCESSED_DATASET_PATH))
    ctx = build_context(rows)
    attrs = ctx['attrs']

    emb, _ = E.load_or_create(rows, attrs, cache_dir=CACHE_DIR,
                              env_path=PROJECT_ROOT / '.env', allow_api=False)
    aux, _ = LV.build(rows, attrs)
    img, _ = IMG.build(rows, IMAGE_CACHE_DIR, allow_download=False)
    emb_by = {r['sku_id']: emb[i] for i, r in enumerate(rows)}
    aux_by = {r['sku_id']: aux[i] for i, r in enumerate(rows)}

    data, ean = make_labels(rows, ctx)
    X = np.asarray([M.feature_vector(attrs[a], attrs[b], ctx, emb_by, img, aux_by)
                    for a, b, _, _ in data], dtype=float)
    y = np.asarray([lab for _, _, lab, _ in data], dtype=float)
    npos = int(y.sum())
    w = np.where(y == 1, (len(y) - npos) / max(1, npos), 1.0)
    fa = np.asarray([fold_of(a, ean) for a, _, _, _ in data])
    fb = np.asarray([fold_of(b, ean) for _, b, _, _ in data])
    keep = fa == fb
    return (rows, ctx, emb, emb_by, aux_by, img,
            X[keep], y[keep], w[keep], fa[keep])


def out_of_fold_auc(make_model, X, y, w, folds):
    """Train on the other four folds, predict this one; AUC once over all OOF predictions."""
    pred = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        if y[tr].sum() < 1 or te.sum() == 0:
            continue
        pred[te] = make_model(k).fit(X[tr], y[tr], w[tr]).predict_proba(X[te])
    ok = ~np.isnan(pred)
    return auc(y[ok], pred[ok]), int(ok.sum())


def end_to_end(rows, ctx, emb, emb_by, aux_by, img, X, y, w, folds, depth, n_trees):
    """Swap the GBDT to the given depth, run the honest track, return the four metrics."""
    attrs = ctx['attrs']
    full = GBDT(n_trees=n_trees, lr=0.07, max_depth=depth, seed=503).fit(X, y, w)
    oof = {}
    for k in range(N_FOLDS):
        tr = folds != k
        if y[tr].sum() < 1:
            continue
        oof[k] = GBDT(n_trees=n_trees, lr=0.07, max_depth=depth,
                      seed=500 + k).fit(X[tr], y[tr], w[tr])
    ean = {r['sku_id']: attrs[r['sku_id']]['ean'] for r in rows}

    def model_for(a, b):
        ka = fold_of(a, ean) if ean.get(a) else None
        kb = fold_of(b, ean) if ean.get(b) else None
        if ka is not None and ka == kb and ka in oof:
            return oof[ka]
        return full

    info = C.generate(attrs, emb, include_ean=False, include_legacy_hint=False)
    scores = M.score_candidates(info, attrs, ctx, model_for, emb_by,
                                use_ean_must_link=False, image_by_sku=img,
                                auxiliary_embedding_by_sku=aux_by)
    must = {p for p, v in scores.items() if v == 1.0}
    groups = M.constrained_clusters(attrs, scores, THRESHOLD, ctx['sigs'],
                                    ctx['veto'], must)
    m = M.metrics(groups, rows, attrs, ctx['sigs'], info)

    # Production track: the barcode channel is on. This is the grouping actually shipped
    # and the basis metrics.csv uses for the adversarial regression (same as
    # ablation_form_ann.py).
    prod_info = C.generate(attrs, emb, include_ean=True, include_legacy_hint=True)
    prod_scores = M.score_candidates(prod_info, attrs, ctx, model_for, emb_by,
                                     use_ean_must_link=True, image_by_sku=img,
                                     auxiliary_embedding_by_sku=aux_by)
    prod_must = {p for p, v in prod_scores.items() if v == 1.0}
    prod_groups = M.constrained_clusters(attrs, prod_scores, THRESHOLD,
                                         ctx['sigs'], ctx['veto'], prod_must)
    gid = {mem['sku_id']: key for key, ms in prod_groups.items() for mem in ms}
    csv_rows = [{'sku_id': sid, 'group_id': gid[sid],
                 'sku_name': attrs[sid]['raw'].get('sku_name_chi') or '',
                 'brand_canonical': attrs[sid]['brand_canonical'] or ''}
                for sid in sorted(attrs)]
    checks = RC.check(csv_rows, gid)
    m['adversarial'] = (sum(bool(ok) for _, ok, _ in checks), len(checks))
    return m


def main():
    (rows, ctx, emb, emb_by, aux_by, img, X, y, w, folds) = build_matrix()

    print('=' * 78)
    print('scorer selection: out-of-fold AUC   (method 5, %d pairs / %d pos, %d EAN folds)'
          % (len(y), int(y.sum()), N_FOLDS))
    print('=' * 78)
    print()

    configs = [
        ('GBDT 180 trees/depth 2', lambda k: GBDT(n_trees=180, lr=0.07, max_depth=2, seed=500 + k)),
        ('GBDT 180 trees/depth 3 (prod)', lambda k: GBDT(n_trees=180, lr=0.07, max_depth=3, seed=500 + k)),
        ('GBDT 400 trees/depth 4', lambda k: GBDT(n_trees=400, lr=0.07, max_depth=4, seed=500 + k)),
        ('logistic regression', lambda k: LogReg(l2=1.0, iters=300, lr=0.5)),
    ]
    aucs = {}
    print('  %-30s %10s' % ('model', 'OOF AUC'))
    print('  ' + '-' * 42)
    for name, factory in configs:
        value, n = out_of_fold_auc(factory, X, y, w, folds)
        aucs[name] = value
        print('  %-30s %10.4f' % (name, value))
    print()
    print('  AUC = the chance the model scores a random positive above a random negative. 0.5 = coin flip.')
    print()

    print('-' * 78)
    print('end to end: swap the tree depth, the four honest-track metrics (threshold %.2f)' % THRESHOLD)
    print('-' * 78)
    print()
    print('  %-24s %8s %9s %9s %9s %7s' % ('config', 'complete', 'mismerge', 'precision',
                                           'pack inv', 'advers.'))
    for label, depth, n_trees in (('depth 2/180 trees', 2, 180),
                                  ('depth 3/180 trees (prod)', 3, 180),
                                  ('depth 4/400 trees', 4, 400)):
        m = end_to_end(rows, ctx, emb, emb_by, aux_by, img, X, y, w, folds,
                       depth, n_trees)
        print('  %-24s %7.1f%% %8.2f%% %8.1f%% %8.1f%% %5d/%d' % (
            label, m['silver_recall'] * 100,
            m['silver_negative_merge_rate'] * 100,
            m['case_control_precision'] * 100, m['pack_invariance'] * 100,
            m['adversarial'][0], m['adversarial'][1]))
    print()
    print('  How to read: AUC judges pairwise ranking only; the release gate is these four metrics plus the')
    print('  adversarial regression. If the two disagree, the lower table wins: we ship a pipeline, not a ranker.')


if __name__ == '__main__':
    main()
