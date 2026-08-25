# -*- coding: utf-8 -*-
"""Ablation: turn product_form off × swap exact retrieval for ANN.

Two questions:
  1. product_form is a hand-written priority dictionary and will not stay maintainable
     across the few hundred categories of the full site. With that signal removed
     entirely, how far do the four metrics drop?
  2. Exact top-k is O(n^2) and does not finish on 3 million rows. Once ANN approximate
     retrieval takes over, how much candidate recall and final metric quality is lost?

Four cells: current config / no form / ANN / both. The base method is pinned to
method_5_hybrid_fusion and the threshold to 0.98 (its production operating point), so the
numbers are directly comparable with metrics.csv.

⚠️ The "no form" cell here is **slightly different** from the production `--no-form` switch:
  probe: form is removed from the rule, feature and label layers, but the embedding text
        keeps its form line. The embedding cache therefore still hits and no paid API
        call is needed. What this isolates is precisely the contribution of form as a
        decision signal.
  --no-form: form is forced to UNKNOWN everywhere, including in the embedding text.
        The first --no-form run of a semantic method therefore misses the cache and
        costs one API call.
  Verified end to end with the real switch on the purely local method 2 (no API):
  completeness 54.9% -> 57.0%.

Usage (from the repo root):
    python3 docs/research/ablation_form_ann.py > docs/research/ablation_form_ann.txt
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core import extract as X                                          # noqa: E402
from core import group as G                                            # noqa: E402
from core.dataset import load                                          # noqa: E402
from paths import (CACHE_DIR, IMAGE_CACHE_DIR, PROJECT_ROOT,           # noqa: E402
                   RAW_DATASET_PATH)
from prediction import regression_checks as RC                         # noqa: E402
from prediction.engine import multipass as M                           # noqa: E402
from prediction.engine.training import build_context                   # noqa: E402
from prediction.lib import candidates as C                             # noqa: E402
from prediction.lib import embeddings as E                             # noqa: E402
from prediction.lib import images as IMG                               # noqa: E402
from prediction.lib import local_vectors as LV                         # noqa: E402
from prediction.lib import ann as ANN                                  # noqa: E402

THRESHOLD = 0.98
TOP_K = 20
NPROBE = 8


# -------------------------------------------------------- disable product_form
_REAL_FORM = X.extract_form
_REAL_SEMANTIC_TEXT = E.semantic_text
REAL_FORM_BY_SKU = {}


def _blank_form(row):
    return 'UNKNOWN', 0.0, None


def _semantic_text_keep_form(attr, summary_chars=220):
    """Keep the real form in the embedding text even while form is disabled.

    Why: the embedding cache fingerprint covers the form line, so blanking it would
    invalidate the cache and force another round of paid API calls. Keeping it also
    sharpens the ablation -- what gets isolated is exactly the contribution of
    "form as a decision signal (rules / features / labels)", not its value as text.
    """
    saved = attr.get('form')
    attr['form'] = REAL_FORM_BY_SKU.get(attr['sku_id'], saved)
    try:
        return _REAL_SEMANTIC_TEXT(attr, summary_chars)
    finally:
        attr['form'] = saved


def set_form_enabled(enabled):
    """The product_form on/off switch.

    core.group and local_vectors did `from ... import` at module load time, so every
    binding has to be swapped or they keep calling the original functions.
    """
    X.extract_form = _REAL_FORM if enabled else _blank_form
    G.extract_form = _REAL_FORM if enabled else _blank_form
    fn = _REAL_SEMANTIC_TEXT if enabled else _semantic_text_keep_form
    E.semantic_text = fn
    LV.semantic_text = fn


# ------------------------------------------------------------- run one cell
def run_cell(rows, form_enabled, use_ann):
    set_form_enabled(form_enabled)
    ctx = build_context(rows)
    attrs = ctx['attrs']
    ids = [r['sku_id'] for r in rows]

    emb, _ = E.load_or_create(rows, attrs, cache_dir=CACHE_DIR,
                              env_path=PROJECT_ROOT / '.env', allow_api=False)
    aux, _ = LV.build(rows, attrs)
    img, _ = IMG.build(rows, IMAGE_CACHE_DIR, allow_download=False)
    ebs, abs_ = M._embedding_map(rows, emb), M._embedding_map(rows, aux)

    matrix = np.asarray(emb, dtype=np.float32)
    ann_meta, ann_fidelity = {}, None
    # Time only the retrieval the pipeline really performs; the fidelity check is excluded.
    if use_ann:
        t0 = time.perf_counter()
        neighbours, ann_meta = ANN.top_k(matrix, ids, k=TOP_K, nprobe=NPROBE)
        retrieval_seconds = time.perf_counter() - t0
        ann_fidelity = ANN.recall_against_exact(
            neighbours, ANN.exact_top_k(matrix, ids, k=TOP_K))
    else:
        neighbours = None
        t0 = time.perf_counter()
        ANN.exact_top_k(matrix, ids, k=TOP_K)   # normally hidden in C.generate; timed here
        retrieval_seconds = time.perf_counter() - t0

    full, model_for, _ = M.train(rows, ctx, ebs, img, abs_)
    info = C.generate(attrs, None if use_ann else emb,
                      include_ean=False, include_legacy_hint=False,
                      semantic_neighbors=neighbours)
    scores = M.score_candidates(info, attrs, ctx, model_for, ebs,
                                use_ean_must_link=False, image_by_sku=img,
                                auxiliary_embedding_by_sku=abs_)
    must = {p for p, v in scores.items() if v == 1.0}
    groups = M.constrained_clusters(attrs, scores, THRESHOLD, ctx['sigs'], ctx['veto'], must)
    m = M.metrics(groups, rows, attrs, ctx['sigs'], info)

    # Production track: the barcode channel is on. This is the grouping actually shipped
    # and the basis metrics.csv uses for the adversarial regression and the release gate.
    prod_info = C.generate(attrs, None if use_ann else emb,
                           include_ean=True, include_legacy_hint=True,
                           semantic_neighbors=neighbours)
    prod_scores = M.score_candidates(prod_info, attrs, ctx, model_for, ebs,
                                     use_ean_must_link=True, image_by_sku=img,
                                     auxiliary_embedding_by_sku=abs_)
    prod_must = {p for p, v in prod_scores.items() if v == 1.0}
    prod_groups = M.constrained_clusters(attrs, prod_scores, THRESHOLD,
                                         ctx['sigs'], ctx['veto'], prod_must)

    def adversarial(gs):
        gid = {}
        for key, members in gs.items():
            for mem in members:
                gid[mem['sku_id']] = key
        rc_rows = [{'sku_id': sid, 'group_id': gid.get(sid, sid),
                    'sku_name': attrs[sid]['raw'].get('sku_name_chi') or '',
                    'brand_canonical': attrs[sid]['brand_canonical'] or ''} for sid in ids]
        res = RC.check(rc_rows, gid)
        return '%d/%d' % (sum(bool(ok) for _, ok, _ in res), len(res))

    m['adversarial'] = adversarial(prod_groups)
    m['adversarial_honest'] = adversarial(groups)
    m['n_groups_prod'] = len(prod_groups)
    m['n_candidates_total'] = len(info)
    m['ann_fidelity'] = ann_fidelity
    m['ann_meta'] = ann_meta
    m['retrieval_seconds'] = retrieval_seconds
    set_form_enabled(True)
    return m


def main():
    rows = load(RAW_DATASET_PATH)
    # Record each row's real form with the genuine extractor first, for the embedding text
    for r in rows:
        REAL_FORM_BY_SKU[r['sku_id']] = _REAL_FORM(r)[0]
    cells = [
        ('current config (form + exact)', True, False),
        ('no form + exact', False, False),
        ('form + ANN', True, True),
        ('no form + ANN', False, True),
    ]
    print('=' * 96)
    print('ablation: product_form on/off × exact vs ANN   (method 5, threshold %.2f, honest track)' % THRESHOLD)
    print('=' * 96)
    out = []
    for label, form_on, ann in cells:
        m = run_cell(rows, form_on, ann)
        out.append((label, m))
    print()
    print('%-30s %9s %9s %9s %9s %9s %8s' % (
        'config', 'cand rec', 'complete', 'mismerge', 'precision', 'pack inv', 'adv prod'))
    for label, m in out:
        print('%-30s %8.1f%% %8.1f%% %8.2f%% %8.1f%% %8.1f%% %8s' % (
            label, m['candidate_recall'] * 100, m['silver_recall'] * 100,
            m['silver_negative_merge_rate'] * 100, m['case_control_precision'] * 100,
            m['pack_invariance'] * 100, m['adversarial']))
    print()
    print('%-30s %10s %12s %12s %9s' % ('config', 'cand pairs', 'grps(honest)', 'grps(prod)', 'retrieval'))
    for label, m in out:
        print('%-30s %10d %12d %12d %8.2fs' % (
            label, m['n_candidates_total'], m['n_groups'], m['n_groups_prod'],
            m['retrieval_seconds']))
    print()
    print('--- ANN fidelity and compute ---')
    for label, m in out:
        if m['ann_fidelity'] is None:
            continue
        a = m['ann_meta']
        print('  %-28s recall against exact top-%d: %.1f%%' % (label, TOP_K, m['ann_fidelity'] * 100))
        print('    %d buckets, %d probed per query: %.0f rows compared on average, exact needs %d (%.1f%% saved)'
              % (a['nlist'], a['nprobe'], a['avg_compared'], a['exhaustive_compared'],
                 (1 - a['avg_compared'] / a['exhaustive_compared']) * 100))
    print()
    print('--- extrapolated to 3 million rows (the actual reason to switch to ANN) ---')
    for n in (1472, 100_000, 3_000_000):
        nlist = int(np.sqrt(n))
        ann_cmp = NPROBE * (n / nlist)
        print('  n=%9d : exact compares %12d rows/query   ANN about %8d rows   saves %.2f%%'
              % (n, n - 1, ann_cmp, (1 - ann_cmp / (n - 1)) * 100))
    print()
    print('  gate: wrong merges <= 2.5%, pack invariance >= 98%, adversarial regression 7/7')


if __name__ == '__main__':
    main()
