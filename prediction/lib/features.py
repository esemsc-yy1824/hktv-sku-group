# -*- coding: utf-8 -*-
"""Pairwise feature engineering.

Design principle: **every feature must be explainable to the stakeholder in one sentence**.
Interpretable features + GBDT (rather than a deep model) is what makes the feature
importance table something we can lay out and walk through.

Three groups of features:
  A. Attribute agreement -- a softened form of the stakeholder's hard rules (the model
     learns how much each one is worth)
  B. Text similarity     -- the IDF-weighted ones target the known failure modes of a
     purely attribute-based rule
  C. Diagnostics         -- pack_count_same / same_merchant, **expected importance ~0**.
     High importance there means the model learned the wrong thing ("same merchant means
     same product", or "same pack count means same product"); this is the exoneration
     exhibit for a statistically literate stakeholder.
"""
import collections
import math
import re

from core.norm import clean_title, fold
from core import variant as V


# ---------------------------------------------------------------- IDF

def build_idf(rows, brand_of=None):
    """Build IDF over the **residual text**.

    This is the main weapon against the number-one failure mode of a pure attribute rule:
      法國有機橄欖油乳木果皂125g - 鈴蘭
      法國有機橄欖油乳木果皂125g - 我愛你，媽媽
    Plain Jaccard weights `法國/橄欖油/乳木果` (frequent, no discriminating power) exactly
    like `鈴蘭` (lily of the valley -- rare, decisive), so the shared prefix alone carries
    the similarity over the threshold. Under IDF weighting `鈴蘭` counts more than ten
    times as much as `法國`.
    """
    df = collections.Counter()
    for r in rows:
        toks = V.residual_tokens(r.get('sku_name_chi'), r.get('brand_name_chi'))
        df.update(toks)
    n = len(rows)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}, n


def _idf(idf, n, t):
    return idf.get(t, math.log(n + 1) + 1.0)


# ---------------------------------------------------------------- text helpers

def packfree_text(s):
    """The title with pack-count / bundle / size markers stripped out.

    Used by the must-link rule in the clustering layer: if two listings have identical
    titles once the pack-count and size markers are gone, they are "the same product in
    different packaging", which the stakeholder's rule requires to be one group, so they
    are merged unconditionally.

    ⚠️ Feeding this into title_2gram_jaccard / len_ratio was tried (to hide pack count
    from the model) and measured worse: at the 0.95 threshold precision went 87.4% ->
    83.8% and invariance 93.5% -> 92.5%. Stripping the size tokens makes products of
    different weights look more alike in title space. Pack-count invariance is therefore
    guaranteed by a **structural must-link** instead of by hiding features.
    """
    return V._residual_keep_variant(s)


def char_ngrams(s, n=2):
    s = fold(s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def weighted_jaccard(a, b, idf, n):
    """IDF-weighted Jaccard: numerator and denominator both weight tokens by their IDF.

    ⚠️ sorted() before summing is mandatory. Float addition is not associative, and set
    iteration order depends on string hash randomisation -- without the sort, the same
    pair of products yields minutely different similarities across processes, enough to
    flip a decision at the threshold boundary and make results irreproducible.
    """
    if not a and not b:
        return 0.0
    inter = sum(_idf(idf, n, t) for t in sorted(a & b))
    union = sum(_idf(idf, n, t) for t in sorted(a | b))
    return inter / union if union else 0.0


def tail_text(s, k=12):
    """Title tail. The part after a dash or pipe is usually the variant name (`... - 鈴蘭`)."""
    t = clean_title(s)
    m = re.split(r'[-–—|｜:：]', t)
    tail = m[-1] if len(m) > 1 else t[-k:]
    return tail.strip()[-k:] if tail.strip() else t[-k:]


def lcp_ratio(a, b):
    """Longest common prefix ratio. Sizes up the "long shared prefix" confusion source."""
    x, y = fold(a), fold(b)
    if not x or not y:
        return 0.0
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i / max(len(x), len(y))


# ---------------------------------------------------------------- feature names

FEATURE_NAMES = [
    # A. Attribute agreement
    'form_same', 'form_both_known',
    'unit_ratio', 'unit_same_uom', 'unit_both_known',
    'origin_compatible', 'origin_exact_same', 'origin_both_known',
    'brand_same', 'brand_both_known',
    'gs1_same', 'gs1_both_known', 'gs1_conflict',
    # B. Text similarity
    'title_2gram_jaccard',
    'resid_jaccard',
    'resid_jaccard_idf',          # ★ aimed at the long-prefix failure mode
    'max_idf_in_diff',            # ★ rarest-token IDF in the diff -- "any decisive word?"
    'sum_idf_in_diff',
    'variant_sig_jaccard',
    'variant_conflict',
    'variant_diff_has_known_term',
    'lcp_ratio',                  # ★ strength of the long shared prefix
    'tail_2gram_jaccard',         # ★ similarity of the title tail (after the dash)
    'tail_idf_overlap',
    'len_ratio',
    # C. Diagnostics (expected importance ~0)
    'pack_count_same',
    'same_merchant',
]


# ⚠️ Features that must be excluded from training: they are **circular by definition**
# with the "same barcode" label. A positive is defined as "same EAN", and the same EAN
# necessarily implies the same GS1 company prefix -> gs1_same is identically 1. Feed them
# to the model and it learns the shortcut "same company prefix means same product":
# measured AUC inflates to 0.997, with gs1_same+gs1_conflict alone taking 76% of the gain.
# The right treatment: GS1 agreement/conflict acts as a **hard constraint** in the
# clustering layer (already implemented in group._same_group); the model is left to learn
# only the text-level discrimination.
LEAKY_FEATURES = {'gs1_same', 'gs1_conflict', 'gs1_both_known'}

# Diagnostic features: **present only in the diagnostic model**, there to expose whether
# the model is taking a shortcut. Measured on the pairwise classifier (this is what the
# feature design rests on):
#   same_merchant    importance 0.095, rank 5 -> the model learned "same merchant ⇒
#                    different product" (a merchant rarely lists the same item twice).
#                    A genuine correlation, but a shortcut, and it does not transfer.
#   pack_count_same  importance 0.041         -> **a direct violation of the
#                    stakeholder's rule**: the rule says pack count does not separate
#                    groups, yet the model treats it as evidence.
# Both are therefore dropped from the production model. The diagnostic model keeps them
# precisely so this table can be laid out for the stakeholder.
DIAGNOSTIC_FEATURES = ['pack_count_same', 'same_merchant']

# Diagnostic model: drop only the definitionally circular GS1 features
DIAG_FEATURES = [f for f in FEATURE_NAMES if f not in LEAKY_FEATURES]
DIAG_IDX = [FEATURE_NAMES.index(f) for f in DIAG_FEATURES]

# Production model: additionally drop the shortcut features
MODEL_FEATURES = [f for f in DIAG_FEATURES if f not in DIAGNOSTIC_FEATURES]
MODEL_IDX = [FEATURE_NAMES.index(f) for f in MODEL_FEATURES]


def to_model_matrix(rows_of_features):
    """Pick the trainable columns out of the full feature vectors."""
    import numpy as _np
    return _np.asarray(rows_of_features, dtype=float)[:, MODEL_IDX]


def pair_features(a, b, sigs, resid, idf, n, alt_pairs, kvt):
    """One product pair -> feature vector. a/b are attr dicts from group.build_attributes."""
    ta, tb = a['raw'].get('sku_name_chi') or '', b['raw'].get('sku_name_chi') or ''
    ra, rb = resid[a['sku_id']], resid[b['sku_id']]
    sa, sb = sigs[a['sku_id']], sigs[b['sku_id']]
    diff = ra ^ rb

    ua, ub = a['unit_value'], b['unit_value']
    unit_known = ua is not None and ub is not None
    unit_ratio = (min(ua, ub) / max(ua, ub)) if unit_known and max(ua, ub) else 0.0

    oa, ob = a['origin'], b['origin']
    origin_known = bool(oa) and bool(ob)

    ga, gb = a['gs1'], b['gs1']
    gs1_known = bool(ga) and bool(gb)

    tail_a, tail_b = tail_text(ta), tail_text(tb)
    tail_ta, tail_tb = V.residual_tokens(tail_a), V.residual_tokens(tail_b)

    return [
        # A
        float(a['form'] == b['form']),
        float(a['form'] != 'UNKNOWN' and b['form'] != 'UNKNOWN'),
        unit_ratio,
        float(unit_known and a['unit_uom'] == b['unit_uom']),
        float(unit_known),
        float(not origin_known or bool(oa & ob)),
        float(origin_known and oa == ob),
        float(origin_known),
        float(bool(a['brand_key']) and a['brand_key'] == b['brand_key']),
        float(bool(a['brand_key']) and bool(b['brand_key'])),
        float(gs1_known and ga == gb),
        float(gs1_known),
        float(gs1_known and ga != gb),
        # B
        jaccard(char_ngrams(ta), char_ngrams(tb)),
        jaccard(ra, rb),
        weighted_jaccard(ra, rb, idf, n),
        max([_idf(idf, n, t) for t in sorted(diff)], default=0.0),
        sum(_idf(idf, n, t) for t in sorted(diff)),   # sorted: see the weighted_jaccard note
        jaccard(sa, sb),
        float(V.variant_conflict(sa, sb, alt_pairs)),
        float(bool((sa ^ sb) & kvt)),
        lcp_ratio(ta, tb),
        jaccard(char_ngrams(tail_a), char_ngrams(tail_b)),
        weighted_jaccard(tail_ta, tail_tb, idf, n),
        (min(len(ta), len(tb)) / max(len(ta), len(tb))) if max(len(ta), len(tb)) else 0.0,
        # C diagnostics
        float(a['pack_count'] == b['pack_count']),
        float(a['sku_id'][:5] == b['sku_id'][:5]),
    ]
