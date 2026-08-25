# -*- coding: utf-8 -*-
"""Barcode-derived silver-standard labels and the pack-count invariance test set.

These are **evaluation evidence**, not the internals of any single prediction method, so they
live in core/ rather than under prediction/engine/ — preprocessing writes them out as standalone
files in data/labels/, and all five prediction methods read the same definition at evaluation
time.

⚠️ The silver standard is a proxy label with barcode selection bias, not human-confirmed ground
truth.
"""
from __future__ import annotations

import collections
import itertools

from . import extract as X

def silver_pairs(rows):
    """Barcode silver-standard positives: same barcode -> same product.

    ⚠️ Used only as a recall lower bound; it is not the final metric definition (see the selection
    bias in PLAN §5.2).
    """
    e2s = collections.defaultdict(set)
    for r in rows:
        for e in X.extract_ean_direct(r):
            e2s[e].add(r['sku_id'])
    out = set()
    for skus in e2s.values():
        for a, b in itertools.combinations(sorted(skus), 2):
            out.add((a, b))
    return out


def silver_negatives(rows, pos, attrs):
    """Barcode silver-standard negatives: same brand, **same pack count, same spec**, but disjoint
    barcode sets -> different products.

    🚨 Why "same pack count, same spec" are mandatory conditions:
        GTINs are assigned per **tradable unit** — a multipack carries a barcode of its own,
        separate from the single unit. So "different barcode ⇒ different product" is **wrong**
        under the business definition: the 1-pack and the 3-pack of one soap have different
        barcodes, yet the rules require them in the same group.
        Without these two filters, the negatives would be full of genuinely same-group pairs that
        differ only in pack count, precision would be systematically underestimated, and a model
        trained on them would learn the rule backwards.
    ⚠️ Residual noise: the same product can also change barcode on a packaging refresh. Which is
    why this is only a "silver" standard.
    """
    ean = {r['sku_id']: X.extract_ean_direct(r) for r in rows}
    have = [r for r in rows if ean[r['sku_id']]]
    by_brand = collections.defaultdict(list)
    for r in have:
        by_brand[X.brand_key(r.get('brand_name_chi'))].append(r)
    out = set()
    for b, items in by_brand.items():
        if not b:
            continue
        for x, y in itertools.combinations(sorted(items, key=lambda r: r['sku_id']), 2):
            ax, ay = attrs[x['sku_id']], attrs[y['sku_id']]
            if ean[x['sku_id']] & ean[y['sku_id']]:
                continue
            if ax['pack_count'] != ay['pack_count']:
                continue                       # different pack count -> may be a multipack
            if not X.unit_compatible((ax['unit_value'], ax['unit_uom']),
                                     (ay['unit_value'], ay['unit_uom'])):
                continue                       # different spec -> separate anyway, too easy
            pair = (x['sku_id'], y['sku_id'])
            if pair not in pos:
                out.add(pair)
    return out


def packcount_invariance_pairs(rows, attrs, sigs):
    """Pack-count invariance test set: pairs with the same brand, spec and form, compatible origin,
    an **identical variant signature**, and differing only in pack count.

    ⚠️ The identical-signature condition cannot be dropped. The first version missed it and counted
    "蘆薈 4件裝 vs 可可脂 100克" as a test pair — those belong apart anyway, so the resulting 31.8%
    was meaningless.

    This is the direct regression test for the business rule "a 3-pack and a 6-pack are one group".
    These pairs **must be judged same-group 100% of the time**; any change that breaks it should
    block the release.
    """
    by = collections.defaultdict(list)
    for r in rows:
        a = attrs[r['sku_id']]
        if a['unit_value'] is None or not a['brand_id']:
            continue
        sig = sigs[r['sku_id']]
        if not sig:
            continue                      # no variant info -> cannot confirm same product, exclude
        by[(a['brand_id'], a['unit_value'], a['unit_uom'], a['form'], sig)].append(a)
    out = set()
    for k, items in by.items():
        for x, y in itertools.combinations(sorted(items, key=lambda a: a['sku_id']), 2):
            # An origin conflict already means different products under the business definition,
            # so such a pair cannot test "only the pack count changed".
            if (x['pack_count'] != y['pack_count']
                    and X.origin_compatible(x['origin'], y['origin'])):
                out.add((x['sku_id'], y['sku_id']))
    return out



def load_packcount_pairs(path):
    """Read the frozen pack-count invariance test set.

    🚨 Why it must be frozen: these pairs are bucketed by **variant signature** (see
    packcount_invariance_pairs above), which means **the denominator is derived from the variant
    vocabulary**. Any change that shrinks the vocabulary shrinks the denominator with it, kicking
    the newly failing pairs straight out of the test set — the metric then still prints 100% while
    the business rule is already being violated. This happened once for real: one vocabulary
    threshold change quietly moved the denominator from 85 to 84, hiding 7 genuine failures, and
    all four metrics came out flat or improved.

    So preprocessing writes these pairs to a standalone file, and evaluation only reads the file
    and never recomputes it. A change in the denominator then becomes an explicit file change
    (visible in git diff) instead of a silent metric improvement.
    """
    import csv as _csv
    with open(path, encoding='utf-8') as f:
        return {(r['sku_a'], r['sku_b']) for r in _csv.DictReader(f)}
