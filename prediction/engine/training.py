# -*- coding: utf-8 -*-
"""Training infrastructure shared by all five methods: context, label factory, leak-free folds.

Only the parts reused by the multi-pass retrieval pipeline live here; the actual model
training is in multipass.train().

⚠️ Three designs that keep the evaluation honest (see the risk table in PLAN §8):
 1. **Training is restricted to the sub-population where both listings carry a valid EAN**.
    Positives all have barcodes by construction; if the negatives did not, the model would
    learn "both sides have a barcode ⇒ same product", a pure artefact of the dataset. Inside
    one sub-population barcode availability is constant across positives and negatives, and
    the artefact disappears. The price: this sub-population is selection-biased (PLAN §5.2),
    so transferring to the full population needs a covariate-shift assumption and must be
    checked against the human gold standard -- which the report states explicitly.
 2. **Folds are split by barcode family**. Listings sharing a barcode never straddle
    train/test, otherwise the model scores well simply by memorising barcodes.
 3. **Diagnostic features** (pack_count_same / same_merchant) stay in the model with an
    expected importance of ≈0. If they rank near the top, the model has learned the wrong
    thing -- they are an alibi, not useful features.
"""

# On reproducibility: results were once stabilised by forcing PYTHONHASHSEED=0, but that only
# masked the symptom. The root cause is fixed at the source -- every `key=len` sort over a
# set/dict now carries a lexicographic secondary key, and floats are sorted() before summing
# (float addition is not associative, so set iteration order changes the result).
# Verified: output is bit-identical under 4 different PYTHONHASHSEED values, no environment
# variable required.

import sys, os, itertools, collections, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core import extract as X, variant as V, group as G                                    # noqa
from prediction.lib import features as F                                                   # noqa

N_FOLDS = 5


def build_context(rows, use_form=True):
    attrs, gs1_brand = G.build_attributes(rows, use_form=use_form)
    brand_of = {k: a['brand_id'] for k, a in attrs.items()}
    sigs, vocab, tc = G.build_signatures(rows, attrs)
    alt = V.mine_alternative_pairs(rows, brand_of, tc, vocab, attrs=attrs)
    resid = G.residual_cache(attrs)
    # Cleaned alternative pairs: support>=2 minus the barcode-contradicted ones (paraphrases).
    # ⚠️ The feature layer must use the cleaned set too -- min_brands=1 inflates the raw alt
    # set to 15k pairs, 14k of which have support=1 and are noise; feeding those into the
    # variant_conflict feature measurably depresses the model probability of genuine
    # same-group pairs (silver recall 63.7% -> 56.5%).
    ean_of = {r['sku_id']: X.extract_ean_direct(r) for r in rows}
    contra = V.contradicted_pairs(rows, sigs, ean_of)
    minimal = V.mine_minimal_pairs(rows, brand_of, sigs, attrs=attrs)
    # Veto set = cross-group alternative pairs with support>=2 ∪ minimal pairs (one-slot
    # substitutions under an equal template), minus the barcode-contradicted ones
    veto = (V.veto_pairs(alt, contra, min_support=2) | (minimal - contra))
    kvt = V.known_variant_terms(veto)
    # The feature layer uses "clean signatures": keep only signature tokens that hit a
    # confirmed variant term (the veto-set vocabulary). The rich vocabulary (min_brands=1) is
    # necessary for mining but is noise for features -- within-brand paraphrases such as
    # 瘦羊/山羊奶 leaking into the signature depress variant_sig_jaccard on genuine same-group
    # pairs, measurably dropping CV AUC from 0.960 to 0.925. Filtering by kvt leaves the
    # features looking only at real variant dimensions.
    tok_df = collections.Counter()
    for k, v in sigs.items():
        tok_df.update(v)

    core_sorted = sorted(kvt, key=lambda t: (len(t), t))   # shortest core first; deterministic tie-break

    def _clean(sig):
        # Three rules:
        # ① Tokens that hit kvt are **normalised to the shortest core term**:
        #    檸檬味/檸檬香/檸檬精油 -> 檸檬. Without it, two spellings of one variant score
        #    jaccard=0 and the model probability drops -- in practice it split
        #    "瘦羊皂檸檬味100g*2" from "山羊奶皂-檸檬香100克" (breaking pack invariance).
        # ② Low-frequency decisive tokens (df<=5, e.g. 鈴蘭/我愛你媽媽) are kept verbatim --
        #    dropping them blinds Bio d'Azur's variant features, which then merge on
        #    title-prefix similarity.
        # ③ Everything else (within-brand paraphrases such as 瘦羊/山羊奶) is discarded.
        out = set()
        for t in sig:
            hit = next((u for u in core_sorted if u in t), None)
            if hit:
                out.add(hit)
            elif any(t in u for u in kvt) or tok_df[t] <= 5:
                out.add(t)
        return frozenset(out)
    sigs_feat = {k: _clean(v) for k, v in sigs.items()}
    idf, n = F.build_idf(rows)
    return dict(attrs=attrs, sigs=sigs, sigs_feat=sigs_feat, resid=resid, alt=alt,
                veto=veto, kvt=kvt, idf=idf, n=n)


def make_labels(rows, ctx):
    """Label factory. Returns [(sku_a, sku_b, y, source)] within the barcoded sub-population."""
    attrs = ctx['attrs']
    sigs = ctx['sigs']
    ean = {r['sku_id']: X.extract_ean_direct(r) for r in rows}
    have = [r for r in rows if ean[r['sku_id']]]
    ids = sorted(r['sku_id'] for r in have)

    pos, neg = set(), []

    # ---- Positives: same barcode ----
    e2s = collections.defaultdict(set)
    for sid in ids:
        for e in ean[sid]:
            e2s[e].add(sid)
    for skus in e2s.values():
        for a, b in itertools.combinations(sorted(skus), 2):
            pos.add((a, b))

    def add(a, b, src):
        key = tuple(sorted((a, b)))
        if key not in pos:
            neg.append((key[0], key[1], 0, src))

    by_brand = collections.defaultdict(list)
    for sid in ids:
        by_brand[attrs[sid]['brand_key']].append(sid)

    for b, group in by_brand.items():
        if not b:
            continue
        for a, c in itertools.combinations(sorted(group), 2):
            aa, ac = attrs[a], attrs[c]
            if ean[a] & ean[c]:
                continue
            same_unit = X.unit_compatible((aa['unit_value'], aa['unit_uom']),
                                          (ac['unit_value'], ac['unit_uom']))
            # HARD-A: same brand + same unit + same pack count + different barcode -> a real
            # variant difference. The "same pack count" clause is not optional: GTINs are
            # assigned per tradeable unit, so multi-packs carry their own barcode; without it
            # many genuinely same-group "pack count only" pairs leak in (1217 -> 645)
            if same_unit and aa['pack_count'] == ac['pack_count']:
                add(a, c, 'HARD-A same brand, unit and pack count')
            # HARD-B: same brand + identical variant signature + unit size differing by >10%
            elif (not same_unit and sigs[a] and sigs[a] == sigs[c]):
                add(a, c, 'HARD-B same variant, different unit size')

    # HARD-C: origin conflict
    for a, c in itertools.combinations(ids, 2):
        aa, ac = attrs[a], attrs[c]
        if aa['origin'] and ac['origin'] and not (aa['origin'] & ac['origin']) \
                and aa['brand_key'] and aa['brand_key'] == ac['brand_key']:
            add(a, c, 'HARD-C origin conflict')
    # HARD-D/E: different product form
    # ⚠️ There are 4000+ different-form pairs and they are trivially easy (the model gets them
    # right without learning anything). Feeding in all of them would drown out the genuinely
    # hard HARD-A pairs and inflate AUC for nothing, so they are downsampled to HARD-A scale.
    de = []
    for a, c in itertools.combinations(ids, 2):
        aa, ac = attrs[a], attrs[c]
        if aa['form'] != ac['form'] and aa['form'] != 'UNKNOWN' and ac['form'] != 'UNKNOWN':
            de.append((a, c))
    rng0 = np.random.default_rng(7)
    for i in rng0.choice(len(de), min(300, len(de)), replace=False):
        add(de[i][0], de[i][1], 'HARD-DE different form (downsampled)')

    # EASY: random cross-brand pairs (also restricted to the barcoded sub-population, so
    # barcode availability stays constant)
    rng = np.random.default_rng(11)
    tries = 0
    easy = 0
    target = len(pos) * 3
    while easy < target and tries < target * 60:
        tries += 1
        a, c = ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]
        if a == c or attrs[a]['brand_key'] == attrs[c]['brand_key']:
            continue
        key = tuple(sorted((a, c)))
        if key in pos:
            continue
        neg.append((key[0], key[1], 0, 'EASY random cross-brand'))
        easy += 1

    # Deduplicate (one pair can match several rules; keep the first source)
    seen = {}
    for a, b, y, s in neg:
        seen.setdefault((a, b), s)
    data = [(a, b, 1, 'POS same barcode') for a, b in sorted(pos)]
    data += [(a, b, 0, s) for (a, b), s in sorted(seen.items())]
    return data, ean


def fold_of(sid, ean):
    """Fold by **barcode family**: same barcode -> same fold, or the model just memorises EANs."""
    e = sorted(ean[sid])[0]
    return int(hashlib.sha1(e.encode()).hexdigest(), 16) % N_FOLDS
