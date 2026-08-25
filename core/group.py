# -*- coding: utf-8 -*-
"""Blocking + group building.

The blocking scheme and its rationale are in research/README.md §15:
  blocking_key = (GS1 company prefix ∥ brand) × unit_value+uom -> recall ceiling 95.9%,
  9,523 candidate pairs
  ⚠️ origin must never enter blocking_key (with it, recall drops from 94.3% to 76.2%)
"""
import hashlib
import collections
import re
import unicodedata

from .extract import (extract_ean_direct, extract_ean_propagated, build_ean_pool,
                      build_gs1_brand_dict, gs1_prefix, brand_key,
                      extract_unit, extract_origin, extract_form,
                      unit_compatible, origin_compatible)
from .norm import nfkc
from . import variant as V


def build_attributes(rows, use_form=True):
    """Extract every attribute in one pass. Returns sku_id -> dict."""
    pool = build_ean_pool(rows)
    direct = {r['sku_id']: extract_ean_direct(r) for r in rows}
    brand_of_ean = {}
    for r in rows:
        b = brand_key(r.get('brand_name_chi'))
        if b:
            for e in direct[r['sku_id']]:
                brand_of_ean.setdefault(e, b)

    attrs = {}
    for r in rows:
        eans, src = extract_ean_propagated(r, pool, brand_of_ean)
        v, u, pc, usrc = extract_unit(r)
        form, fconf, fsrc = extract_form(r) if use_form else ('UNKNOWN', 0.0, None)
        attrs[r['sku_id']] = {
            'sku_id': r['sku_id'],
            'raw': r,
            'ean': eans, 'ean_source': src,
            'gs1': sorted({gs1_prefix(e) for e in eans})[0] if eans else None,
            'unit_value': v, 'unit_uom': u, 'unit_source': usrc,
            'pack_count': pc,                       # ⚠️ extracted, but never enters group_key
            'form': form, 'form_conf': fconf, 'form_source': fsrc,
            'origin': extract_origin(r),
            'brand_raw': r.get('brand_name_chi'),
            'brand_key': brand_key(r.get('brand_name_chi')),
        }

    gs1_brand = build_gs1_brand_dict(rows, {k: a['ean'] for k, a in attrs.items()})
    for a in attrs.values():
        if a['gs1'] and a['gs1'] in gs1_brand:
            a['brand_canonical'] = gs1_brand[a['gs1']]
            a['brand_source'] = 'gs1'
        elif a['brand_key']:
            a['brand_canonical'] = nfkc(a['brand_raw'])
            a['brand_source'] = 'field'
        else:
            a['brand_canonical'] = None
            a['brand_source'] = None
        # Brand identity used for blocking: GS1 prefix first (it bridges brand aliases for free)
        a['brand_id'] = ('GS1:' + a['gs1']) if (a['gs1'] and a['gs1'] in gs1_brand) \
            else (a['brand_key'] or '')
    return attrs, gs1_brand


def blocking_key(a):
    """Keep it wide — this sets the recall ceiling. Only (brand identity, spec)."""
    unit = (a['unit_value'], a['unit_uom'])
    return (a['brand_id'] or '<NB>', unit)


def _same_group(a, b, alt_pairs, sigs, kvt, resid, use_barcode=True, tau=0.34):
    """Whether two items inside a block belong to the same group. Every hard business constraint
    takes effect here.

    use_barcode=False is for **honest evaluation**: the silver standard is defined as "same
    barcode", so if the algorithm also uses the same barcode as a must-link, it is computing the
    answer from the answer (the first risk listed in PLAN §8). This rule has to be off during
    evaluation to see what the model actually learned on its own.
    """
    if a['form'] != b['form']:
        return False                                              # form mismatch
    if not unit_compatible((a['unit_value'], a['unit_uom']),
                           (b['unit_value'], b['unit_uom'])):
        return False                                              # spec mismatch (with tolerance)
    if not origin_compatible(a['origin'], b['origin']):
        return False                                              # origin conflict (NULL tolerant)
    if a['gs1'] and b['gs1'] and a['gs1'] != b['gs1']:
        return False                                              # different maker -> cannot-link
    if use_barcode and a['ean'] and b['ean'] and (a['ean'] & b['ean']):
        return True                                               # same barcode -> must-link
    # ⚠️ Merging demands "positive evidence", not merely "no conflict". The first version used the
    # latter and merged GrabNat's 88 different scents into one group (absence of conflicting
    # evidence != same product).
    return V.variant_compatible(sigs[a['sku_id']], sigs[b['sku_id']], alt_pairs, kvt,
                                resid[a['sku_id']], resid[b['sku_id']], tau)


def build_signatures(rows, attrs):
    """Variant vocabulary + the variant signature of every item.

    This lives in core/ because **both preprocessing and prediction need it**: preprocessing uses
    it to freeze the pack-count invariance test set, prediction uses it for hard vetoes and for
    naming. Both sides must share one definition, or the frozen test set and the signatures used
    at evaluation time will not line up. Keeping it here also spares preprocessing a reverse
    dependency on prediction.
    """
    brand_of = {k: a['brand_id'] for k, a in attrs.items()}
    vocab, tok_cache = V.mine_vocabulary(rows, brand_of, attrs=attrs)
    sigs = {s: V.signature(s, tok_cache, vocab) for s in attrs}
    return sigs, vocab, tok_cache


def residual_cache(attrs):
    return {a['sku_id']: V.residual_tokens(a['raw'].get('sku_name_chi'),
                                           a['raw'].get('brand_name_chi'))
            for a in attrs.values()}


def build_groups(rows, attrs, alt_pairs, sigs, use_barcode=True, tau=0.34):
    kvt = V.known_variant_terms(alt_pairs)
    resid = {a['sku_id']: V.residual_tokens(a['raw'].get('sku_name_chi'),
                                            a['raw'].get('brand_name_chi'))
             for a in attrs.values()}
    """Constrained agglomeration inside each block. Connected components are used, but **every
    edge passes the hard constraints**, so there is no naive-transitive-closure avalanche (one
    bad edge welding two large clusters together)."""
    blocks = collections.defaultdict(list)
    for a in attrs.values():
        blocks[blocking_key(a)].append(a)

    groups = {}
    for key, items in blocks.items():
        parent = {a['sku_id']: a['sku_id'] for a in items}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _same_group(items[i], items[j], alt_pairs, sigs, kvt, resid, use_barcode, tau):
                    ra, rb = find(items[i]['sku_id']), find(items[j]['sku_id'])
                    if ra != rb:
                        parent[ra] = rb
        for a in items:
            groups.setdefault(find(a['sku_id']), []).append(a)
    return groups


def group_id_of(members, alt_pairs, sigs, resid=None):
    """Content-addressed group_id: the same attribute combination always yields the same id, which
    is incremental-friendly.

    ⚠️ The cluster's **shared residual tokens** must go into the canonical string as well.
    Using only (form, brand, variant_label, unit, origin) breaks down: the algorithm correctly
    separated GrabNat's "薑黃沒藥" cluster from its "肉桂橙" cluster, but both have an empty
    variant label -> identical hash -> and the output CSV merged them back into a single 31-row
    group. The shared residual is exactly "what this cluster is about"; only by including it does
    the id truly determine the grouping.

    Cost: change one residual token and the id changes, so a group_id -> previous_group_id
    migration table has to be maintained (PLAN §6.1 already lists this cost). Once attribute
    extraction is stable, the id can go back to attributes alone.
    """
    rep = members[0]
    sig = frozenset.union(*[sigs[m['sku_id']] for m in members]) if members else frozenset()
    origins = sorted(set().union(*[m['origin'] for m in members])) or ['?']
    shared = ''
    if resid is not None and members:
        inter = set.intersection(*[set(resid[m['sku_id']]) for m in members])
        # Keep maximal terms only, so n-gram fragments cannot destabilize the id
        terms = sorted(inter, key=lambda t: (-len(t), t))
        keep = []
        for t in terms:
            if not any(t in k for k in keep):
                keep.append(t)
        # ⚠️ No truncation. The first version took sorted(keep)[:4] and cut off the distinguishing
        # tokens (e.g. 鈴蘭), hashing Bio d'Azur's 14 different scents into one and the same id.
        shared = ','.join(sorted(keep))
    parts = [
        'form=' + rep['form'],
        'brand=' + (rep['brand_id'] or '?'),
        'variant=' + (V.variant_label(sig) or '?'),
        'shared=' + (shared or '?'),
        'unit=%s%s' % (rep['unit_value'], rep['unit_uom']),
        'origin=' + '/'.join(origins),
    ]
    canon = '|'.join(parts)
    return 'G_' + hashlib.sha1(canon.encode()).hexdigest()[:12], canon


FORM_ZH = {'soap_bar': '香皂', 'soap_paper': '皂紙', 'shampoo_bar': '洗髮皂',
           'laundry_soap': '洗衣皂', 'body_wash': '沐浴露', 'hand_wash': '洗手液',
           'deodorant': '止汗/體香', 'body_lotion': '身體乳/護手霜',
           'bath_salt_bomb': '浴鹽/沐浴球', 'face_cleanser': '潔面',
           'accessory': '配件', 'candle': '香氛蠟燭', 'shampoo_liquid': '洗髮水',
           'UNKNOWN': '未分類'}


def name_parts(members, sigs):
    """The individual slots of group_name. Naming and disambiguation share one decomposition so it
    is not written out twice.

    Returns (brand, sig, form_zh, unit, origins, attest). `attest` is the residual text of the
    member titles, handed to variant_label so it can verify a stitched phrase really occurred.
    """
    rep = members[0]
    sig = frozenset.union(*[sigs[m['sku_id']] for m in members]) if members else frozenset()
    brand = next((m['brand_canonical'] for m in members if m['brand_canonical']), None)
    origins = sorted(set().union(*[m['origin'] for m in members]))
    unit = ('%g%s' % (rep['unit_value'], rep['unit_uom'])) if rep['unit_value'] else '規格未知'
    attest = '\x00'.join(V._residual(m['raw'].get('sku_name_chi'),
                                     m['raw'].get('brand_name_chi')) for m in members)
    return brand, sig, FORM_ZH.get(rep['form'], rep['form']), unit, origins, attest


def compose_name(brand, variant, form, unit, origins):
    bits = [b for b in [brand, variant, form, unit] if b]
    name = ' '.join(bits)
    if origins:
        name += ' (%s)' % '/'.join(origins)
    return name


def group_name_of(members, sigs):
    """Human-readable group_name: {brand} {variant} {form} {spec} ({origin})"""
    brand, sig, form, unit, origins, attest = name_parts(members, sigs)
    label = V.variant_label(sig, attest, exclude=[brand, form, unit] + list(origins))
    return compose_name(brand, label, form, unit, origins)


# ---------------------------------------------------------------- Duplicate-name disambiguation
# One group_name spread across several group_ids makes people think the grouping is wrong or buggy.
# Measured on 1472 rows: 50 names were shared by 139 group_ids (179 SKUs), while
# docs/TECHNICAL_PLAN.md §Uniqueness required that number to be 0 in the first place.
#
# ⚠️ This whole section is **pure post-processing**: it only reads members / sigs / the already
#    assigned group_ids and never writes back to sigs, the vetoes, the vocabulary or anything in
#    ctx. So the clustering, the group_ids, the lineage and the signature-derived pack-count
#    invariance test set are all unaffected — a naming-layer fix must not move grouping decisions.
_ATOM_CJK = re.compile(r'[一-鿿]+')
_ATOM_LAT = re.compile(r'[A-Za-z][A-Za-z ]*[A-Za-z]')
_ATOM_TRIM = '的及和與与或含加之了是'          # function words, stripped from an atom's edges

# Logistics / shelf-life / promotional copy: it distinguishes how a merchant lists an item, not
# the product identity. Left in, you get results such as "Medimix 最佳食用日期 香皂" that treat
# the best-before wording as part of the product name.
_ATOM_STOP = re.compile(
    r'最佳(食用)?(日期|期限)|保質期|保质期|到期|臨期|临期|有效期|生產日期|生产日期'
    r'|大掃除|大扫除|清倉|清仓|特價|特价|優惠|优惠|折扣|買一|买一|贈品|赠品|禮盒|礼盒'
    r'|隨機|随机|現貨|现货|預購|预购|補貨|补货|順豐|顺丰|包郵|包邮|發貨|发货')


def _atoms(members, printed):
    """The "atoms" of the members' residual titles — runs of Chinese characters or runs of English
    words, in the order they appear in the title.

    `printed` is text the name already prints (brand / origin / form); it is removed first so it
    is not repeated. Only atoms shared by more than half the members are kept: one merchant's SEO
    keyword stuffing should not become the product name.
    Returns (atom list, a haystack of every atom joined) — the latter tells which atoms are unique
    to a group among its siblings.
    """
    per, hay = [], []
    for m in members:
        raw = V._residual(m['raw'].get('sku_name_chi'), m['raw'].get('brand_name_chi'))
        # ⚠️ The haystack uses the **untrimmed** residual. Building it out of atoms goes wrong:
        # signature words are n-grams, "台灣銷量" is not a substring of any single atom, so a term
        # that all five sibling groups share gets judged "unique" and pushes the genuinely
        # distinguishing words (紫草根 / 芙蓉艾草 / 酪梨) further down.
        hay.append(raw)
        t = raw
        for p in printed:
            if p and len(p) >= 2:
                t = re.sub(re.escape(nfkc(p)), ' ', t, flags=re.I)
        t = _ATOM_STOP.sub(' ', t)
        hits = sorted(list(_ATOM_CJK.finditer(t)) + list(_ATOM_LAT.finditer(t)),
                      key=lambda o: o.start())
        per.append([h.group(0).strip(_ATOM_TRIM).strip() for h in hits
                    if len(h.group(0).strip(_ATOM_TRIM).strip()) >= 2])
    seen_count = collections.Counter(a for p in per for a in set(p))
    need = (len(per) + 1) // 2
    seen, keep = set(), []
    for a in [a for p in per for a in p]:                 # title order, keep first occurrence
        if seen_count[a] >= need and a not in seen:
            seen.add(a)
            keep.append(a)
    return keep, '\x00'.join(hay)


def _width(name):
    """Excel column width counts in the default font's '0' glyph width; CJK characters take two.
    prediction/runner.py hard-codes the group_name column width to 72, so this measures display
    width rather than character count.
    """
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in name)


def _pool(sig, mine, others, printed=()):
    """Candidate pool of distinguishing terms: signature words ∪ title atoms, minus anything
    **every sibling group has** (zero signal).

    Returns [(term, rank)] where rank = (number of sibling groups containing it, first-seen order).
    The fewer siblings contain it, the better it distinguishes and the earlier it sorts. That is
    the core of this layer: the old implementation sorted by code point, and measured, across the
    67 duplicate-named groups all 201 printed variant slots were shared words with 0 that
    distinguished anything.
    """
    pool, rank = [], {}
    for t in sorted(sig, key=lambda x: (-len(x), x)) + list(mine):
        if t in rank:
            continue
        # Logistics/shelf-life copy also "distinguishes", but what it distinguishes is how the
        # merchant listed the item, not the product. Signature words must clear this gate too —
        # being n-gram mined, they happily pick up things like 有效期至 (valid until).
        if _ATOM_STOP.search(t):
            continue
        # Do not repeat what another slot already prints: if origin shows (法國), skip 法國 here
        if any(t in p for p in printed if p and len(p) >= 2):
            continue
        share = sum(1 for h in others if t in h)
        if others and share == len(others):
            continue
        rank[t] = (share, len(pool))
        pool.append(t)
    return pool, rank


def disambiguate_names(groups, sigs, ids, max_terms=3, max_width=72):
    """Add distinguishing information to duplicate group names. Three layers, each handling only
    what the previous one left behind.

      1. Signature words **unique** among the same-named sibling groups take the variant slot
         first (the distinguishing word is usually already in the signature, just pushed past
         position 3 by the old code-point sort)
      2. When the signature holds no distinguishing word, take an atom from the members' residual
         titles that no sibling group has
      3. When the text genuinely cannot distinguish them (members differ only by barcode or
         merchant id), append the group_id's short prefix `#hex6` — a last resort that guarantees
         uniqueness while keeping the id greppable

    The label only widens to a second and third term while conflicts remain; if one term
    distinguishes, only one is written, otherwise the name gets padded out with uninformative
    shared words (the Excel column is only 72 wide).

    `groups`: {key: [member dict]}    `ids`: {key: group_id}
    Returns {key: name}.
    """
    keys = sorted(groups)
    parts = {k: name_parts(groups[k], sigs) for k in keys}
    names, atoms = {}, {}
    for k in keys:
        brand, sig, form, unit, origins, attest = parts[k]
        names[k] = compose_name(
            brand, V.variant_label(sig, attest,
                                   exclude=[brand, form, unit] + list(origins)),
            form, unit, origins)

    def atoms_of(k):
        if k not in atoms:
            brand, _s, form, _u, origins, _a = parts[k]
            atoms[k] = _atoms(groups[k], [brand, form] + list(origins))
        return atoms[k]

    for limit in range(1, max_terms + 1):
        cohorts = collections.defaultdict(list)
        for k in keys:
            cohorts[names[k]].append(k)
        dup = sorted(n for n, ks in cohorts.items() if len(ks) > 1)
        if not dup:
            break
        for name in dup:
            sibs = sorted(cohorts[name])
            hay = {k: atoms_of(k)[1] for k in sibs}
            for k in sibs:
                brand, sig, form, unit, origins, attest = parts[k]
                others = [hay[o] for o in sibs if o != k]
                pool, rank = _pool(sig, atoms_of(k)[0], others,
                                   [brand, form, unit] + list(origins))
                if not pool:
                    continue
                chosen = set(sorted(pool, key=lambda t: rank[t])[:limit])
                stitched = V.stitch_windows(frozenset(chosen), attest)

                def srank(s, chosen=chosen, rank=rank):
                    hits = [rank[t] for t in chosen if t in s or s in t]
                    return min(hits) if hits else (99, 99)

                label = '·'.join(sorted(stitched, key=srank)[:limit])
                cand = compose_name(brand, label, form, unit, origins)
                # When too wide, drop trailing terms one at a time but keep at least one —
                # uniqueness outranks width
                terms = sorted(stitched, key=srank)[:limit]
                while len(terms) > 1 and _width(cand) > max_width:
                    terms = terms[:-1]
                    cand = compose_name(brand, '·'.join(terms), form, unit, origins)
                names[k] = cand

    cohorts = collections.defaultdict(list)
    for k in keys:
        cohorts[names[k]].append(k)
    for name, ks in sorted(cohorts.items()):
        if len(ks) < 2:
            continue
        for k in sorted(ks):                              # layer 3: last-resort uniqueness
            names[k] = '%s #%s' % (name, str(ids[k]).split('_')[-1][:6])
    return names
