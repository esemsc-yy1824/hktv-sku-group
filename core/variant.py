# -*- coding: utf-8 -*-
"""Automatic mining of variants (scent / ingredient / formulation).

⚠️ Design constraint: **never hand-write a soap dictionary**. The site has hundreds of leaf
categories, and hand-writing a variant vocabulary per category does not scale — it directly
violates the project's top-priority requirement. So everything here is mined statistically
from the data.

Two steps:
  1. Mine the variant vocabulary: tokens that discriminate inside a brand (they split the brand)
     but are not site-wide generic words.
  2. Mine mutually exclusive families: if two words inside one brand each occur >=2 times yet
     never co-occur -> they are alternative values of the same dimension (檸檬 "lemon" / 燕麥
     "oat" / 椰子油 "coconut oil" form one family; the families are what make a "conflict"
     decidable).
"""
import re
import collections
import itertools

from .norm import clean_title
from .extract import FORM_RULES

# Variant = residual. Delete everything nameable from the text; what is left is the variant.
_SPEC = re.compile(r'\d')
_FORM_WORDS = re.compile('|'.join(p for _, p in FORM_RULES), re.I)
# Pack-count / bundle / spec tokens
_PACK = re.compile(
    r'\d+\s*[x×*]\s*\d+'
    r'|[x×*]\s*\d+'
    r'|[一二兩两三四五六七八九十\d]+\s*(件|塊|块|個|个|入|枚|條|支|包|盒|片|粒|pcs?)\s*裝?'
    r'|套裝|組合|优惠|優惠|原盒|pack\s*of\s*\d+|\d+\s*入裝?',
    re.I)
_UNIT_TOK = re.compile(
    r'\d+(\.\d+)?\s*(kg|公斤|g|克|gram[s]?|gr|公克|ml|毫升|cc|l|升|公升|oz|安士|盎司|片|粒|枚|pcs?)',
    re.I)
# Generic adjectives / marketing words (cross-category, not soap-specific — these can be
# maintained as a site-wide stopword list)
_GENERIC = re.compile(
    r'天然|純天然|纯天然|有機|有机|溫和|温和|保濕|保湿|滋潤|滋润|清潔|清洁|潔淨|洁净'
    r'|舒緩|舒缓|修護|修护|深層|深层|適用|适用|專用|专用|配方|系列|精華|精华'
    r'|嫩膚|嫩肤|美肌|抗菌|殺菌|杀菌|去角質|去角质|控油|新舊|包裝|包装|限量|人氣|人气'
    r'|推薦|推荐|必備|必备|正品|優質|优质|高級|高级|經典|经典|升級|升级|加強|加强|冷製|冷制|研磨|手工|手作|工藝|工艺|香氛|香味|味道|配製|配制|製造|制造|補濕|补湿|嫩白|美白|淨透|净透|舒緩|去味|除味|潤澤|润泽|全家|適合|适合|品牌|行貨|行货|正貨|正货|進口|进口|代理|專櫃|专柜')


def _residual(text, brand=None):
    """Strip form words, pack counts, specs, generic marketing words and **brand words** from the
    title; the residual left over is the variant.

    ⚠️ Brand words must be stripped. Measured, leaving them in makes `goat` a variant word that
    enters a mutually exclusive pair, so "純天然山羊奶皂-原味" and "Goat Soap 瘦羊皂原味" are
    wrongly judged incompatible because their difference set contains goat.
    """
    t = clean_title(text)
    if brand:
        for piece in re.split(r'[\s\-_/]+', str(brand)):
            piece = piece.strip()
            if len(piece) >= 2:
                t = re.sub(re.escape(piece), ' ', t, flags=re.I)
    t = _UNIT_TOK.sub(' ', t)
    t = _PACK.sub(' ', t)
    t = _FORM_WORDS.sub(' ', t)
    t = _GENERIC.sub(' ', t)
    return t


def _residual_keep_variant(text):
    """Strip only pack counts / specs / marketing noise, **keeping** form words and variant words.

    Used by features.packfree_text: title similarity should not be influenced by pack markers,
    but form words (皂 "soap" / 沐浴露 "body wash") carry information and must not be dropped the
    way variant mining drops them.
    """
    t = clean_title(text)
    t = _UNIT_TOK.sub(' ', t)
    t = _PACK.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def tokens(text, brand=None, nmin=2, nmax=4):
    """2-4 grams for Chinese, whole words for English. No tokenizer — that avoids both a
    dependency and a dictionary.

    ⚠️ Take the grams on the **residual text**. Taking them straight from the raw title mixes in
    form+packaging fragments such as 件裝 / 天然手工 / 然手工皂 / 油手工皂; measured, that inflates
    the signature similarity of 可可脂 vs 迷迭香 to 0.75 and merges 88 differently-scented items
    into a single group.
    """
    t = _residual(text, brand)
    out = set()
    for w in re.findall(r'[a-z]{3,}', t.lower()):
        out.add(w)
    for seg in re.findall(r'[一-鿿]+', t):
        for n in range(nmin, nmax + 1):
            for i in range(len(seg) - n + 1):
                out.add(seg[i:i + n])
    return {g for g in out if not _SPEC.search(g) and not _FORM_WORDS.search(g)
            and not _GENERIC.search(g)}


def mining_key(sku_id, brand_of, attrs=None):
    """Grouping key used for mining. Both pitfalls below were hit for real; neither can be undone:

    1. brand_id (the GS1-first one) cannot be used directly: some NAJEL items carry a barcode
       (brand_id='GS1:3760061') and some do not (brand_id='najel'), so one brand gets
       **fragmented** into two mining groups. 紅泥 and 蜜糖 then each fall short of the in-group
       min_count=2, the (紅泥,蜜糖) alternative pair has support=0, and the hard veto stops working.
       → Prefer the **brand field itself** (brand_key), which does not depend on barcode
       availability.
    2. Brandless items (189/1472 rows) cannot be skipped wholesale: the entire soap-paper family
       lives in that bucket.
       → Use (form, spec) as a pseudo-brand.
    """
    if attrs and sku_id in attrs:
        a = attrs[sku_id]
        if a.get('brand_key'):
            return a['brand_key']
        b = brand_of.get(sku_id, '')
        if b:
            return b                      # no brand field but a GS1 prefix -> use the GS1 group
        # ⚠️ The pseudo-brand must carry the merchant id. support counts "the number of independent
        # evidence sources": the soap-paper family's 45 items come from 3 merchants, and if they
        # all land in one NB group, (玫瑰,草莓) can only ever reach support=1 no matter how many
        # merchants corroborate it, never clearing the >=2 threshold. Split by merchant, an
        # independent merchant = independent evidence, exactly matching a brand group's semantics.
        return 'NB:%s:%s:%s%s' % (sku_id[:5], a['form'], a['unit_value'], a['unit_uom'])
    return brand_of.get(sku_id, '')


def mine_vocabulary(rows, brand_of, min_brands=1, max_global_df=0.10, min_brand_count=2,
                    attrs=None):
    """Mine the variant vocabulary.

    A token is selected only if it satisfies both conditions:
      - it "splits" at least min_brands brands internally (the brand has items with it and items
        without it)
      - its global document frequency is <= max_global_df (excludes generic marketing words such
        as 天然/溫和/保濕)

    ⚠️ min_brands must be 1, never 2. Variant words are frequently **brand-exclusive**: NAJEL's
    紅泥/死海泥/黑茴香油 appear only under NAJEL, so requiring 2 brands filters all of them out;
    measured, that drops 紅泥 from the signature -> a false 1.0 subset relation -> 紅泥 merges with
    蜜糖. Cross-brand noise is caught instead by max_global_df and the support threshold on
    alternative pairs.
    """
    by_brand = collections.defaultdict(list)
    for r in rows:
        by_brand[mining_key(r['sku_id'], brand_of, attrs)].append(r)

    tok_cache = {r['sku_id']: tokens(r.get('sku_name_chi'), r.get('brand_name_chi'))
                 for r in rows}
    global_df = collections.Counter()
    for tk in tok_cache.values():
        global_df.update(tk)
    n_all = len(rows)

    splits = collections.Counter()
    for b, items in by_brand.items():
        if not b or len(items) < 3:
            continue
        df = collections.Counter()
        for r in items:
            df.update(tok_cache[r['sku_id']])
        for t, c in df.items():
            if min_brand_count <= c < len(items):      # present but not universal -> informative
                splits[t] += 1

    vocab = {t for t, k in splits.items()
             if k >= min_brands and global_df[t] / n_all <= max_global_df}
    vocab = _drop_fragments(vocab, global_df)
    return vocab, tok_cache


def _drop_fragments(vocab, df):
    """Drop n-gram fragments: if t only ever occurs inside a longer t' (equal df), it carries no
    information of its own.

    Example: 敘利亞阿勒頗 (Aleppo, Syria) spawns fragments such as 亞阿勒頗 / 利亞阿勒 / 勒頗古皂
    whose df equals the parent's; keeping them only pollutes the alternative pairs (measured to
    remove roughly 1/3 of the noise words).
    """
    by_len = sorted(vocab, key=lambda t: (-len(t), t))
    drop = set()
    for i, t in enumerate(by_len):
        if t in drop:
            continue
        for u in by_len[i + 1:]:
            if u in drop or len(u) >= len(t):
                continue
            if u in t and df[u] == df[t]:
                drop.add(u)
    return vocab - drop


def signature(sku_id, tok_cache, vocab):
    """One item's variant signature = the variant words it hits (maximal terms only, so n-gram
    fragments are dropped)."""
    hit = tok_cache[sku_id] & vocab
    terms = sorted(hit, key=lambda t: (-len(t), t))
    keep = []
    for t in terms:
        if not any(t in k for k in keep):
            keep.append(t)
    return frozenset(keep)


def residual_tokens(text, brand=None):
    """Residual-text tokens (not limited to the variant vocabulary). Fallback when it misses."""
    return tokens(text, brand)


def known_variant_terms(alt_pairs):
    """Words appearing in any alternative pair = words confirmed to be a "variant dimension"."""
    return {t for pair in alt_pairs for t in pair}      # iterating a Counter yields its keys


def variant_asymmetry(sig_a, sig_b, kvt):
    """Whether the difference set contains a "confirmed variant dimension" word.

    ⚠️ **Deprecated as a hard veto** (kept for experiments only). Measured, it wrongly kills 47 of
    the 193 silver-standard positive pairs, and every one of them is a **synonymous rewording**:
    兒童 vs 嬰幼兒, 摩洛哥油 vs 摩洛哥堅果油, 燕麥味 vs 燕麥. "Never co-occurring ⇒ mutually
    exclusive" cannot in principle tell synonyms from real variants — that needs external evidence
    (the barcode contradiction whitelist, see contradicted_pairs) or a semantic model.
    """
    only_a, only_b = sig_a - sig_b, sig_b - sig_a
    if not only_a and not only_b:
        return False

    def expand(terms):
        out = set(terms)
        for t in terms:
            for u in kvt:
                if u != t and u in t:
                    out.add(u)
        return out

    ea, eb = expand(only_a), expand(only_b)
    common = ea & eb
    ea, eb = ea - common, eb - common
    return bool(ea & kvt) or bool(eb & kvt)


def variant_compatible(sa, sb, alt_pairs, kvt, ra, rb, tau=0.34, tau_res=0.60):
    """Whether two items' variants are compatible.

    Three tests, strongest first:
      1) conflicting alternative values -> incompatible
      2) the difference set contains a "known variant dimension" word -> incompatible
         (Goat 摩洛哥 vs 蜂蜜味: the shared goat alone pushes similarity to 0.5, so a
         similarity-only test would merge them by mistake)
      3) both sides have variant words -> strict Jaccard >= tau
         neither side has any          -> fall back to residual-text Jaccard >= tau_res
         (neither 檜木皂 nor 金銀花皂 had its variant word mined, so only the residual separates them)
    """
    if variant_conflict(sa, sb, alt_pairs):
        return False
    if (sa ^ sb) & kvt:
        return False
    if sa and sb:
        return len(sa & sb) / len(sa | sb) >= tau
    if not sa and not sb:
        if not ra or not rb:
            return False                  # no residual either -> no positive evidence, no merge
        return len(ra & rb) / len(ra | rb) >= tau_res
    return False                          # only one side has variant words -> asymmetric evidence


def mine_alternative_pairs(rows, brand_of, tok_cache, vocab,
                           min_count=2, min_coverage=0.0, attrs=None):
    """Mine "alternative value pairs": two words that each occur >=min_count times within a brand
    yet never co-occur -> they substitute for one another.

    ⚠️ No transitive closure. "A never co-occurs with B" + "B never co-occurs with C" does not
    imply "A excludes C" (measured: a union-find collapses 411 words into a single family, useless).
    Only pairwise relations are kept.

    ⚠️ min_coverage defaults to off. The first version set it to 0.35 to filter marketing words and
    it backfired: inside a large brand block (GrabNat, 88 rows) each scent covers only 2-4 rows,
    about 7% coverage, so it **filtered out precisely the real variant pairs** and merged 88
    differently-scented soaps into one group. Filtering marketing words is now the job of
    `maximal_signature` plus the positive-evidence requirement (see group.py).
    """
    by_brand = collections.defaultdict(list)
    for r in rows:
        by_brand[mining_key(r['sku_id'], brand_of, attrs)].append(r)

    pairs = collections.Counter()
    for b, items in by_brand.items():
        if not b or len(items) < 4:
            continue
        sigs = [tok_cache[r['sku_id']] & vocab for r in items]
        df = collections.Counter()
        for s in sigs:
            df.update(s)
        cand = [t for t, c in df.items() if c >= min_count]
        n = len(items)
        for t1, t2 in itertools.combinations(sorted(cand), 2):
            if any(t1 in s and t2 in s for s in sigs):
                continue                                  # co-occurred -> not exclusive
            if (df[t1] + df[t2]) / n < min_coverage:
                continue                                  # coverage too low -> likely marketing
            pairs[(t1, t2)] += 1
    return pairs        # pair -> support (in how many groups they behave as mutually exclusive)


def mine_minimal_pairs(rows, brand_of, sigs, attrs=None):
    """Minimal-pair mining: within one mining group, two titles whose variant signatures differ by
    exactly **one word**, and whose remainders are **identical** once that word is removed from
    each — those two words are two values of the same variant slot, hence mutually exclusive.

    This is far stronger evidence than "never co-occurs" (the linguistic notion of a minimal pair):
      (50片玫瑰味）日本便攜香皂紙...  vs  (50片草莓味）日本便攜香皂紙...
      紅泥 阿勒頗天然手工皂 法國品牌...  vs  蜜糖 阿勒頗天然手工皂 法國品牌...
      法國有機橄欖油乳木果皂125g - 鈴蘭  vs  ... - 麝香
    Same template, one slot substituted. The "never co-occurs" support count cannot catch these
    (草莓/蘋果 appear only 1-2 times per merchant group and never reach the df>=2 threshold), but
    template equality is itself strong evidence and needs no frequency at all.

    Synonym risk is low: a merchant copying a listing does not swap exactly one synonym and leave
    every other character untouched; and if it happens, the barcode contradiction whitelist
    (contradicted_pairs) still catches it one layer up.
    """
    from .norm import fold
    by = collections.defaultdict(list)
    for r in rows:
        by[mining_key(r['sku_id'], brand_of, attrs)].append(r)
    out = set()
    for key, items in by.items():
        if len(items) < 2:
            continue
        folded = {r['sku_id']: fold(clean_title(r.get('sku_name_chi'))) for r in items}
        for x, y in itertools.combinations(items, 2):
            a, b = x['sku_id'], y['sku_id']
            oa, ob = sigs[a] - sigs[b], sigs[b] - sigs[a]
            if len(oa) != 1 or len(ob) != 1:
                continue
            t1, t2 = next(iter(oa)), next(iter(ob))
            ta = folded[a].replace(t1, '')
            tb = folded[b].replace(t2, '')
            if ta and ta == tb:
                out.add((min(t1, t2), max(t1, t2)))
    return out


def contradicted_pairs(rows, sigs, ean_of):
    """Alternative pairs refuted by barcode evidence: the same barcode means the same product, so
    if the difference set between two spellings contains a pair (t1,t2) that was mined as mutually
    exclusive, that pair is really a **synonymous rewording**, not a variant difference.

    This is the standard way to clean statistical rules with weak supervision (barcodes).
    Measured, it rescues synonym pairs such as (兒童,嬰幼兒), (燕麥味,燕麥), (摩洛哥油,摩洛哥堅果油).

    ⚠️ Honest disclosure: this whitelist uses the barcode signal, and silver-standard recall is
    computed from barcodes too — so recall after cleaning is **slightly optimistic** (the rules
    have seen the same barcodes). Final numbers must come from the human gold standard; when
    scaling up, split the "cleaning" and "evaluation" barcodes by hash.
    """
    import itertools as _it
    e2s = collections.defaultdict(set)
    for r in rows:
        for e in ean_of.get(r['sku_id'], ()):
            e2s[e].add(r['sku_id'])
    bad = set()
    for skus in e2s.values():
        for a, b in _it.combinations(sorted(skus), 2):
            oa, ob = sigs[a] - sigs[b], sigs[b] - sigs[a]
            for t1 in oa:
                for t2 in ob:
                    bad.add((min(t1, t2), max(t1, t2)))
    return bad


def high_confidence_pairs(pairs, min_support=2):
    """Keep only high-confidence alternative pairs, for use as a **hard veto**.

    Statistically mined pairs are noisy, and feeding all of them to the hard veto causes false
    kills: measured (min_support=1), recall at the 0.90 threshold falls from 72.5% to 61.1%.
    Requiring "mutually exclusive in >=2 different groups" filters out pairs whose zero
    co-occurrence was accidental. The soft feature (variant_conflict, fed to the model) still uses
    every pair — being wrong there is not fatal.

    ⚠️ min_support was measured against the silver standard, not guessed:
      support>=1: 15,020 veto pairs, 53/193 = 27.5% of silver positives wrongly killed  ← unusable
      support>=2:    416 veto pairs, 0/193 = 0.0% killed                                ← production
    The 14k support=1 pairs are essentially noise; barcodes cover only 15% of the items, so the
    contradiction whitelist cannot clean noisy pairs that sit on barcode-less items.
    Cost: **brand-exclusive** variant pairs (紅泥/死海泥 occur only in NAJEL, so cross-brand
    support never reaches 2) are out of reach on a purely statistical route — a known boundary
    that needs a semantic model.
    """
    return {p for p, c in pairs.items() if c >= min_support}


def veto_pairs(alt_pairs, contradicted, min_support=1):
    """Veto pairs = mined pairs (incl. brand-exclusive support=1 ones) minus barcode-refuted
    synonym pairs."""
    return {p for p, c in alt_pairs.items()
            if c >= min_support and p not in contradicted}


# Cache for substring expansion: the hot spot is "every candidate pair scans all alternative pairs
# with its signature words". A single run only ever sees a handful of distinct pair-set objects,
# so memoizing on (id, term) is enough.
_EXPAND_CACHE = {}


def _expand(terms, alt_pairs):
    key_base = id(alt_pairs)
    out = set(terms)
    for t in terms:
        k = (key_base, t)
        hit = _EXPAND_CACHE.get(k)
        if hit is None:
            hit = set()
            for u1, u2 in alt_pairs:
                if u1 != t and u1 in t:
                    hit.add(u1)
                if u2 != t and u2 in t:
                    hit.add(u2)
            _EXPAND_CACHE[k] = hit
        out |= hit
    return out


def variant_conflict(sig_a, sig_b, alt_pairs):
    """Whether two items' variants conflict: one has t1, the other t2, and (t1,t2) is a known pair
    of mutually exclusive values.

    ⚠️ Matching must be by **substring**, not by exact equality.
    Signatures keep maximal terms (`檸檬味`) while alternative pairs store core terms (`檸檬`,`玫瑰`)
    — measured, the high-confidence conflict between (50片玫瑰味) and (50片檸檬味) was missed purely
    because exact matching could not find it.
    Rule: signature word t counts as hitting alternative word u when u ⊆ t (u is a substring of t).

    Note that the "both sides have it" case must be excluded — that means a verbose and a terse
    spelling of the same product, not a conflict.
    """
    only_a = sig_a - sig_b
    only_b = sig_b - sig_a
    if not only_a or not only_b:
        return False

    ea, eb = _expand(only_a, alt_pairs), _expand(only_b, alt_pairs)
    # Core terms shared after expansion are not conflict evidence (verbose vs terse spelling)
    common = ea & eb
    ea, eb = ea - common, eb - common
    for t1 in ea:
        for t2 in eb:
            if (t1, t2) in alt_pairs or (t2, t1) in alt_pairs:
                # Shared-core guard: 檸檬味 vs 檸檬香 share the core 檸檬 — these are two
                # spellings of the same variant (minimal-pair mining mislabels them as mutually
                # exclusive), not a conflict. A real conflict (玫瑰味 vs 檸檬味) shares no core
                # and is unaffected.
                if any(c in t1 and c in t2 for c in common):
                    continue
                return True
    return False


def _overlap(a, b):
    """Longest overlap between a's suffix and b's prefix. Only >=2 counts — one shared character
    is coincidence, not the same phrase."""
    for k in range(min(len(a), len(b)) - 1, 1, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def stitch_windows(terms, attest=None):
    """Glue the overlapping n-gram windows of one phrase back into that phrase, then drop the
    fragments it contains.

    ⚠️ This fixes the garbled fragments in group_name. With nmax=4, tokens() means a 6-character
    phrase such as "橄欖油乳木果" can **never** become a term; it exists only as the three
    overlapping windows {橄欖油乳, 欖油乳木, 油乳木果}. The maximal-term filter only checks
    substring containment, and sliding windows do not contain one another, so all three fragments
    end up printed (measured: 140 of 1091 names were garbled this way).

    `attest` is the residual text of the member titles. A stitched phrase is only accepted if it
    really occurs there, which stops coincidental AB + BC overlaps from inventing a word.
    """
    out = sorted(terms, key=lambda t: (-len(t), t))       # secondary key: reproducible per process
    for _ in range(len(out)):                             # at most n merges to reach a fixed point
        found = None
        for a in out:
            for b in out:
                if a == b:
                    continue
                k = _overlap(a, b)
                if k < 2:
                    continue
                cand = a + b[k:]
                if attest is not None and cand not in attest:
                    continue
                found = (a, b, cand)
                break
            if found:
                break
        if not found:
            break
        a, b, cand = found
        out = sorted({cand} | {t for t in out if t != a and t != b},
                     key=lambda t: (-len(t), t))
    keep = []
    for t in out:                                         # already sorted by descending length
        if not any(t in k for k in keep):
            keep.append(t)
    return keep


def variant_label(sig, attest=None, prefer=(), limit=3, exclude=()):
    """Human-readable variant label for group_name.

    ⚠️ The old implementation was `'·'.join(sorted(keep)[:3])` — two flaws stacked on one line:
      1. `sorted()` orders by **code point**, unrelated to how informative a word is;
      2. `[:3]` then discards everything after it (387/1091 groups have more than 3 maximal terms,
         so something was always hidden).
    Measured consequence: across the 67 duplicate-named groups, **all** 201 printed variant slots
    held words shared with the same-named siblings and 0 of them distinguished the products — the
    distinguishing word was in the signature all along, just sorted past position 3.

    `prefer` is passed in by the disambiguation stage: words **unique** among the same-named
    sibling groups take the slots first. Without `prefer` this degrades to "longest word first"
    (longer is more specific), then lexicographic order for reproducibility.
    """
    keep = stitch_windows(sig, attest)
    # Do not repeat what another slot already prints: if origin shows (法國), skip 法國 here
    keep = [t for t in keep
            if not any(t in x for x in exclude if x and len(x) >= 2)] or keep
    if not keep:
        return ''
    pref = [t for t in keep if t in prefer]
    rest = [t for t in keep if t not in prefer]
    ordered = pref + sorted(rest, key=lambda t: (-len(t), t))
    return '·'.join(ordered[:limit])
