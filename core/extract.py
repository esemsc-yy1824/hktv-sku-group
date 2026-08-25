# -*- coding: utf-8 -*-
"""Attribute extraction: turn dirty text into the structured fields group_key needs.

Every extractor returns (value, confidence, source) so downstream code can weight results
and trace problems. Rationale in research/README.md.
"""
import re
import collections

from .norm import nfkc, clean_title, fold

# ---------------------------------------------------------------- EAN-13 barcodes

_D13 = re.compile(r'(?<!\d)(\d{13})(?!\d)')
# Internal short article numbers found in titles: #3288 / (02059) / [86017]
_SHORT = re.compile(r'#\s*(\d{4,6})(?!\d)|[\(\[]\s*(\d{4,6})\s*[\)\]]')


def ean13_valid(s):
    """EAN-13 check digit. The only thing that guarantees 100% precision on barcode extraction."""
    if len(s) != 13 or not s.isdigit():
        return False
    d = [int(c) for c in s]
    chk = (10 - sum(d[i] * (1 if i % 2 == 0 else 3) for i in range(12)) % 10) % 10
    return chk == d[12]


def _text_of(row, with_desc=True):
    fields = [row.get('sku_name_chi'), row.get('summary_chi'), row.get('pack_size_chi')]
    if with_desc:
        fields.append(row.get('description_chi'))
    return ' '.join(nfkc(f) for f in fields if f)


def extract_ean_direct(row):
    """Extract valid EAN-13s from sku_id segments and body text. Measured coverage 18.5%."""
    out = set()
    for part in str(row.get('sku_id', '')).split('_'):
        if ean13_valid(part):
            out.add(part)
    for m in _D13.findall(_text_of(row)):
        if ean13_valid(m):
            out.add(m)
    return out


def build_ean_pool(rows):
    return {e for r in rows for e in extract_ean_direct(r)}


def extract_ean_propagated(row, pool, brand_of_ean):
    """Cross-row propagation of short code -> barcode suffix (FINDINGS §11.1, 8/8 precision).

    ⚠️ At 3 million rows, 4-digit short codes collide in a pool that large, so brand consistency
    is added as a second check.
    """
    direct = extract_ean_direct(row)
    if direct:
        return direct, 'direct'
    text = ' '.join(nfkc(f) for f in [row.get('sku_name_chi'), row.get('summary_chi')] if f)
    codes = {x for tup in _SHORT.findall(text) for x in tup if x}
    if not codes:
        return set(), None
    b = brand_key(row.get('brand_name_chi'))
    hits = set()
    for c in codes:
        for e in pool:
            if not e.endswith(c):
                continue
            # Second check: this barcode's known brand must match the row's brand (or none is set)
            eb = brand_of_ean.get(e)
            if b and eb and b != eb:
                continue
            hits.add(e)
    return hits, 'propagated' if hits else None


def gs1_prefix(ean, n=7):
    """GS1 company prefix. Same prefix -> same manufacturer (a strong must-link prior);
    two differing known prefixes -> hard cannot-link (the Goat case in FINDINGS §13.3b)."""
    return ean[:n] if ean else None


# ---------------------------------------------------------------- Brand

# Non-brand values (shop names / delivery methods), FINDINGS §13.1
_NON_BRAND = re.compile(
    r'直送|百貨|百货|專門店|专门店|代購|代购|旗艦|旗舰|商行|貿易|贸易|批發|批发'
    r'|優質生活|优质生活|生活百貨|官方旗艦',
    re.I)
_NON_BRAND_EXACT = {'a1', 'hongkong', 'hong kong', 'oem', 'n/a', 'na', '-', '無', '无'}


def is_non_brand(b):
    if not b:
        return True
    s = nfkc(b)
    if _NON_BRAND.search(s):
        return True
    return fold(s) in {fold(x) for x in _NON_BRAND_EXACT}


def brand_key(b):
    """Character-level brand normalization (semantics come from GS1 prefixes, FINDINGS §13.3)."""
    if is_non_brand(b):
        return ''
    return re.sub(r"[\s'’`\.\-_&,]", '', nfkc(b).lower())


def build_gs1_brand_dict(rows, ean_of_row):
    """GS1 company prefix -> canonical brand name. Measured 6/6 precision (FINDINGS §13.3c).

    This dictionary is a cross-category asset and can be accumulated across batches.
    """
    votes = collections.defaultdict(collections.Counter)
    for r in rows:
        b = r.get('brand_name_chi')
        if is_non_brand(b):
            continue
        for e in ean_of_row.get(r['sku_id'], ()):
            votes[gs1_prefix(e)][nfkc(b)] += 1
    # For each prefix, the most frequent spelling of the brand becomes the canonical name
    return {p: c.most_common(1)[0][0] for p, c in votes.items() if c}


# ---------------------------------------------------------------- Unit spec

_UOM = {
    'kg': ('g', 1000.0), '公斤': ('g', 1000.0),
    'g': ('g', 1.0), '克': ('g', 1.0), 'gram': ('g', 1.0), 'grams': ('g', 1.0),
    'gr': ('g', 1.0), '公克': ('g', 1.0), 'G': ('g', 1.0),
    'ml': ('ml', 1.0), '毫升': ('ml', 1.0), 'cc': ('ml', 1.0),
    'l': ('ml', 1000.0), '升': ('ml', 1000.0), '公升': ('ml', 1000.0),
    'oz': ('g', 28.3495), '安士': ('g', 28.3495), '盎司': ('g', 28.3495),
    '片': ('pc', 1.0), '粒': ('pc', 1.0), '枚': ('pc', 1.0), 'pcs': ('pc', 1.0), 'pc': ('pc', 1.0),
}
# The secondary sort key t is required: when sorting a set/dict by length alone, ties are broken
# by Python's per-process string hash randomization -> results would not be reproducible.
_UOM_RE = '|'.join(sorted(map(re.escape, _UOM), key=lambda t: (-len(t), t)))

# 100g x 4 / 100克 X 6 / 110g*2
_MULT_A = re.compile(r'(\d+(?:\.\d+)?)\s*(' + _UOM_RE + r')\s*[x×*]\s*(\d+)', re.I)
# 4 x 100g / 5x100克
_MULT_B = re.compile(r'(\d+)\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(' + _UOM_RE + r')', re.I)
# Plain quantity + unit
_NUMU = re.compile(r'(\d+(?:\.\d+)?)\s*(' + _UOM_RE + r')(?![a-z])', re.I)
# Pack counts: 兩件裝 (two-pack) / 4件 / x6 / 原盒×12個 / 12pcs
_CN_NUM = {'一': 1, '兩': 2, '两': 2, '二': 2, '三': 3, '四': 4, '五': 5,
           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十二': 12}
_COUNT_CN = re.compile(r'([一兩两二三四五六七八九十]+)\s*(件|塊|块|個|个|入|枚|條|支|包|盒)\s*裝?')
# ⚠️ The (?!\s*unit) negative lookahead is not optional: titles often carry decorative asterisks,
# e.g. "*不同包裝版本可能隨機出貨* 150g" lets `[x×*]\s*(\d+)` capture 150 and mis-parse
# pack_count as 150. A count immediately followed by a unit is a spec, not a count.
_NOT_UNIT = r'(?!\s*(?:kg|公斤|g|克|gram|gr|ml|毫升|cc|l|升|oz|安士|盎司)\b)'
# (?!\d) blocks backtracking: without it `(\d+)` backs off to "15", the lookahead sees only "0g",
# accepts it as a non-unit and lets it through, so pack_count becomes 15. Lock the whole number
# down first, then decide whether a unit follows.
_COUNT_NUM = re.compile(r'[x×*]\s*(\d+)(?!\d)' + _NOT_UNIT + r'\s*(?:件|塊|块|個|个|片|入|枚|條|支|包|盒|pcs?)?'
                        r'|(\d+)\s*(?:件|塊|块|個|个|入|枚|條|支)\s*裝'
                        r'|原盒\s*[x×*]?\s*(\d+)'
                        r'|pack\s*of\s*(\d+)', re.I)

# Dimensions (length/width/height) must be excluded, or "長104× 寬63× 高32毫米" reads as a spec
_DIMENSION = re.compile(r'[長长寬宽高厚直徑径]\s*\d+')


def _to_base(val, uom_raw):
    base, mul = _UOM[uom_raw.lower()] if uom_raw.lower() in _UOM else _UOM.get(uom_raw, ('g', 1.0))
    return round(val * mul, 2), base


def extract_unit(row):
    """Per-item spec. Returns (value, uom, pack_count, source).

    Priority: name > pack_size > summary > description (FINDINGS §17.2: the title is the most
    trustworthy source). Measured coverage: 85.5% for name+pack+summary, 88.9% once description
    is added (FINDINGS §16).
    """
    sources = [('name', row.get('sku_name_chi')),
               ('pack_size', row.get('pack_size_chi')),
               ('summary', row.get('summary_chi')),
               ('description', row.get('description_chi'))]
    for src, raw in sources:
        if not raw:
            continue
        t = clean_title(raw) if src == 'name' else nfkc(raw)
        t = _DIMENSION.sub(' ', t)
        m = _MULT_A.search(t)
        if m:
            v, u = _to_base(float(m.group(1)), m.group(2))
            return v, u, int(m.group(3)), src
        m = _MULT_B.search(t)
        if m:
            v, u = _to_base(float(m.group(2)), m.group(3))
            return v, u, int(m.group(1)), src
        m = _NUMU.search(t)
        if m:
            v, u = _to_base(float(m.group(1)), m.group(2))
            return v, u, extract_pack_count(row), src
    return None, None, extract_pack_count(row), None


def extract_pack_count(row):
    """Pack count. ⚠️ Business rule: this field is extracted but never enters group_key."""
    # Strip marketing noise first, so symbols in copy like "隨機出貨" do not interfere
    t = ' '.join(clean_title(f) for f in [row.get('sku_name_chi'),
                                          row.get('pack_size_chi')] if f)
    t = _DIMENSION.sub(' ', t)
    m = _MULT_A.search(t)
    if m:
        return int(m.group(3))
    m = _MULT_B.search(t)
    if m:
        return int(m.group(1))
    m = _COUNT_CN.search(t)
    if m and m.group(1) in _CN_NUM:
        return _CN_NUM[m.group(1)]
    m = _COUNT_NUM.search(t)
    if m:
        for g in m.groups():
            if g:
                return int(g)
    return 1


def unit_compatible(a, b, tol=0.10):
    """Whether two specs are compatible, with tolerance — handmade soaps are often labelled
    120g±10% or 100-110g (FINDINGS §10)."""
    (va, ua), (vb, ub) = a, b
    if va is None or vb is None:
        return va is None and vb is None      # compatible only when both are unknown
    if ua != ub:
        return False
    return abs(va - vb) / max(va, vb) <= tol


# ---------------------------------------------------------------- Origin

_ORIGIN_SPLIT = re.compile(r'[/、,，]')
_ORIGIN_CANON = {
    '中國香港': '香港', '中国': '中國', '台灣': '台灣', '臺灣': '台灣',
    '阿拉伯敘利亞共和國（敘利亞）': '敘利亞', '阿拉伯敘利亞共和國': '敘利亞',
    '澳大利亞': '澳洲', '澳大利亚': '澳洲',
}


def extract_origin(row):
    """Origin -> set. Multi-valued entries (e.g. 印度/印尼) are split; missing gives an empty set.

    ⚠️ An empty set means wildcard, not "different origin". See FINDINGS §15.1 — treating missing
    as different costs 18pp of recall.
    """
    raw = row.get('origin_chi')
    if not raw:
        return frozenset()
    parts = [nfkc(p).strip() for p in _ORIGIN_SPLIT.split(nfkc(raw)) if p.strip()]
    return frozenset(_ORIGIN_CANON.get(p, p) for p in parts)


def origin_compatible(a, b):
    """NULL tolerant, intersecting on multi-valued fields."""
    if not a or not b:
        return True          # either side missing -> wildcard
    return bool(a & b)


# ---------------------------------------------------------------- Product form

# Highest priority first. FINDINGS §17 covers the fixes for three pitfalls:
#   1) classify on name only (summary is fallback)  2) strip gifts first  3) weight near specs
FORM_RULES = [
    ('accessory',       r'皂盒|皂[網网]|起泡[網网]|皂碟|皂托|皂盤|皂架|瀝水|收納盒|皂[盤盒]架'),
    ('soap_paper',      r'皂紙|香皂紙|梘紙|皂片|洗手片|紙皂|花瓣皂'),
    ('bath_salt_bomb',  r'沐浴球|入浴劑|bath ?ball|浴鹽|泡澡球|鹽粒|浴[鹽盐]'),
    ('deodorant',       r'止汗|體香劑|除臭劑|香體露|deodorant|吸汗貼|腋下.{0,4}貼'),
    ('body_lotion',     r'護手霜|身體乳|潤膚露|body ?lotion|護膚霜|乳液'),
    ('candle',          r'香[氛薰]蠟燭|candle'),
    ('laundry_soap',    r'洗衣皂|家事皂|去污皂|內衣.{0,4}(洗|皂)|洗衣'),
    ('shampoo_bar',     r'洗髮皂|髮皂|shampoo ?bar|洗髮餅|洗髮.{0,3}皂'),
    ('shampoo_liquid',  r'洗髮水|洗髮露|洗发水|shampoo'),
    ('body_wash',       r'沐浴露|沐浴乳|沐浴液|沐浴啫喱|body ?wash|shower ?gel|潔膚露|沐浴精華'),
    ('hand_wash',       r'洗手液|潔手液|洗手乳|潔手乳|hand ?wash|洗手露'),
    ('face_cleanser',   r'潔面(乳|膏|啫喱|泡|露)|洗面(乳|奶|膏)|卸妝|潔顏(乳|露)'),
    ('soap_bar',        r'香皂|肥皂|番梘|番鹼|香梘|香枧|石鹸|石鹼|手工皂|冷[製制]皂|soap|皂'),
]
_FORM_COMPILED = [(k, re.compile(p, re.I)) for k, p in FORM_RULES]


def extract_form(row):
    """product_form. Returns (form, confidence, source).

    Classify on name only; fall back to summary only when that fails (FINDINGS §17.2: using
    summary misclassifies 75 rows).
    """
    name = clean_title(row.get('sku_name_chi'))
    for k, rx in _FORM_COMPILED:
        if rx.search(name):
            return k, 1.0, 'name'
    summ = clean_title(row.get('summary_chi'))
    for k, rx in _FORM_COMPILED:
        if rx.search(summ):
            return k, 0.6, 'summary'
    return 'UNKNOWN', 0.0, None
