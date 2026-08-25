# -*- coding: utf-8 -*-
"""LLM structured extraction (opt-in via --llm-extract, off by default).

Motivation: regex extraction has to **guess the role of every number**, and it only knows
two roles (unit size, pack count). Three classes of systematic error follow from that
(see the project review notes):
  1. `12%月桂油` vs `40%月桂油` (12% vs 40% lauric oil) -- the formulation percentage is
     stripped along with its number by variant._SPEC, so different formulations land in
     one group (9 groups in the deliverable mixed different percentages);
  2. `4PCS` parses as unit=4pc, which conflicts with the `110g 4個裝` (unit=110g) size on
     the same barcode, so the same-barcode pair is hard-vetoed and can never merge;
  3. `_mixed_bundle` looks for `+` in the raw title, mistakes the age marker `0M+` for a
     mixed bundle, and cannot separate `恆久嫩膚*3+幽蓮魅膚*2` from
     `舒緩潔淨*3+幽蓮魅膚*2` -- two different bundle compositions.

Approach: one LLM call per SKU returning structured JSON with evidence; **the LLM only
reads, the code verifies**:
  - evidence must be a substring of the source text (span grounding), otherwise the whole
    field is discarded and the regex result is kept;
  - the value must appear inside the evidence (Arabic or Chinese numerals), which blocks
    hallucinated numbers;
  - unit conversion and EAN check digits are still done in code; the LLM only decides what
    role a token plays.

⚠️ The evaluation basis is unaffected: silver negatives are built from the **regex
attribute snapshot** (label_attrs in multipass.metrics) and the frozen pack-invariance
file is read as-is -- both extraction modes are compared against the same denominator.

Caching follows the same pattern as embeddings: content-addressed on (model, schema
version, single text), so a SKU costs money once and --offline reproduces it fully after.
"""
from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request

from core.extract import _UOM
from core.norm import nfkc

API_URL = 'https://api.openai.com/v1/chat/completions'
DEFAULT_MODEL = 'gpt-5-mini'
SCHEMA_VERSION = 'sku-extract-v1'
MAX_WORKERS = 8
TIMEOUT = 90

# Chinese numerals: pack evidence is often written as `兩件裝` (two-piece pack), so value
# grounding has to recognise them.
_CN_NUM = {1: '一', 2: '兩两二', 3: '三', 4: '四', 5: '五',
           6: '六', 7: '七', 8: '八', 9: '九', 10: '十', 12: '十二'}

RESPONSE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'unit': {'type': 'object', 'additionalProperties': False, 'properties': {
            'value': {'type': ['number', 'null']},
            'uom': {'type': ['string', 'null']},
            'evidence': {'type': ['string', 'null']}},
            'required': ['value', 'uom', 'evidence']},
        'pack_count': {'type': 'object', 'additionalProperties': False, 'properties': {
            'value': {'type': ['integer', 'null']},
            'evidence': {'type': ['string', 'null']}},
            'required': ['value', 'evidence']},
        'formulation': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'name': {'type': 'string'}, 'pct': {'type': 'number'},
                           'evidence': {'type': 'string'}},
            'required': ['name', 'pct', 'evidence']}},
        'bundle_components': {'anyOf': [
            {'type': 'null'},
            {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {'name': {'type': 'string'}, 'count': {'type': 'integer'}},
                'required': ['name', 'count']}}]},
    },
    'required': ['unit', 'pack_count', 'formulation', 'bundle_components'],
}

SYSTEM_PROMPT = """You are a structured extractor for e-commerce product listings. The input is one product text from HKTVmall Hong Kong (mostly Traditional Chinese).
Extract only from the text you are given; do not guess. If a field is not stated, use null / an empty array.

Field definitions (be strict about the role each number plays):
1. unit -- the net content of a **single item**: weight (g/kg/oz) or volume (ml/l). For items sold by the sheet or piece such as 皂紙/皂片/洗淨丸,
   the number of sheets in one box is the unit ("[50片裝] 香皂紙" → value=50, uom=片).
   ⚠️ "100g x 6" → unit=100g (a single item), not 600g.
2. pack_count -- the **number of identical items/boxes** within one listing: "4PCS"/"4個裝"/"兩件裝"/"x6"/"3個/盒" → 4/4/2/6/3.
   If it is not stated, null. ⚠️ "4PCS" is a pack count, not a unit size. ⚠️ "0M+"/"3Y+" is an age range, neither a pack count nor a bundle.
   ⚠️ The sheet count of 皂紙 belongs to unit, not to pack count: "[20片裝] 香皂紙 [6件販量裝]" → unit={20,片}, pack_count=6.
3. formulation -- formulation percentages: "12%月桂油 88%橄欖油" → [{name:月桂油,pct:12},{name:橄欖油,pct:88}].
   This is part of the product's identity; a different percentage means a different product.
4. bundle_components -- a **mixed bundle** (one listing containing different scents/models):
   "盈潤煥采105g*3+幽蓮魅膚105g*2" → [{name:盈潤煥采,count:3},{name:幽蓮魅膚,count:2}].
   A multi-pack of one product ("香皂x6") is not a mixed bundle → null. An ingredient list ("12%月桂油+88%橄欖油") is not a bundle either → null.

evidence must be a **verbatim substring** of the input text (copied character for character, spaces included) so the program can verify it; the value must appear inside the evidence."""


def sku_text(row):
    """Per-SKU text for the LLM. Unlike embeddings.semantic_text, span checks need raw text."""
    return '標題: %s\n包裝欄: %s\n摘要: %s' % (
        nfkc(row.get('sku_name_chi') or ''),
        nfkc(row.get('pack_size_chi') or ''),
        nfkc(row.get('summary_chi') or '')[:200])


def _fingerprint(model, text):
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b'\0')
    h.update(SCHEMA_VERSION.encode())
    h.update(b'\0')
    h.update(text.encode('utf-8'))
    return h.hexdigest()


def _request(text, key, model):
    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                     {'role': 'user', 'content': text}],
        'response_format': {'type': 'json_schema',
                            'json_schema': {'name': 'sku_extraction', 'strict': True,
                                            'schema': RESPONSE_SCHEMA}},
        'max_completion_tokens': 2000,
        'reasoning_effort': 'minimal',
    }, ensure_ascii=False).encode('utf-8')
    last = None
    for attempt in range(4):
        req = urllib.request.Request(
            API_URL, data=payload,
            headers={'Authorization': 'Bearer ' + key,
                     'Content-Type': 'application/json'},
            method='POST')
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read().decode('utf-8'))
            usage = body.get('usage') or {}
            return json.loads(body['choices'][0]['message']['content']), {
                'prompt_tokens': int(usage.get('prompt_tokens', 0)),
                'completion_tokens': int(usage.get('completion_tokens', 0)),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:500]
            last = RuntimeError('LLM API HTTP %s: %s' % (exc.code, detail))
            if exc.code not in (429, 500, 502, 503, 504):
                raise last from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = RuntimeError('LLM API network error: %s' % exc)
        time.sleep(2 ** attempt)
    raise last


def extract_all(rows, cache_dir, env_path='.env', allow_api=True,
                model=DEFAULT_MODEL, max_workers=MAX_WORKERS):
    """Return ``(sku_id -> raw extraction JSON, meta)``. Cache hits never call the API."""
    from .embeddings import api_key

    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = {r['sku_id']: sku_text(r) for r in rows}
    paths = {sid: cache_dir / (_fingerprint(model, t)[:24] + '.json')
             for sid, t in texts.items()}

    out, missing = {}, []
    for sid, path in paths.items():
        if path.exists():
            try:
                out[sid] = json.loads(path.read_text(encoding='utf-8'))
                continue
            except ValueError:
                pass
        missing.append(sid)

    usage = collections.Counter()
    failures = collections.Counter()
    if missing:
        if not allow_api:
            raise FileNotFoundError(
                '%d SKUs have no LLM extraction cache and API calls are disabled (--offline)'
                % len(missing))
        key = api_key(env_path)
        if not key:
            raise RuntimeError('OPENAI_API_KEY is missing from the environment and %s' % env_path)

        def work(sid):
            return sid, _request(texts[sid], key, model)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for future in concurrent.futures.as_completed(
                    [pool.submit(work, sid) for sid in missing]):
                try:
                    sid, (record, use) = future.result()
                except Exception as exc:            # one failure must not sink the run
                    failures[str(exc)[:120]] += 1
                    continue
                paths[sid].write_text(json.dumps(record, ensure_ascii=False),
                                      encoding='utf-8')
                out[sid] = record
                usage.update(use)

    meta = {
        'kind': 'llm_structured_extraction',
        'model': model,
        'schema_version': SCHEMA_VERSION,
        'n_rows': len(rows),
        'n_cached': len(rows) - len(missing),
        'n_called': len(missing) - sum(failures.values()),
        'n_failed': sum(failures.values()),
        'failures': dict(failures),
        'usage': dict(usage),
        'cache_hit': not missing,
    }
    return out, meta


# ---------------------------------------------------------------- validation (code verifies)

def _norm(s):
    return re.sub(r'\s+', '', nfkc(s or '')).lower()


def _grounded(evidence, source_norm):
    return bool(evidence) and _norm(evidence) in source_norm


def _value_in_evidence(value, evidence):
    """Value grounding: an Arabic (125 / 12.5) or Chinese (兩件裝) numeral must be in evidence."""
    if value is None or not evidence:
        return False
    text = _norm(evidence)
    v = float(value)
    forms = {('%g' % v)}
    if v == int(v):
        forms.add(str(int(v)))
        cn = _CN_NUM.get(int(v))
        if cn and any(c in text for c in cn):
            return True
    return any(f in text for f in forms)


def validate(record, source_text):
    """Clean one raw LLM output into usable fields. Fields failing grounding are dropped (None)."""
    src = _norm(source_text)
    rejected = []

    unit = record.get('unit') or {}
    unit_out = None
    unit_explicit_null = unit.get('value') is None
    if not unit_explicit_null:
        uom = (unit.get('uom') or '').strip()
        if (uom.lower() in _UOM or uom in _UOM) \
                and _grounded(unit.get('evidence'), src) \
                and _value_in_evidence(unit.get('value'), unit.get('evidence')):
            base, mul = _UOM.get(uom.lower(), _UOM.get(uom))
            unit_out = (round(float(unit['value']) * mul, 2), base)
        else:
            rejected.append('unit')

    pack = record.get('pack_count') or {}
    pack_out = None
    if pack.get('value') is not None:
        if _grounded(pack.get('evidence'), src) \
                and _value_in_evidence(pack.get('value'), pack.get('evidence')) \
                and 1 <= int(pack['value']) <= 100:
            pack_out = int(pack['value'])
        else:
            rejected.append('pack_count')

    formulation = []
    for item in record.get('formulation') or ():
        name = nfkc(item.get('name') or '').strip()
        if name and _grounded(item.get('evidence'), src) \
                and _value_in_evidence(item.get('pct'), item.get('evidence')) \
                and 0 < float(item['pct']) <= 100:
            formulation.append((name, float(item['pct'])))
        else:
            rejected.append('formulation:%s' % item.get('name'))

    bundle = None
    components = record.get('bundle_components')
    if components:
        kept = [(nfkc(c.get('name') or '').strip(), int(c.get('count') or 0))
                for c in components]
        kept = [(n, c) for n, c in kept if n and c >= 1 and _norm(n) in src]
        if len(kept) >= 2:
            # Pack-count invariance applies to bundles too: 3件裝(1,1,1) and 6件裝(2,2,2)
            # are one composition at two pack scales, so component counts are normalised
            # by their GCD and the key expresses only the **ratio**.
            import math
            g = math.gcd(*[c for _, c in kept]) if len(kept) > 1 else 1
            bundle = frozenset((n, c // max(1, g)) for n, c in kept)
        else:
            rejected.append('bundle_components')

    return {'unit': unit_out, 'unit_explicit_null': unit_explicit_null,
            'pack_count': pack_out, 'formulation': frozenset(formulation),
            'bundle_key': bundle, 'rejected': rejected}


# ---------------------------------------------------------------- merge policy (conservative)

def apply_overrides(attrs, sigs, sigs_feat, by_sku, rows):
    """Merge the validated LLM fields into attrs / signatures. Returns statistics.

    Conservative principle: the LLM overrides the regex only when it is **grounded**; a
    failed check counts as having said nothing, and the regex result is kept.
      - unit: a valid, grounded size from the LLM -> override (after conversion).
        If the LLM explicitly says there is no size, the regex misread a pack count as a
        pc size (4PCS -> 4pc), and the LLM's pack count equals that pc value -> clear the
        regex size (this is the fix for Bug 2; all three conditions are required).
      - pack_count: grounded -> override.
      - formulation: written into attrs for hard_veto, and tokens such as `12%月桂油` are
        injected into the variant signatures (sigs / sigs_feat) so that both the scoring
        features and the pack-count must-link can see the formulation difference.
      - bundle_key: the component set of a mixed bundle; on rows with an LLM result it
        replaces the `+` heuristic.
    """
    text_of = {r['sku_id']: sku_text(r) for r in rows}
    stats = collections.Counter()
    for sid, record in by_sku.items():
        a = attrs.get(sid)
        if a is None:
            continue
        v = validate(record, text_of[sid])
        stats['grounding_rejected'] += len(v['rejected'])

        if v['unit'] is not None:
            new = (v['unit'][0], v['unit'][1])
            if (a['unit_value'], a['unit_uom']) != new:
                stats['unit_overridden'] += 1
            a['unit_value'], a['unit_uom'] = new
            a['unit_source'] = 'llm'
        elif (v['unit_explicit_null'] and a['unit_uom'] == 'pc'
              and v['pack_count'] is not None
              and a['unit_value'] == v['pack_count']):
            a['unit_value'], a['unit_uom'], a['unit_source'] = None, None, 'llm_cleared'
            stats['unit_cleared_pc'] += 1

        if v['pack_count'] is not None:
            if a['pack_count'] != v['pack_count']:
                stats['pack_overridden'] += 1
            a['pack_count'] = v['pack_count']

        if v['formulation']:
            a['formulation'] = v['formulation']
            tokens = frozenset('%g%%%s' % (pct, name) for name, pct in v['formulation'])
            sigs[sid] = sigs[sid] | tokens
            sigs_feat[sid] = sigs_feat[sid] | tokens
            stats['formulation_set'] += 1

        a['bundle_key'] = v['bundle_key']        # write None too: hard_veto uses the LLM
        if v['bundle_key'] is not None:
            stats['bundle_set'] += 1
    return dict(stats)
