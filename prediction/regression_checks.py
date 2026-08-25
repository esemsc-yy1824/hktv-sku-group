# -*- coding: utf-8 -*-
"""Adversarial regression tests (the third evaluation layer in PLAN §5.1).

Every case here is a **manually verified** real failure or success pattern, and most of them
sit in the silver standard's blind spot (silver covers only 14.8% of the listings, and the
soap-paper family appears in it exactly 0 times).
Run after every change: `python3 -m prediction.regression_checks [groups_csv]`

⚠️ Division of labour with the silver standard: silver looks at "resold items that carry a
   barcode", this file looks at "the mistakes silver cannot see". Only when both pass can a
   change be called free of robbing Peter to pay Paul.
"""
import collections
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import review_groups_csv  # noqa: E402



def load(path):
    rows = list(csv.DictReader(open(path)))
    gid = {r['sku_id']: r['group_id'] for r in rows}
    return rows, gid


def check(rows, gid):
    results = []

    def add(name, ok, detail=''):
        results.append((name, ok, detail))

    # 1. Soap-paper scents must stay apart (50片玫瑰味 vs 50片檸檬味 ...)
    paper = collections.defaultdict(collections.Counter)
    for r in rows:
        if '香皂紙' in r['sku_name'] and '50片' in r['sku_name']:
            m = re.search(r'50片(\S+?)[味）\)]', r['sku_name'])
            if m and m.group(1) not in ('裝', '/香'):
                paper[r['group_id']][m.group(1)] += 1
    mixed = [dict(c) for c in paper.values() if len(c) > 1]
    add('soap paper scents not merged', not mixed, str(mixed[:2]))

    # 2. Different GrabNat scents/ingredients must stay apart
    gn = collections.defaultdict(set)
    for r in rows:
        if r['brand_canonical'] == 'GrabNat':
            for v in ['可可脂', '迷迭香', '玫瑰', '洋甘菊', '薰衣草', '橄欖油',
                      '乳木果', '駝奶', '蘆薈', '肉桂', '蕁麻', '黑種草']:
                if v in r['sku_name']:
                    gn[r['group_id']].add(v)
    m2 = [sorted(v) for v in gn.values() if len(v) > 1]
    add('GrabNat variants not merged', not m2, str(m2[:2]))

    # 3. Bio d'Azur 125g suffix variants must stay apart (鈴蘭 / 麝香 / 我愛你媽媽 ...)
    bio = collections.defaultdict(set)
    for r in rows:
        if "Bio d'Azur" in (r['brand_canonical'] or ''):
            m = re.search(r'125g - (.+)', r['sku_name'])
            if m:
                bio[r['group_id']].add(m.group(1)[:5])
    m3 = [sorted(v) for v in bio.values() if len(v) > 1]
    add("Bio d'Azur variants not merged", not m3, str(m3[:2]))

    # 4. Goat Soap and THE GOAT SKINCARE are two different manufacturers (GS1 prefix 9349254
    #    vs 9329401) and must never share a group
    goat = {r['group_id'] for r in rows if r['brand_canonical'] == 'Goat Soap'}
    tgs = {r['group_id'] for r in rows
           if 'GOAT' in (r['brand_canonical'] or '').upper()
           and 'SKIN' in (r['brand_canonical'] or '').upper()}
    add('Goat Soap ≠ THE GOAT SKINCARE', not (goat & tgs), str(len(goat & tgs)))

    # 5. NAJEL variants (known limit: brand-unique variant pairs are only partly covered by
    #    minimal pairs, so up to 3 merged groups are tolerated)
    naj = collections.defaultdict(set)
    for r in rows:
        if '阿勒頗' in r['sku_name']:
            for v in ['紅泥', '蜜糖', '橙花', '茉莉', '黑木炭', '紫羅蘭', '死海泥', '黑茴香']:
                if v in r['sku_name']:
                    naj[r['group_id']].add(v)
    m5 = [sorted(v) for v in naj.values() if len(v) > 1]
    add('NAJEL variant merges <=3 (known limit)', len(m5) <= 3, '%d groups: %s' % (len(m5), m5[:2]))

    # 6. Goat Soap: same variant, different pack count must share a group (the stakeholder's rule)
    pairs = [('Goat Soap 瘦羊皂檸檬味100g*2（平行進口）', '純天然山羊奶皂 - 檸檬香 100克 (平行進口)'),
             ('Goat Soap 瘦羊皂原味100g*2（平行進口）', '純天然山羊奶皂 - 原味 100克 (平行進口)')]
    byname = collections.defaultdict(list)
    for r in rows:
        byname[r['sku_name']].append(r['sku_id'])
    ok6 = all(byname[a] and byname[b] and gid[byname[a][0]] == gid[byname[b][0]]
              for a, b in pairs if byname.get(a) and byname.get(b))
    add('Goat same variant grouped across pack counts', ok6)

    # 7. A Lux mixed-flavour bundle must not share a group with a single flavour
    lux_mix = [r for r in rows if '力士' in r['sku_name'] and '+' in r['sku_name']]
    lux_pure = [r for r in rows if '力士' in r['sku_name'] and '+' not in r['sku_name']
                and '盈潤煥采' in r['sku_name']]
    ok7 = all(gid[a['sku_id']] != gid[b['sku_id']] for a in lux_mix for b in lux_pure)
    add('Lux mixed bundle is its own group', ok7)
    return results


def main(path=None):
    # Default to whichever method the evaluation picked, rather than hard-coding one.
    path = path or str(review_groups_csv())
    rows, gid = load(path)
    results = check(rows, gid)
    print('Adversarial regression  (%s)' % path)
    print('-' * 60)
    n_ok = 0
    for name, ok, detail in results:
        n_ok += ok
        print('  %s  %s %s' % ('PASS' if ok else 'FAIL', name,
                               ('| ' + detail) if (detail and not ok) else ''))
    print('-' * 60)
    print('  %d / %d passed' % (n_ok, len(results)))
    return n_ok == len(results)


if __name__ == '__main__':
    ok = main(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if ok else 1)
