# -*- coding: utf-8 -*-
"""Brand alias / brand backfill feasibility probe."""
import re,collections,unicodedata,itertools
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))
def n(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s or ''))).strip()
def bkey(b): return re.sub(r"[\s'’`\.\-_&]",'',n(b)).lower()

# 1) non-brand detection: brand values that are really a shipping mode / shop name / generic word
NONBRAND_HINT=re.compile(r'直送|百貨|百货|優質|优质|生活|專門店|专门店|旗艦|旗舰|代購|代购|商行|貿易|贸易|официал|官方|正品|批發|批发')
nb=[b for b in {d['brand_name_chi'] for d in data if d['brand_name_chi']} if NONBRAND_HINT.search(b)]
print("=== suspected non-brand values ===")
for b in sorted(nb):
    c=sum(1 for d in data if d['brand_name_chi']==b); print(f"  {c:3d}  {b}")

# 2) brand missing, but a known brand can be found in the title
known={bkey(d['brand_name_chi']):d['brand_name_chi'] for d in data if d['brand_name_chi'] and len(bkey(d['brand_name_chi']))>=3}
missing=[d for d in data if not d['brand_name_chi']]
back=0; ex=[]
for d in missing:
    t=bkey(d['sku_name_chi'])
    hits=[v for k,v in known.items() if k in t]
    if hits: back+=1; ex.append((hits[0], n(d['sku_name_chi'])[:60]))
print(f"\n=== brand backfill ===\n  brand missing: {len(missing)};  known brand matched in title: {back} ({back/len(missing)*100:.1f}%)")
for h,t in ex[:12]: print(f"    -> [{h}]  {t}")

# 3) brand alias candidates: same origin, one brand string is a substring of the other / heavy overlap
brands=collections.Counter(bkey(d['brand_name_chi']) for d in data if d['brand_name_chi'])
borig=collections.defaultdict(collections.Counter)
for d in data:
    if d['brand_name_chi']: borig[bkey(d['brand_name_chi'])][n(d['origin_chi'])]+=1
print("\n=== brand alias candidates (substring relation + same main origin) ===")
bl=[b for b in brands if len(b)>=4]
for a,b in itertools.combinations(sorted(bl),2):
    if a in b or b in a:
        oa=borig[a].most_common(1)[0][0] if borig[a] else ''
        ob=borig[b].most_common(1)[0][0] if borig[b] else ''
        if oa and oa==ob: print(f"  {a}({brands[a]}) <-> {b}({brands[b]})   origin={oa}")

# 4) the Goat case: merge opportunities across brand strings but same origin and unit size
print("\n=== merge opportunities across brand strings (same origin, heavy title-token overlap) ===")
def toks(s):
    s=re.sub(r'[^一-鿿a-z0-9]','',n(s).lower()); return {s[i:i+2] for i in range(len(s)-1)}
groups=collections.defaultdict(list)
for d in data:
    if d['brand_name_chi']: groups[bkey(d['brand_name_chi'])].append(d)
cands=[]
keys=[k for k in groups if len(groups[k])>=5]
for a,b in itertools.combinations(sorted(keys),2):
    oa=borig[a].most_common(1)[0][0]; ob=borig[b].most_common(1)[0][0]
    if oa!=ob or not oa: continue
    ta=set().union(*[toks(x['sku_name_chi']) for x in groups[a]])
    tb=set().union(*[toks(x['sku_name_chi']) for x in groups[b]])
    j=len(ta&tb)/len(ta|tb)
    if j>=0.30: cands.append((round(j,3),a,len(groups[a]),b,len(groups[b]),oa))
for c in sorted(cands,reverse=True)[:12]: print("  ",c)
