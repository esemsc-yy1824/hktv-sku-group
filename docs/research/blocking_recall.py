# -*- coding: utf-8 -*-
"""Blocking recall: how many silver positive pairs land in one block? (= the whole pipeline's recall ceiling)"""
import re,collections,unicodedata,itertools
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))
by={d['sku_id']:d for d in data}
def ean_ok(s):
    if len(s)!=13 or not s.isdigit(): return False
    d=[int(c) for c in s]; return (10-sum(d[i]*(1 if i%2==0 else 3) for i in range(12))%10)%10==d[12]
def eans_of(d):
    t=' '.join(filter(None,[d['sku_name_chi'],d['summary_chi'],d['description_chi'],d['pack_size_chi']]))
    o={m for m in re.findall(r'(?<!\d)(\d{13})(?!\d)',t) if ean_ok(m)}
    return o|{p for p in d['sku_id'].split('_') if len(p)==13 and p.isdigit() and ean_ok(p)}
e2s=collections.defaultdict(set)
for d in data:
    for e in eans_of(d): e2s[e].add(d['sku_id'])
silver=set()
for e,s in e2s.items():
    for a,b in itertools.combinations(sorted(s),2): silver.add((a,b))
N=len(silver)

FW=str.maketrans('０１２３４５６７８９','0123456789')
def n(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s or '')).translate(FW)).strip().lower()
def bnorm(b): return re.sub(r"[\s'’`\.\-_&]",'',n(b))
NUMU=re.compile(r'(\d+(?:\.\d+)?)\s*(kg|公斤|g|克|gram[s]?|gr|ml|毫升|cc|l|升|oz|安士|盎司)',re.I)
def usize(d):
    for f in (d['sku_name_chi'],d['pack_size_chi'],d['summary_chi']):
        t=n(f)
        m=NUMU.search(t)
        if m:
            v=float(m.group(1)); u=m.group(2).lower()
            if re.fullmatch(r'kg|公斤',u): return (v*1000,'g')
            if u in ('oz','安士','盎司'): return (round(v*28.35),'g')
            if re.fullmatch(r'ml|毫升|cc',u): return (v,'ml')
            if re.fullmatch(r'l|升',u): return (v*1000,'ml')
            return (v,'g')
    return (None,None)
def gs1(d):
    e=eans_of(d); return sorted(e)[0][:7] if e else None

SCHEMES={
 'A brand': lambda d: (bnorm(d['brand_name_chi']),),
 'B brand × size': lambda d: (bnorm(d['brand_name_chi']), usize(d)),
 'C brand × size × origin': lambda d: (bnorm(d['brand_name_chi']), usize(d), n(d['origin_chi'])),
 'D size × origin (no brand)': lambda d: (usize(d), n(d['origin_chi'])),
 'E size only': lambda d: (usize(d),),
 'F GS1 prefix/brand': lambda d: (gs1(d) or ('B:'+bnorm(d['brand_name_chi'])),),
 'G GS1 prefix/brand × size': lambda d: ((gs1(d) or ('B:'+bnorm(d['brand_name_chi']))), usize(d)),
}
print(f"silver-standard positive pairs N={N}\n")
print(f"{'blocking scheme':26s} {'blocks':>6s} {'max':>6s} {'cand pairs':>10s} {'recall (ceil)':>14s}")
print('-'*70)
for name,fn in SCHEMES.items():
    blocks=collections.defaultdict(list)
    for d in data: blocks[fn(d)].append(d['sku_id'])
    inb=set()
    for k,v in blocks.items():
        if len(v)>1:
            for a,b in itertools.combinations(sorted(v),2): inb.add((a,b))
    rec=len(inb&silver)/N
    print(f"{name:26s} {len(blocks):6d} {max(len(v) for v in blocks.values()):6d} {len(inb):10d} {rec*100:13.1f}%")

print("\n=== silver pairs missed by scheme C (why they were missed) ===")
blocks=collections.defaultdict(list)
for d in data: blocks[SCHEMES['C brand × size × origin'](d)].append(d['sku_id'])
inb=set()
for k,v in blocks.items():
    if len(v)>1:
        for a,b in itertools.combinations(sorted(v),2): inb.add((a,b))
miss=silver-inb
reasons=collections.Counter()
for a,b in miss:
    da,db=by[a],by[b]
    if bnorm(da['brand_name_chi'])!=bnorm(db['brand_name_chi']): reasons['brand mismatch']+=1
    elif usize(da)!=usize(db): reasons['unit size parse mismatch']+=1
    elif n(da['origin_chi'])!=n(db['origin_chi']): reasons['origin mismatch']+=1
    else: reasons['other']+=1
print(f"  missed {len(miss)}/{N} pairs, main causes: {dict(reasons)}")
for a,b in list(miss)[:8]:
    da,db=by[a],by[b]
    print(f"\n  A: [{da['brand_name_chi']}|{da['origin_chi']}|{usize(da)}] {da['sku_name_chi'][:52]}")
    print(f"  B: [{db['brand_name_chi']}|{db['origin_chi']}|{usize(db)}] {db['sku_name_chi'][:52]}")
