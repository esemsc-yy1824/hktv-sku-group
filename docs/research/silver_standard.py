# -*- coding: utf-8 -*-
"""Silver-standard positives from barcodes + a naive baseline's pairwise precision/recall."""
import re, collections, unicodedata, itertools

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))
by_id={d['sku_id']:d for d in data}

def ean_ok(s):
    if len(s)!=13 or not s.isdigit(): return False
    d=[int(c) for c in s]
    return (10-sum(d[i]*(1 if i%2==0 else 3) for i in range(12))%10)%10==d[12]
def eans_of(d):
    t=' '.join(filter(None,[d['sku_name_chi'],d['summary_chi'],d['description_chi'],d['pack_size_chi']]))
    out={m for m in re.findall(r'(?<!\d)(\d{13})(?!\d)',t) if ean_ok(m)}
    out|={p for p in d['sku_id'].split('_') if len(p)==13 and p.isdigit() and ean_ok(p)}
    return out
def merchant(sid): 
    m=re.match(r'^([A-Z]\d{4})',sid); return m.group(1) if m else sid[:5]

ean2sku=collections.defaultdict(set)
for d in data:
    for e in eans_of(d): ean2sku[e].add(d['sku_id'])
print(f"barcode pool: {len(ean2sku)}; SKUs carrying a barcode: {len({s for v in ean2sku.values() for s in v})}")

# Silver-standard positive pairs: same barcode
silver_pos=set()
for e,skus in ean2sku.items():
    for a,b in itertools.combinations(sorted(skus),2): silver_pos.add((a,b))
cross=[(a,b) for a,b in silver_pos if merchant(a)!=merchant(b)]
print(f"silver-standard positive pairs: {len(silver_pos)}; of which cross-merchant: {len(cross)}")

# Bias check on the silver standard: how do SKUs with and without a barcode differ systematically
has={d['sku_id'] for d in data if eans_of(d)}
def rate(pred, subset): 
    s=[d for d in data if d['sku_id'] in subset]; return sum(1 for d in s if pred(d))/len(s)
print("\n=== silver-standard selection bias check (with vs without barcode) ===")
checks=[('brand set', lambda d: bool(d['brand_name_chi'])),
        ('pack_size set', lambda d: bool(d['pack_size_chi'])),
        ('origin set', lambda d: bool(d['origin_chi'])),
        ('description set', lambda d: bool(d['description_chi'])),
        ('name has noise', lambda d: bool(re.search(r'平行進口|行貨|現貨|熱賣|隨機發',d['sku_name_chi'] or ''))),
        ]
no=set(by_id)-has
for name,p in checks:
    print(f"  {name:16s}  with EAN {rate(p,has)*100:5.1f}%   without EAN {rate(p,no)*100:5.1f}%   diff {(rate(p,has)-rate(p,no))*100:+5.1f}pp")

# ---- naive baseline: exact match on the normalised title ----
FW=str.maketrans('０１２３４５６７８９（）＊－','0123456789()*-')
NOISE=re.compile(r'平行進口(貨品|貨)?|平行进口|原裝行貨|香港行貨|行貨|現貨|熱賣|包郵|官方正品|新舊包裝隨機發(貨|出)?|不同版本包裝隨機發|包裝隨機發貨|新舊版本隨機發貨|no box|到期日?[:：]?\s*[\d/\.年月日]+|exp[:：]?\s*\S+|最佳使用期[:：]?\s*\S+',re.I)
def title_key(d):
    s=unicodedata.normalize('NFKC',d['sku_name_chi'] or '').translate(FW).lower()
    s=NOISE.sub(' ',s)
    s=re.sub(r'[^\w一-鿿%]+','',s)
    s=re.sub(r'\d+\s*[x×\*]\s*\d+','',s)
    return s
tk=collections.defaultdict(set)
for d in data: tk[title_key(d)].add(d['sku_id'])
pred=set()
for k,v in tk.items():
    if len(v)>1:
        for a,b in itertools.combinations(sorted(v),2): pred.add((a,b))
print(f"\nnaive baseline (exact match on normalised title) predicted positive pairs: {len(pred)}")
tp=len(pred & silver_pos)
print(f"  intersection with silver standard (TP): {tp}")
print(f"  silver recall: {tp/len(silver_pos)*100:.1f}%  ({tp}/{len(silver_pos)})")
print("  note: precision cannot be computed against the silver standard (positives only, no negatives)")

# ---- baseline 2: blocking + title token Jaccard ----
def toks(d):
    s=title_key(d)
    return {s[i:i+2] for i in range(len(s)-1)}
def bnorm(b): return re.sub(r'[\s\'’`\.\-_]','',unicodedata.normalize('NFKC',(b or '')).lower())
blocks=collections.defaultdict(list)
for d in data: blocks[bnorm(d['brand_name_chi'])].append(d)
for th in (0.5,0.6,0.7,0.8):
    pred2=set()
    for b,rows in blocks.items():
        for x,y in itertools.combinations(rows,2):
            tx,ty=toks(x),toks(y)
            if not tx or not ty: continue
            j=len(tx&ty)/len(tx|ty)
            if j>=th: pred2.add(tuple(sorted((x['sku_id'],y['sku_id']))))
    tp2=len(pred2 & silver_pos)
    print(f"  [brand blocking + 2-gram Jaccard>={th}] pairs={len(pred2):6d}  silver recall={tp2/len(silver_pos)*100:5.1f}%")
