# -*- coding: utf-8 -*-
"""Normalisation ablation: silver recall gained per layer, to set defensible acceptance numbers."""
import re,collections,unicodedata,itertools
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))
def ean_ok(s):
    if len(s)!=13 or not s.isdigit(): return False
    d=[int(c) for c in s]; return (10-sum(d[i]*(1 if i%2==0 else 3) for i in range(12))%10)%10==d[12]
def eans_of(d):
    t=' '.join(filter(None,[d['sku_name_chi'],d['summary_chi'],d['description_chi'],d['pack_size_chi']]))
    out={m for m in re.findall(r'(?<!\d)(\d{13})(?!\d)',t) if ean_ok(m)}
    out|={p for p in d['sku_id'].split('_') if len(p)==13 and p.isdigit() and ean_ok(p)}
    return out
e2s=collections.defaultdict(set)
for d in data:
    for e in eans_of(d): e2s[e].add(d['sku_id'])
silver=set()
for e,s in e2s.items():
    for a,b in itertools.combinations(sorted(s),2): silver.add((a,b))
N=len(silver); print(f"silver-standard positive pairs: {N}\n")

NOISE=re.compile(r'平行進口(貨品|貨)?|平行进口|原裝行貨|香港行貨|行貨|現貨|熱賣|包郵|官方正品|新舊包裝隨機發(貨|出)?|不同版本包裝隨機發|包裝隨機發貨|新舊版本隨機發貨|新舊包裝交替出貨|no box|市集世界|日本市集|到期日?[:：]?\s*[\d/\.年月日]+|exp[:：]?\s*[\d/\.年月]+|最佳使用期[:：]?\s*\S+|送起泡[網袋]\S*|送皂網\S*|附起泡袋|免費送\S*',re.I)
MULT=re.compile(r'\d+\s*[x×\*]\s*\d+|[x×\*]\s*\d+\s*(件|塊|块|個|个|片|入|枚|條|支|包|盒|pcs?)?|\d+\s*(件|塊|块|個|个|入|枚|條|支)\s*裝|優惠?套?裝|兩件裝|原盒|套裝|組合|\d+\s*pcs',re.I)
SYN=[(r'番梘|番鹼|香梘|香枧|梘|肥皂|石鹸|石鹼',r'皂'),(r'冷制皂|冷製皂|手工皂',r'皂')]
SIMP={'纯':'純','制':'製','台':'臺'}

def L0(d): return d['sku_name_chi'] or ''
def L1(s): return unicodedata.normalize('NFKC',s).lower()                       # +fullwidth/case
def L2(s): return NOISE.sub(' ',s)                                              # +marketing noise
def L3(s): return MULT.sub(' ',s)                                               # +multi-pack markers
def L4(s):                                                                       # +soap synonyms
    for a,b in SYN: s=re.sub(a,b,s)
    return s
def L5(s): return re.sub(r'[^\w一-鿿%]+','',s)                                  # +drop punctuation
def L6(s):                                                                       # +token sort (bag of words)
    t=re.findall(r'[一-鿿]|[a-z0-9]+|%',s); return ''.join(sorted(set(t)))

LEVELS=[('L0 raw title',[L0]),
        ('L1 +NFKC/lowercase',[L0,L1]),
        ('L2 +strip marketing',[L0,L1,L2]),
        ('L3 +strip pack markers',[L0,L1,L2,L3]),
        ('L4 +soap synonyms',[L0,L1,L2,L3,L4]),
        ('L5 +drop punctuation',[L0,L1,L2,L3,L4,L5]),
        ('L6 +bag of words',[L0,L1,L2,L3,L4,L5,L6])]
print(f"{'normalisation level':22s} {'pairs':>8s} {'silver rec':>10s}")
print('-'*46)
for name,fns in LEVELS:
    k=collections.defaultdict(set)
    for d in data:
        s=fns[0](d)
        for f in fns[1:]: s=f(s)
        k[s].add(d['sku_id'])
    pred=set()
    for kk,v in k.items():
        if len(v)>1:
            for a,b in itertools.combinations(sorted(v),2): pred.add((a,b))
    tp=len(pred&silver)
    print(f"{name:22s} {len(pred):8d} {tp/N*100:9.1f}%")
