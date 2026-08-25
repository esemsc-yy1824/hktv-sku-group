# -*- coding: utf-8 -*-
"""Baseline probe: feasibility and scale of normalise -> blocking -> group, not final precision."""
import re, collections, unicodedata

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))

FW=str.maketrans('０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ（）＊－＋／',
                 '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()*-+/')
def norm(s):
    if not s: return ''
    s=unicodedata.normalize('NFKC',str(s)).translate(FW)
    s=re.sub(r'[【】\[\]（）()《》〈〉{}｛｝]',' ',s)
    s=re.sub(r'\s+',' ',s).strip().lower()
    return s

# Noise phrases (marketing copy) -- first check how much of the catalogue they cover
NOISE=[r'平行進口(貨品|貨)?',r'平行进口',r'行貨',r'原裝行貨',r'現貨',r'熱賣',r'包郵',r'官方正品',
       r'新舊包裝隨機發(貨)?',r'不同版本包裝隨機發',r'順豐',r'到期日?[:：]?\s*\S+',r'exp[:：]?\s*\S+',
       r'最佳使用期[:：]?\s*\S+',r'no box',r'\+/-\s*\d+%?g?',r'±\s*\d+%',r'送皂網\S*',r'送起泡網\S*',r'附起泡袋']
NOISE_RE=re.compile('|'.join(NOISE),re.I)
n_noise=sum(1 for d in data if NOISE_RE.search(norm(d['sku_name_chi'])))
print(f"titles carrying marketing/noise phrases: {n_noise}/{len(data)} ({n_noise/len(data)*100:.1f}%)")

# ---- unit size parsing: prefer name, then pack_size, then summary ----
UNIT={'g':'g','克':'g','gram':'g','grams':'g','gr':'g','公克':'g','kg':'g','公斤':'g',
      'ml':'ml','毫升':'ml','cc':'ml','l':'ml','升':'ml','公升':'ml',
      'oz':'oz','安士':'oz','盎司':'oz','片':'pc','枚':'pc','塊':'pc','块':'pc','個':'pc','个':'pc','pcs':'pc','pc':'pc','條':'pc','支':'pc'}
NUMU=re.compile(r'(\d+(?:\.\d+)?)\s*(kg|公斤|g|克|gram[s]?|gr|公克|ml|毫升|cc|l|升|公升|oz|安士|盎司)',re.I)
# multipliers: 100g*2, 2 x 100g, 100克 x 4件, 3塊, 12個
MULT_A=re.compile(r'(\d+(?:\.\d+)?)\s*(?:kg|公斤|g|克|gram[s]?|gr|ml|毫升|oz|安士)\s*[x×\*]\s*(\d+)',re.I)
MULT_B=re.compile(r'(\d+)\s*[x×\*]\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|g|克|gram[s]?|gr|ml|毫升|oz|安士)',re.I)
COUNT =re.compile(r'[x×\*]\s*(\d+)\s*(?:件|塊|块|個|个|片|入|枚|條|支|包|盒|pcs?)?|(\d+)\s*(?:件|塊|块|個|个|入|枚|條|支)\s*裝',re.I)

def parse_unit(d):
    """Return (unit_value_in_base, base_unit, pack_count)."""
    fields=[d['sku_name_chi'], d['pack_size_chi'], d['summary_chi']]
    for f in fields:
        t=norm(f)
        if not t: continue
        m=MULT_A.search(t) or MULT_B.search(t)
        if m:
            if m.re is MULT_A: val,cnt=float(m.group(1)),int(m.group(2))
            else: cnt,val=int(m.group(1)),float(m.group(2))
            u=NUMU.search(m.group(0))
            unit=UNIT.get(u.group(2).lower(),'g') if u else 'g'
            v=val*1000 if unit=='g' and re.search(r'kg|公斤',m.group(0),re.I) else val
            if unit=='oz': v,unit=round(val*28.35,1),'g'
            return (v,unit,cnt)
        m=NUMU.search(t)
        if m:
            val=float(m.group(1)); u=m.group(2).lower()
            unit=UNIT.get(u,'g')
            if re.search(r'kg|公斤',u): val,unit=val*1000,'g'
            if unit=='oz': val,unit=round(val*28.35,1),'g'
            if unit=='ml' and re.fullmatch(r'l|升|公升',u): val=val*1000
            c=COUNT.search(t); cnt=int(c.group(1) or c.group(2)) if c else 1
            return (val,unit,cnt)
    return (None,None,None)

parsed=[parse_unit(d) for d in data]
ok=sum(1 for p in parsed if p[0] is not None)
print(f"unit size parsed: {ok}/{len(data)} ({ok/len(data)*100:.1f}%)")
print("unit distribution:", collections.Counter(p[1] for p in parsed if p[1]).most_common())
print("most common unit sizes (top 15):", collections.Counter(f"{p[0]:g}{p[1]}" for p in parsed if p[0]).most_common(15))
print("pack_count distribution:", collections.Counter(p[2] for p in parsed if p[2]).most_common(10))

# ---- brand normalisation ----
def bnorm(b):
    b=norm(b)
    b=re.sub(r"[’'`\.\-_]",'',b)
    b=re.sub(r'\s+','',b)
    return b
NON_BRAND={'內地直送','优质生活百货','優質生活百貨','a1',''}
bn=collections.Counter(bnorm(d['brand_name_chi']) for d in data)
print(f"\nunique brands after normalisation: {len(bn)} (258 raw)")

# ---- simple blocking: brand x unit_size x origin ----
blocks=collections.defaultdict(list)
for d,p in zip(data,parsed):
    key=(bnorm(d['brand_name_chi']) or '<NOBRAND>', f"{p[0]:g}{p[1]}" if p[0] else '<NOSIZE>', norm(d['origin_chi']) or '<NOORIGIN>')
    blocks[key].append(d['sku_id'])
sizes=collections.Counter(len(v) for v in blocks.values())
print(f"\nblocking (brand×unit_size×origin): {len(blocks)} blocks")
print("block size distribution:", sorted(sizes.items())[:12], "... max=",max(len(v) for v in blocks.values()))
sing=sum(1 for v in blocks.values() if len(v)==1)
print(f"singleton blocks: {sing} ({sing/len(blocks)*100:.1f}%)  SKUs covered: {sing}/{len(data)}")
big=sorted(blocks.items(),key=lambda kv:-len(kv[1]))[:8]
print("\nlargest blocks:")
for k,v in big: print(f"  {len(v):4d}  {k}")
