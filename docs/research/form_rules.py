# -*- coding: utf-8 -*-
"""product_form mutual-exclusion priority: can the order disambiguate the 403 multi-keyword rows?"""
import re,collections
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from core.dataset import load                                    # noqa: E402

# Formerly read data/exploration/raw.jsonl -- a row-for-row duplicate of the raw xlsx, now deleted.
data = load(os.path.join(ROOT, 'data', 'raw', 'soap_sku_data.xlsx'))
# Priority runs high to low: the more specific, and the less it is soap itself, the higher it sits
RULES=[
 ('accessory',      r'皂盒|皂網|起泡網|皂碟|皂托|皂盤|起泡袋|瀝水|皂架|皂盤|收納盒'),
 ('soap_paper',     r'皂紙|香皂紙|梘紙|皂片|洗手片|紙皂|片裝.*皂|皂.*\d+片'),
 ('bath_bomb_salt', r'沐浴球|入浴劑|bath ?ball|浴鹽|泡澡球|鹽粒'),
 ('deodorant',      r'止汗|體香劑|除臭劑|香體露|deodorant|吸汗貼|腋下.*貼'),
 ('hand_cream',     r'護手霜|身體乳|潤膚露|body ?lotion|護膚霜'),
 ('laundry_soap',   r'洗衣皂|家事皂|內衣.*洗|洗衣'),
 ('shampoo_bar',    r'洗髮皂|髮皂|shampoo ?bar|洗髮餅'),
 ('shampoo_liquid', r'洗髮水|洗髮露|洗发水|shampoo'),
 ('body_wash',      r'沐浴露|沐浴乳|沐浴液|沐浴啫喱|body ?wash|shower ?gel|潔膚露|沐浴精華'),
 ('hand_wash',      r'洗手液|潔手液|洗手乳|潔手乳|hand ?wash|洗手露'),
 ('face_cleanser',  r'潔面(乳|膏|啫喱|泡)|洗面(乳|奶|膏)|卸妝'),
 ('soap_bar',       r'香皂|肥皂|番梘|番鹼|香梘|香枧|石鹸|石鹼|手工皂|冷製皂|冷制皂|soap|皂'),
]
res=collections.Counter(); amb=[]
for d in data:
    t=' '.join(filter(None,[d['sku_name_chi'],d['summary_chi']]))
    hits=[k for k,p in RULES if re.search(p,t,re.I)]
    f=hits[0] if hits else 'UNKNOWN'
    res[f]+=1
    if len(hits)>1: amb.append((f,hits,(d['sku_name_chi'] or '')[:56]))
print("=== product_form distribution under the priority rules ===")
for k,v in res.most_common(): print(f"  {v:5d}  {k}")
print(f"\nrows hitting several form keywords (priority needed): {len(amb)}")
print("\n=== 18 sampled rows: did priority pick the right one? ===")
import random; random.seed(11)
for f,h,n in random.sample(amb,18): print(f"  -> {f:15s} (candidates {h})  {n}")
