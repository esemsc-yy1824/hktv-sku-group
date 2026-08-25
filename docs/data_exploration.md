# Data exploration findings (`data/raw/soap_sku_data.xlsx`)

> Produced by: `docs/research/baseline_probe.py`. The findings are against the 1472 raw rows and still hold;
> the `raw.jsonl` export that the probe scripts read has been deleted (it duplicated the original xlsx exactly), see [docs/README.md](README.md) for how to re-run.

## 0. Data size

| Item | Value |
|---|---|
| Rows (SKUs) | **1,472** |
| Columns | 10 |
| category_code / category_path | **only 2 distinct values** |
| sku_id unique | 1472 / 1472 (100%) |

The two categories:

| category_code | category_path | Rows |
|---|---|---|
| AA13200520001 | 護理保健 > 個人護理用品 > 身體護理 > 香皂 | 965 |
| AA11850520001 | 超級市場 > 個人護理用品 > 身體護理 > 香皂 | 507 |

**Key finding**: the two codes point to leaf categories with exactly the same name (`香皂`, soap); they are just hung under different top-level categories.
That is, "the same kind of product placed into different categories by different merchants" is directly visible in this sample — category_code **cannot** be used as a hard constraint for grouping, only as a weak feature.

## 1. Field fill rates and dirtiness

| Field | Fill rate | Distinct values | Notes |
|---|---|---|---|
| category_code | 100.0% | 2 | See above, weak signal |
| category_path | 100.0% | 2 | Same as above |
| sku_id | 100.0% | 1472 | **Structured, see §2** |
| sku_name_chi | 100.0% | 1353 | The main signal; 33.5% contain marketing noise |
| brand_name_chi | 90.9% | 258 | 9.1% missing; contains aliases / non-brand values |
| pack_size_chi | **38.2%** | 247 | The dirtiest field, see §3 |
| origin_chi | 94.4% | 39 | Relatively clean, but has multi-valued entries such as `印度/印尼` |
| summary_chi | 99.9% | 1206 | Median length 88 characters |
| description_chi | 68.5% | 782 | Median length 283 characters, p90=854 |
| main_image_url | 99.8% | 1469 | No duplicate URLs (cannot dedupe by URL; needs image features) |

## 2. Major finding: sku_id embeds the merchant number and the barcode

Format: `<merchant store no>_S_<merchant's own article no>[_suffix]`

- Examples: `C0195001_S_5677525901211_H`, `H8699001_S_L100130`, `B1604001_S_4987072029022`
- The prefix has 299 distinct values → the first 5 characters (letter + 4 digits) identify the **merchant**, characters 6-8 are that merchant's store sequence number
  (e.g. `B1604001` / `B1604002` are two stores of the same merchant)
- The middle segment is all digits for 456 rows, of which **262 are 13 digits** and **214 pass the EAN-13 check digit**

Adding the barcodes that appear in the body text (e.g. `英國膚適 松焦油梘 100GRAMS（4892659051573）`, 68 rows):

**Combined, 273 / 1472 = 18.5% of SKUs can be given a valid EAN-13.**

Of these, **30 barcodes are shared by ≥2 different merchants** → this is a **natural, zero-cost source of high-confidence positive pairs**, usable directly for:
1. Weak-supervision training labels (seed positive pairs)
2. Seeds for the evaluation set

⚠️ But barcodes get dirty too: `4891080100034` appears on both a Johnson's `150g×3盒` and a `120g×3盒` product
(merchant copy-paste) → a barcode can only be a **high-weight prior**, never an irrefutable hard rule.

## 3. pack_size_chi is essentially unusable as-is

Only 38.2% is filled, and the content mixes five kinds:

| Type | Examples |
|---|---|
| Genuine unit size | `100g`, `110克 x 4件`, `50克 x2` |
| Bare counts | `1件`, `兩件裝`, `1 個`, `6件` |
| Marketing copy | `100%天然成份 法國製造`, `原裝行貨 放心選購 :)`, `栽種健康系列` |
| Expiry / batch | `到期日：20/03/2028`, `150g \| EXP: 2027`, `最佳使用期: 2027年6月30日` |
| Free gift / dimensions | `100g (送起泡網袋1個)`, `長104× 寬63× 高32毫米`, `隨機贈送2塊手工皂` |

→ Unit size must be **extracted jointly from name / pack_size / summary**, with name as the primary source.

## 4. Unit-size parseability (measured)

Results from the hand-written regex parser (`baseline_probe.py`):

- Contains a quantity+unit token: name alone 76.0%, name+pack+summary combined **91.5%**
- Actually parsed out to (unit_value, unit): **1259 / 1472 = 85.5%**
- Unit distribution: g 1217, ml 42 (this category is dominated by gram weights, as expected)
- Most common unit sizes: `100g` 522, `125g` 86, `80g` 65, `90g` 62, `110g` 40
- pack_count: 1 pc 898, 2 pcs 99, 4 pcs 97, 3 pcs 71, 6 pcs 28, 12 pcs 15

**26.8% of products carry a multi-pack / bundle marker** — exactly the population that the stakeholder's rule "a 3-pack and a 6-pack belong to the same group" has to handle.

## 5. Marketing noise

**33.5%** of product names contain the following noise phrases, which must be stripped during normalisation:

`平行進口 / 平行進口貨品 / 行貨 / 原裝行貨 / 現貨 / 熱賣 / 包郵 / 官方正品 /
新舊包裝隨機發貨 / 不同版本包裝隨機發 / 順豐 / 到期日 / EXP / 最佳使用期 /
No Box / +/-10% / 送皂網 / 送起泡網 / 附起泡袋`

## 6. Problems with the brand field

- 258 raw brand values; after simple normalisation (full-width/half-width, strip punctuation and spaces, lowercase) 259 — **the dirt is at the semantic layer, not the character layer**
- Three classes of problem:
  1. **Aliases / different spellings of the same brand**: `Goat Soap` vs `THE GOAT SKINCARE` (in fact the same Australian manufacturer), `Zum` vs `Zum Bar`
  2. **Non-brand values**: `內地直送` (a shipping method), `優質生活百貨` (a shop name), `A1`
  3. **Mixed Chinese/English**: `多芬` = Dove, `梨牌` = Pears, `蛇牌` = Schwarzkopf, `牛乳石鹼` = Cow Brand
- 9.1% are missing entirely, but the brand is often written into the product name (e.g. `La Corvette法國普羅旺斯香氛皂`, with an empty brand field)
  → we need **brand backfill + brand-alias clustering**

## 7. Category noise: what is mixed into the `香皂` (soap) leaf category

> ⚠️ **This section has been corrected.** The first draft counted over "name + summary, without stripping free gifts" and concluded "52 accessories, 161 facial cleansers,
> at least 30% are not soap in the narrow sense" — **all three numbers are wrong**.
> Free-gift copy (`送起泡網`) was counted as an accessory product, soap-sheet packaging (`皂片盒`) was also counted as an accessory,
> and `潔面皂` (facial-cleansing bar) was moved out of soap_bar (but a facial-cleansing bar *is* a soap bar and should not have been moved).
> The table below is a recount using the production extractor in `core/extract.py` (name only, free gifts stripped first).

| product_form | Rows | Share |
|---|---|---|
| **soap_bar** (soap bars for every purpose, including facial-cleansing and anti-mite bars) | **1,262** | 85.7% |
| soap_paper (soap sheets / flakes) | 45 | 3.1% |
| UNKNOWN | 45 | 3.1% |
| shampoo_bar (shampoo bar) | 36 | 2.4% |
| laundry_soap (laundry bar / household cleaning) | 27 | 1.8% |
| body_wash (shower gel) | 17 | 1.2% |
| deodorant (antiperspirant / body deodorant) | 11 | 0.7% |
| hand_wash (liquid hand soap) | 10 | 0.7% |
| bath_salt_bomb (bath salts / bath bombs) | 8 | 0.5% |
| body_lotion (hand cream / body lotion) | 4 | 0.3% |
| face_cleanser (cleansing milk / cream — **liquid**; bars do not count) | 4 | 0.3% |
| candle (scented candle) | 1 | 0.1% |
| **accessory (soap rack)** | **1** | 0.1% |
| shampoo_liquid (liquid shampoo) | 1 | 0.1% |

### Corrected conclusion

**Non-bar products account for 210/1472 = 14.3%** (not the 30% claimed in the first draft).

But that 14.3% still makes the point — a leaf category called `香皂` (soap) contains
**antiperspirants, body lotions, shampoo, scented candles and soap racks**, none of which may ever be grouped with soap itself.
→ `product_form` must still be extracted as an independent field and used as a hard constraint; the contamination is just not as extreme as the first draft claimed.

### Why the first draft got it wrong (a lesson worth recording)

| Source of error | Impact |
|---|---|
| Letting summary take part in the decision | Soap-sheet products mention `皂片盒` (soap-sheet case) in summary → classified as accessory |
| Not stripping free-gift phrases | `(送起泡網)` caused a batch of soaps to be classified as accessory |
| The `潔面\|洗面` match is too broad | `潔面皂` and `洗面皂` were moved out of soap_bar (they are soap bars to begin with) |

Accessory keywords occur only 14 times in name, but 49 times in name+summary —
**the entire 35-occurrence gap comes from free-gift and packaging descriptions in summary**.

These three are exactly where the three fixes proposed in §17 come from, and they confirm that those fixes are necessary.

There are still many spelling variants: `皂 / 肥皂 / 番梘 / 番鹼 / 香梘 / 香枧 / 梘 / 石鹸 / 石鹼 / soap`,
all of which are now in the soap_bar rule of `FORM_RULES`.

## 8. Baseline blocking experiment

Blocking on `(normalised brand, unit_size, origin)`:

- 477 blocks, largest block 88 rows
- 261 singleton blocks (54.7% of blocks, but covering only 261/1472 = 17.7% of products)
- The largest blocks:

```
88  (grabnat,          100g, 土耳其)
66  (goatsoap,         100g, 澳大利亞)
27  (<NOBRAND>,     <NOSIZE>, 中國)
25  (pelican,           80g, 日本)
25  (thegoatskincare,  100g, 澳大利亞)
24  (dindinaturals,    110g, 澳大利亞)
21  (najel,            100g, 阿拉伯敘利亞共和國（敘利亞）)
21  (nominal,          100g, 黎巴嫩)
```

**Conclusions**:
1. Blocking is highly effective — it compresses 1472×1471/2 ≈ 1.08M pairs down to ~15K within-block pairs (a ~98.6% reduction)
2. Within a block we still have to separate "scent / ingredient variants" (origianl / 柠檬 / 燕麦 / 麦卢卡蜂蜜 / 摩洛哥坚果油 / 椰子油 / 婴幼儿)
3. `goatsoap`(66) and `thegoatskincare`(25) land in two different blocks — **barcode evidence confirms these really are different manufacturers**
   (GS1 prefixes `9349254` vs `9329401`), so splitting them is **correct**. See the correction in §13.3.
   But the brand-alias problem is real on other brands (see §13.4) and still has to be solved.

## 9. Golden case: Goat Soap (the family that best illustrates the problem)

> ⚠️ **Correction**: the first draft of this section guessed that `Goat Soap` and `THE GOAT SKINCARE` were the same manufacturer. **That is wrong.**
> The barcode evidence (§13.3) shows their GS1 vendor prefixes differ (`9349254` vs `9329401`): they are **two different Australian manufacturers**,
> each selling 100g goat-milk soap. What follows describes only what happens inside the single brand `Goat Soap`.

One and the same 100g Australian goat-milk soap, in the spellings it takes across the data:

- Brand field: `Goat Soap` / empty (`THE GOAT SKINCARE` is **a different company** and is not included here)
- Name spellings: `純天然山羊奶手工皂` / `山羊奶皂` / `Goat 山羊奶肥皂` / `Goat Soap 瘦羊皂` / `Kids Goat 嬰幼兒有機山羊奶肥皂`
- Variants (the dimension that genuinely distinguishes SKUs): 原味(藍) / 檸檬香桃木(綠) / 燕麥(橙) / 麥蘆卡蜂蜜(紅) / 摩洛哥堅果油(紫) / 椰子油(粉) / 嬰幼兒有機(白) / 木瓜 / 甘油
- Pack count (does **not** distinguish SKUs): 1 pc / ×2 / ×3 / ×6 / ×12 / ×24 / `兩件裝` / `原盒×12個`
- Internal article number inside the name: `02011 / 02028 / 02035 / 02042 / 02059 / 02066 / 04053`
- Barcode inside the name: `9349254002011` — **the last 6 digits are exactly the internal article number 002011**

This family validates all of the following at once:
the barcode signal ✅, the internal-article-number signal ✅, the need to normalise pack count ✅, colour as a variant proxy but a dirty one
(`甘油(紫色)` collides with `摩洛哥堅果油 紫色`) ✅, and that brand aliases must be solved ✅

## 10. Edge cases already identified (need a product-definition ruling)

1. **Mixed-scent bundles**: `力士香皂 盈潤煥采 + 幽蓮魅膚 105g*5` — one box holds different scents.
   The stakeholder's rule "normalise multi-packs down to the unit" breaks here (there is no single unit) → suggest a group of their own, or a `is_mixed_bundle` flag
2. **Free-gift differences**: `100g` vs `100g (送起泡網袋1個)` — same unit, suggest same group
3. **Tolerance notation**: `120g±10%`, `100g (+/-10g)`, `100-110g` — common for handmade soap; needs tolerance matching rather than exact equality
4. **Multi-valued origin**: `印度/印尼`, `中國/台灣`, `日本/韓國/台灣/紐西蘭/中國`
5. **Accessories**: soap boxes, foaming nets — must never be grouped with the soap itself; isolated hard via product_form
6. **oz ↔ g conversion**: `3oz 羊奶香皂` vs `85g` are in fact the same unit size
7. **Old/new packaging**: `新舊包裝隨機發貨` — the same SKU; a packaging difference must not split the group

---

## 11. Follow-up validation (second exploration round)

### 11.1 Barcodes can "propagate" — short code → barcode suffix matching across rows

Merchants often write an internal short article number into the title: `#3288`, `(02059)`, `(86017)`, `#1727`.
Pooling all 190 valid EAN-13s in the dataset and matching these short codes against the pool by **suffix**:

- Rows with no barcode of their own but containing a short code: **91**
- Short codes that match the suffix of some barcode in the pool: **8**, and manual row-by-row checking found **all of them correct**:

```
[和光堂]   Wakodo嬰兒蜜糖梘 85g ... (86017)   -> 4987244286017
[和光堂]   蜜糖護膚 嬰兒蜜糖梘 85克 #86017     -> 4987244286017
[Goat Soap] ... (摩洛哥堅果油) 1件 100g (02059) -> 9349254002059
[Goat Soap] ...麥蘆卡蜂蜜味 100g (02042) 紅     -> 9349254002042
[Goat Soap] ...(原味)(02011)                    -> 9349254002011
[NAJEL]    蜜糖 阿勒頗天然手工古皂 #3288        -> 3760061223288
[NAJEL]    茉莉精油 阿勒頗天然手工古皂 #1727    -> 3760061221727
[梨牌]     PEARS 透明香皂 100克 1件 #8321       -> 5017306008321
```

**Significance**: a barcode is not a static "you either have one or you don't" attribute — it can **propagate** across the graph.
This is a low-cost path to push high-confidence seeds above 18.5% (and precision looks very high).
⚠️ The match rate is low (8/91) because the barcode pool is small (only 190); scaled to 3M rows the pool will be orders of magnitude larger,
but that also brings more false matches (4-digit short-code suffixes will collide in a large pool) → **brand consistency** must be added as a secondary check.

### 11.2 Product images are downloadable (image features are usable)

- `HEAD` requests all 404 (blocked by the CDN), but **`GET` with a `User-Agent` + `Referer: https://www.hktvmall.com/` all return 200**
- Spot-checked 4: `image/jpeg` 102KB, `image/jpeg` (Range supported, 206), `image/jpeg` 107KB, `image/png` 597KB
- Three hosts: `cdn-media.hktvmall.com`, `cdn-mms.hktvmall.com`, `images.hktv-img.com`

→ Image embeddings (CLIP / SigLIP) are viable as an **auxiliary signal**.
Note though: fetching + encoding 3M images is heavy engineering and belongs in a later iteration, and it should be
invoked selectively on "candidate pairs where the text signal is ambiguous" rather than on everything.

### 11.3 Variant terms can be mined automatically by statistics (but need word segmentation)

Auto-mining variant candidates as "character n-grams occurring in 5%~50% of a block", rough run:

```
玫瑰 檸檬 薄荷 薰衣草 橄欖 馬賽皂 月桂油 精油 茉莉 藍色 沐浴皂 滋潤 保濕 ...
```

**The genuine variant terms (玫瑰/檸檬/薄荷/薰衣草/茉莉/月桂油/藍色) really were mined out**,
proving that the scalability path of "auto-generating a controlled vocabulary per category" works —
which is exactly what makes "not hand-coding a soap dictionary" possible.

But a lot of noise came along with it (`包裝隨機`, `平行進口`, `00g`, `10`, `02`), which means we must:
1. First do **noise-phrase stripping** (§5) and **number / unit-size token stripping**
2. Use **Chinese word segmentation** (jieba/pkuseg + a custom dictionary) rather than raw character n-grams
3. Filter by **the difference between within-block and across-block distributions** (chi-square / mutual information / PMI) rather than by a raw frequency threshold

### 11.4 The NAJEL case: category-specific attributes

The NAJEL family (阿勒颇古皂, Aleppo soap) reveals **category-specific attributes that determine the SKU**:

- **Laurel-oil percentage**: `5% / 12% / 30% / 40%` — this is a **formulation difference and must split groups**
- Variants: 大馬士革玫瑰 / 蜜糖 / 死海泥 / 紅泥 / 黑茴香油 / 摩洛哥堅果油+火岩泥 / 茉莉精油 / 檸檬精油 / 琥珀沉香 / 黑木炭
- Unit sizes: 100g / 150g / 170g / 200g
- A "French brand" but origin is 敘利亞 (Syria) — **brand nationality ≠ origin**; the two must not be conflated
- A concrete data-entry error: `60061223820` should be `3760061223820` (the prefix `37` is missing)

→ So the attribute schema cannot be a single fixed site-wide schema; it needs **attribute slots that extend per category**
(soap has "laurel oil %", drinks have "sugared / sugar-free"; this has to be configurable/learnable).

---

## 12. Silver standard construction and naive baselines (measured)

Script: `docs/research/silver_standard.py`, output: `docs/research/silver_standard.txt`

### 12.1 Silver-standard size

| Item | Value |
|---|---|
| Barcode pool | 190 valid EAN-13s |
| SKUs with a barcode | 273 |
| **Same barcode → silver-standard positive pairs** | **193 pairs** |
| Of which **cross-merchant** pairs | **175 pairs** (90.7%) |

The high cross-merchant share is good news — the positive pairs are not one merchant re-listing its own product, but genuinely "different merchants selling the same product".

### 12.2 ⚠️ The silver standard has a systematic selection bias (must be flagged to the stakeholder)

Comparing the field characteristics of products "with a barcode" and "without a barcode":

| Characteristic | With barcode | Without barcode | Diff |
|---|---|---|---|
| brand non-empty | 92.3% | 90.6% | +1.7pp |
| pack_size non-empty | 32.2% | 39.5% | −7.3pp |
| origin non-empty | 93.8% | 94.5% | −0.7pp |
| description non-empty | 59.0% | 70.7% | **−11.8pp** |
| **Name contains marketing noise** | **49.5%** | **28.7%** | **+20.8pp** |

**Conclusion**: products with a barcode are **systematically more like "parallel-import / grey-market resale" listings** —
20.8pp more marketing noise in the title and, conversely, 11.8pp fewer descriptions.

So the silver standard is **not** a random sample of the population. Using it directly as an evaluation set would:
- Overestimate performance on "resale-type products with poorly formatted titles but complete barcodes"
- Give zero coverage of the large block of "local brands, no barcode, distinguished by description"

→ The silver standard can only be used for **(a) weak-supervision training seeds** and **(b) a lower-bound estimate of recall**;
it **cannot** be the reporting basis for final correctness/completeness. The final metrics must come from **stratified random sampling + human annotation**.
And the barcode pairs used for training **must be strictly split from the barcode pairs used for evaluation** (split by barcode hash, to avoid information leakage).

### 12.3 Naive baselines, measured (evidence that this problem is not easy)

Using the 193 silver-standard pairs for a lower-bound recall estimate:

| Method | Predicted positive pairs | Silver-standard recall |
|---|---|---|
| Normalised title **exact match** | 166 | **28.0%** |
| Brand blocking + 2-gram Jaccard ≥ 0.8 | 337 | 28.5% |
| Brand blocking + 2-gram Jaccard ≥ 0.7 | 735 | 35.2% |
| Brand blocking + 2-gram Jaccard ≥ 0.6 | 1,355 | 40.9% |
| Brand blocking + 2-gram Jaccard ≥ 0.5 | 2,342 | **42.5%** |

**This is the most important quantitative finding of this exploration**:

1. **The ceiling on pure string matching is low** — even pushing the threshold down to 0.5 (which already produces wrong merges in bulk, with predicted pairs exploding 14×),
   recall reaches only 42.5%. The way "the same product" is expressed in this data differs at the **semantic** level, not the character level.
   (Look back at Goat Soap in §9: `純天然山羊奶手工皂 (摩洛哥堅果油) 100g` and
   `【紫色】純天然山羊奶皂 - 摩洛哥堅果油100g` have very low character Jaccard, yet are obviously the same group.)
2. Dropping the threshold from 0.8 to 0.5 gains only 14pp of recall while **predicted pairs grow 7×** —
   a textbook severe precision/recall imbalance, showing that **structured attribute constraints must be introduced** to prune wrong merges,
   rather than tuning a similarity threshold.
3. Brand blocking itself already misses some pairs (brand aliases unsolved), which is part of the recall ceiling.

→ **Together these three argue that we need "attribute extraction + semantic embeddings + a learned pairwise discriminator";
pure rules / pure string matching cannot get there.** This is exactly the motivating argument to present to the stakeholder.

---

## 13. Brand handling, feasibility measured (conclusion: string methods are not enough, semantic methods are needed)

Script: `docs/research/brand_alias_probe.py`, output: `docs/research/brand_alias.txt`

### 13.1 Non-brand values can be caught with rules (this part is easy)

| Occurrences | Non-brand value | What it actually is |
|---|---|---|
| 18 | 內地直送 | Shipping method (direct from mainland China) |
| 9 | 優質生活百貨 | Shop name |
| 4 | 屯團百貨 | Shop name |
| 2 | 日豚百貨 | Shop name |

A keyword rule (直送 / 百貨 / 專門店 / 代購 / 旗艦 / 貿易 / 批發 / 官方 / 正品) covers these.
→ **Values like these should be set to NULL and trigger brand backfill**, otherwise 33 unrelated products get wrongly bound into a single "brand block".

### 13.2 ❌ Brand backfill: string matching only reaches 7.5%

Of the 134 products with a missing brand, only **10 (7.5%)** can be backfilled by "is a known brand string a substring of the title"
(the hits are the short English brands JAM, LUX, PELICAN).

A typical failure: `La Corvette法國普羅旺斯香氛皂-羊奶100g` — La Corvette never once appears as a brand value,
so it is not in the known-brand table.

→ **Brand backfill has to rely on NER / LLM extraction, not on substring matching against a known-brand table.**

### 13.3 Brand aliases: string methods cannot catch them, but **the GS1 vendor prefix in the barcode can**

#### (a) Measured failure of the string methods

| Method | Result |
|---|---|
| Substring relation + same dominant origin | Found only `pelican` ↔ `pelicanforback` |
| Cross-brand title-token 2-gram Jaccard ≥ 0.30 (same origin, ≥5 rows each) | **0 candidates** |

#### (b) ⚠️ A hypothesis the data overturned (important, worth putting in the report)

The intuition on first look at the data: `Goat Soap` (67 rows) and `THE GOAT SKINCARE` (25 rows) are both Australian-made, both 100g goat-milk soap,
and their scent ranges overlap heavily (原味 / 檸檬 / 燕麥 / 椰子 / 木瓜), so "these must be two spellings of the same company".

**The barcode evidence overturned that intuition:**

```
Goat Soap          -> 9349254002011 / ...002028 / ...002035 / ...002042 / ...002059 / ...002066 / ...004053 / ...006286
THE GOAT SKINCARE  -> 9329401000107 / ...000169 / ...000183 / ...000428 / ...000435 / ...000800 / ...000817
```

GS1 vendor prefixes **`9349254` vs `9329401`** — **two different Australian manufacturers**.
They are competitors in the same niche, not aliases of one brand. **Splitting them into two groups is correct.**

The lesson from this case (precisely the kind the stakeholder will press on):

> **High text similarity ≠ same product.** The title-token Jaccard between these two brands is only 0.100,
> but a semantic vector will very likely judge them highly similar (both are "Australian 100g handmade goat-milk soap, lemon").
> **If clustering relies on embedding similarity alone, this family gets wrongly merged, and the error is very hard to spot.**
> → There must be a **manufacturer-level / brand-level hard constraint** as a backstop; the embedding may only operate inside that constraint.

#### (c) ✅ The GS1 vendor prefix really does solve brand aliases and backfill

Mapping the first 7 digits of each product's barcode (the GS1 vendor prefix) to its brand string,
**6 of the 77 prefixes map to more than one brand spelling, and all 6 are genuine aliases or genuine backfills:**

| GS1 prefix | Brand strings it maps to | Type |
|---|---|---|
| `4976631` | `PELICAN`(22) / `Pelican For Back`(1) | Alias (sub-brand) |
| `3253581` | `L'Occitane`(5) / `L'OCCITANE EN PROVENCE`(1) | Alias (full name / short name) |
| `9556155` | `花王`(3) / `KAO  花王`(1) | Alias (mixed Chinese/English spelling) |
| `8901030` | `梨牌`(2) / `Unilever`(1) | **Alias (Pears belongs to Unilever)** |
| `6901404` | `蜂花`(7) / **empty**(1) | **Backfill** |
| `4892659` | `膚適`(2) / **empty**(1) | **Backfill** |

**Precision 6/6.** The `梨牌 ↔ Unilever` pair in particular is one that no text-similarity method could ever discover
(one is a Chinese trademark name, the other the English name of the parent company), and the barcode links them effortlessly.

→ **Conclusion: build "GS1 vendor prefix → canonical brand name" into a site-wide dictionary that accumulates incrementally.**
That dictionary has three advantages:
1. **Category-agnostic** — not soap-specific; shared across all 3M rows and more accurate the longer it runs (cold-started from the 18.5% barcode coverage,
   then expanded by the short-code propagation of §11.1)
2. **Zero annotation cost** — no hand-built brand-alias table needed
3. **Supplies positive and negative constraints at once** — same prefix → a strong must-link prior;
   different prefixes, both known → a **cannot-link hard constraint** (exactly what the Goat case needs)

### 13.4 Final recommendation for brand handling

| Problem | Method | Evidence |
|---|---|---|
| Non-brand values (shop name / shipping method) | Keyword rule sets them to NULL | §13.1, the rules are enumerable |
| Missing brand (9.1%) | ① GS1-prefix backfill ② LLM/NER extraction from the title | §13.2 strings reach only 7.5%; §13.3(c) barcodes 6/6 |
| Brand aliases | GS1-prefix clustering + LLM as a backstop | §13.3(c) |
| Brands that are **not** aliases (easy to wrongly merge) | Different GS1 prefixes → cannot-link | §13.3(b), the Goat case |

---

## 14. Normalisation ablation (an important negative result)

Script: `docs/research/norm_ablation.py`, output: `docs/research/norm_ablation.txt`

Stacking normalisation layer by layer, to see how far silver-standard recall of title exact matching can rise:

| Normalisation level | Predicted pairs | Silver-standard recall |
|---|---|---|
| L0 raw title | 153 | 24.9% |
| L1 +NFKC / lowercase | 153 | 24.9% |
| L2 +strip marketing noise (平行進口/行貨/隨機發貨/EXP…) | 153 | 24.9% |
| L3 +strip multi-pack markers (×2 / 兩件裝 / 原盒…) | 164 | 24.9% |
| L4 +normalise soap synonyms (番梘/香梘/石鹸/肥皂 → 皂) | 164 | 24.9% |
| L5 +strip punctuation and spaces | 200 | **28.0%** |
| L6 +bag of words (order-independent) | 231 | 28.0% |

### Conclusion: ⚠️ normalisation barely helps with "finding same-group products"

**Six layers of normalisation gain 3.1pp in total (24.9% → 28.0%), and every bit of it comes from L5, stripping punctuation.**
L2 marketing-noise stripping, L3 multi-pack stripping, L4 synonym normalisation — **these three layers contribute 0.0pp**.

This is a **counter-intuitive but extremely important** result. It says:

1. **The difference between two merchants' titles for the same product is not "noise" but "a completely different way of writing it".**
   Normalisation can remove noise, but it cannot remove the fundamental expressive gap between one person writing `純天然山羊奶皂 - 檸檬香 100克`
   and another writing `Goat Soap 瘦羊皂檸檬味100g*2`.

2. **Do not over-invest engineering in normalisation.** Normalisation is still required (it makes downstream attribute extraction and feature computation cleaner,
   and L3 multi-pack stripping is the direct implementation of the stakeholder's rule), but **do not expect it to lift recall**,
   and **do not use "how much recall rose after normalisation" as its acceptance criterion**.
   Normalisation should be accepted on **the coverage and accuracy of attribute extraction**, not on end-to-end recall.

3. **This is the third independent piece of evidence that "a semantic model is mandatory"** (the first two are in §12.3 and §13.3):
   - String exact matching ceilings at 28%
   - String fuzzy matching (Jaccard 0.5) ceilings at 42.5%, and precision has already collapsed by then
   - **Exhausting every normalisation trick still only gets to 28%**

   → The remaining ~72% has to be won by **attribute extraction + semantic matching**.

### Aside: L6 bag-of-words gains nothing and instead produces 31 more predicted pairs

So word order is not the main obstacle, and loosening to a bag of words introduces wrong merges — **not recommended as a matching key**.

---

## 15. Blocking recall-ceiling experiment (the most decision-relevant result of this exploration)

Script: `docs/research/blocking_recall.py`, output: `docs/research/blocking_recall.txt`

**Blocking recall = the recall ceiling of the whole pipeline**: if a truly same-group pair never lands in the same block,
no downstream model, however strong, can recover it. So this number has to be measured first.

Testing 7 blocking schemes against the 193 silver-standard positive pairs:

| Blocking scheme | Blocks | Largest block | Within-block candidate pairs | **Blocking recall (ceiling)** |
|---|---|---|---|---|
| A Brand | 259 | 134 | 25,308 | 95.9% |
| B Brand × unit size | 421 | 88 | 11,764 | 94.3% |
| **C Brand × unit size × origin** | 478 | 88 | 10,544 | **76.2%** ⚠️ |
| D Unit size × origin (no brand) | 267 | 103 | 23,010 | 79.3% |
| E Unit size only | 91 | 523 | 171,347 | 98.4% |
| F GS1 prefix ∥ brand | 314 | 113 | 20,067 | 97.4% |
| **G GS1 prefix ∥ brand × unit size** | **474** | **88** | **9,523** | **95.9%** ✅ |

### 15.1 ⚠️ Putting origin into the blocking key is wrong (recall drops from 94.3% to 76.2%)

The 46 pairs scheme C misses, by cause:

| Cause of miss | Pairs |
|---|---|
| **Origin mismatch** | **35** |
| Brand mismatch | 8 |
| Unit-size parse mismatch | 3 |

Concretely, here is what those "origin mismatches" are:

```
A: [蛇牌   | None ] Snake brand - 薰衣草舒緩涼感香皂 100g
B: [蛇牌   | 泰國 ] 薰衣草味涼感抗菌香皂 100g | 8852086002797
        ↑ same barcode, same product; one merchant just left origin blank

A: [PELICAN | 日本 ] 去黑色素去角質美臀pp蜜桃香皂 80g
B: [PELICAN | None ] 去角質淡化黑色素保濕蜜桃皂 80g
        ↑ same as above

A: [PELICAN | 日本/韓國/台灣/紐西蘭/中國 ] 臀部護理肥皂(桃之香) 80g
B: [PELICAN | 日本 ]                      去黑色素去角質美臀pp蜜桃香皂 80g
        ↑ one merchant filled in a multi-valued origin

A: [加信氏 | 印尼 ] 加信氏 香皂 4PCS（6297000323060）
B: [加信氏 | 中國 ] 加信氏 皇室香皂110g 4個裝
        ↑ same barcode, but the two merchants filled in flatly contradictory origins
```

**This is not "the origins really differ", it is that the origin field itself is unreliable**: missing (5.6%), multi-valued, merchants contradicting each other.

#### An important clarification of the stakeholder's rule

The stakeholder says "origin mismatch → different group". At the **product-semantics level** the rule is right,
but it **must not be implemented naively as "split whenever the origin field strings are unequal"**, otherwise:

- A merchant leaves origin blank → the product is isolated into a singleton group → a direct loss of completeness
- Two merchants fill in different origins for the same product → a pair that should be one group gets split

**Correct implementation**:

| Case | Handling |
|---|---|
| Both sides have a value and they match | Pass |
| **One side missing** | **Treat as a wildcard, do not block** (missing ≠ different) |
| One side is multi-valued (`日本/韓國/…`) | Pass if the sets intersect |
| Both sides have values and they conflict | **Do not split outright**; arbitrate first with the barcode / GS1 prefix; if it still conflicts, take the value most merchants wrote, mark `origin_conflict=true` and send it to human review |

→ **Origin is a "verification constraint", not a "blocking key"**. Block on (GS1 ∥ brand × unit size), and check origin at the pairwise stage with NULL tolerance.

### 15.2 ✅ Best blocking scheme: GS1 vendor prefix ∥ brand × unit size

Scheme G achieves **the highest recall and the lowest cost at the same time**:

- **Recall 95.9%** (level with brand-only scheme A)
- **Only 9,523 within-block candidate pairs** (38% of scheme A's 25,308, and 19% fewer than scheme B)
- Largest block 88 (manageable; no oversized block to drag down the pairwise stage)

Why: the GS1 vendor prefix automatically bridges brand aliases (`PELICAN` ↔ `Pelican For Back`, see §13.3c),
while unit size slices the blocks finely. **This is the one-two punch of "solve brand with the barcode, solve unit size with the parser".**

From 1.08M theoretical pairs down to 9,523 = **a 99.1% reduction**, at a cost of only 4.1% of recall.

### 15.3 What the ceiling means

**95.9% is the recall ceiling of this technical approach.**
That is, even with a perfect pairwise classifier, completeness tops out at ~96% (on the silver-standard basis).

The remaining 4.1% of misses are mainly caused by **unit-size parse failures** (85.5% coverage; the other 14.5% get no unit_size).
→ Raising unit-size parse coverage is the **only lever** for raising the ceiling, and should be the first investment.

⚠️ Note that this 95.9% is measured on the **silver standard**, and the silver standard has a selection bias (§12.2).
Blocking recall on the true population is **very likely below 95.9%**, because titles of barcode-less products are less well formatted.
This has to be stated when reporting to the stakeholder.

---

## 16. Headroom in unit-size parsing (the "only lever" identified in §15.3)

| Data source | Unit-size token coverage |
|---|---|
| name + pack_size + summary | 85.5% (1259/1472) |
| **+ description** | **88.9% (1309/1472)** |

The remaining 163 rows with no parseable unit size, by product form:

| Type | Rows | Explanation | Response |
|---|---|---|---|
| **No unit-size text at all** | 120 | `梨香小蒼蘭香皂`, `清新馬鞭草香皂`, `橡木苔岩蘭草香皂` — the merchant genuinely never wrote a weight | Unsolvable. Degrade to `unit=UNKNOWN` and mark low-confidence; allow merging only with same-brand, same-variant products that are also UNKNOWN |
| **Accessories** | 23 | `壁掛式瀝水肥皂架`, `皂片盒` — they have no weight to begin with | Handled by `product_form=accessory`, with `unit` set to `1 pc`; **should not count as a parse failure** |
| **Soap flakes / sheets** | 20 | `[20片裝]`, `（雲朵款/綠色/50片裝）`, `12粒` | Parseable once **`片/粒/枚/pcs` are added to the valid uom list** |

### Reachable-coverage estimate

```
88.9%  current (with description)
+1.4%  add 片/粒/枚/pcs as valid uom (20 rows)
+1.6%  accessories routed to the product_form branch, no longer counted as failures (23 rows)
─────
≈ 91.9%   realistic upper bound
remaining ≈ 8.1% (120 rows) is merchants genuinely not writing a unit size — missing data, not a parsing problem
```

→ **Setting the unit-size parse-coverage acceptance bar at ≥ 90% is both reachable and justified** (currently 85.5%;
it only needs the description source + count-type uom + the accessory branch).
Above that lies missing data itself, and further engineering is not warranted.

For the 8.1% that genuinely lack a unit size, the suggestions are:
1. Fall back to **image OCR** (the weight is usually printed on the packaging) — but this is optional, for a later iteration
2. Or a degraded strategy: group on `(brand, variant, form, origin)` with `unit=UNKNOWN`,
   and mark the group "unit size unknown" so the business side knows its confidence is lower

---

## 17. product_form classification: priority rules are not enough (three failure modes measured)

Script: `docs/research/form_rules.py`, output: `docs/research/form_rules.txt`

Turning the §7 product-form dictionary into **mutually exclusive priority rules** (the more specific the form, the higher its priority), run over name+summary:

| product_form | Rows |
|---|---|
| soap_bar | 1,177 |
| accessory | 96 |
| UNKNOWN | 46 |
| laundry_soap | 31 |
| shampoo_bar | 30 |
| soap_paper | 22 |
| body_wash | 20 |
| deodorant | 15 |
| hand_wash | 10 |
| face_cleanser / bath_bomb_salt | 8 / 8 |
| shampoo_liquid / hand_cream | 5 / 4 |

220 rows hit more than one form keyword and rely on priority to disambiguate. A sampled manual check found **three systematic failures**:

### 17.1 Free-gift text contamination (47 rows affected)

```
Najel 阿勒頗天然手工古皂 12%月桂油 88%橄欖油 200g (送起泡袋)
                                              ↑ classified as accessory
```

`送起泡袋 / 附起泡網 / 送皂網 / 隨機贈送` are **free gifts**, not the product itself.
→ **Free-gift phrases must be stripped before form classification.**

### 17.2 summary introduces noise (75 rows affected)

```
(50片蘋果味）日本便攜香皂紙 /兒童洗手香皂片 /外出旅行/防疫衛生殺菌/一次性番梘紙
   ↑ the title clearly says 皂紙 (soap sheet), but summary mentions 皂片盒 (soap-sheet case), so it is classified as accessory
```

→ **The title must carry far more weight than summary/description.** Recommendation: decide on name alone first, and fall back to summary only when that yields nothing.

**Measured: switching to "name only + free gifts stripped" changes the verdict on 122 rows**
(47 from free gifts, 75 from summary noise) — 55% of the 220 ambiguous samples.

### 17.3 Keyword stuffing (SEO stuffing)

```
Best Botanicals 香皂 200g*8 塊 洗手液 潔手液 香皂 潔面皂 沐浴皂 肥皂香皂 滋潤肌膚
   ↑ the product itself is a 200g×8 bar soap, but the title stuffs in 洗手液/潔手液/潔面皂/沐浴皂, so first-match calls it hand_wash
```

In this sample only 1 row stuffs ≥3 different form keywords into the title, **but that is because the soap category is itself relatively clean**.
Across 3M site-wide rows (especially mainland direct-mail products), SEO stuffing will be far more common.

### 17.4 Conclusion: rules are workable, but must be explicitly labelled a **transitional solution**

| Improvement | Benefit |
|---|---|
| Strip free-gift phrases | Fixes 47 rows |
| Use name only (summary as fallback only) | Fixes 75 rows |
| Weight form keywords near the unit-size token (`皂` next to `200g*8 塊` is more trustworthy than `洗手液` at the end of the string) | Resists SEO stuffing |

Even so, rules will still fail on SEO stuffing and on newly appearing forms.
→ **product_form should later be replaced by a learned classifier**
(trained on the high-confidence rule outputs + LLM-labelled hard cases); only that scales to the several hundred site-wide categories
— hand-writing a priority rule set per category does not scale, which runs directly against the stakeholder's top-priority requirement.

---

## 18. Quantifying the silver standard's blind spots (the most important methodological finding of this iteration)

| Item | Value |
|---|---|
| Products covered by the silver standard (positives + negatives) | **218 / 1472 = 14.8%** |
| Times the soap-sheet family (45 rows) appears in the silver standard | **0** |
| Times brand-less products (189 rows) appear in the silver standard | 2 |

**Implication**: any grouping error occurring on 85.2% of the products is invisible to the silver standard.

Actual consequences (which really happened during this iteration):
- 5 soap-sheet scents were merged into one 18-row group — silver-standard recall/precision **did not move at all**
- The change that fixed it made silver-standard recall drop — it looks like "getting worse", but it actually repaired a blind-spot error
- Two configurations each won on some silver-standard metrics, and every point of disagreement fell squarely in the blind spot

**Conclusion**: the silver standard is only good for (a) leakage-safe training and (b) coarse operating-point tuning.
**The final adjudication between configurations must rest on a human gold standard + adversarial-set regression** (`prediction/regression_checks.py`).
This is also the strongest answer, when reporting to the stakeholder, to "why do we still need human annotation".
