# Technical Plan and Experiment Record

> For the currently recommended implementation and the latest metrics, see the [project README](../README.md) and
> the [implementation notes](IMPLEMENTATION.md). This document is the problem definition, the group_key design, the metric definitions and the risk register;
> the full set of measured conclusions from data exploration is in [data exploration findings](data_exploration.md).
> Every percentage in this document was **actually measured** on the 1,472-row soap sample — none of them are estimates.

---

## 0. Conclusions first

**Recommended backbone: attribute extraction → a structured group_key → ML disambiguation inside the constraints.**

```
  Raw SKU
  ─────────────────────────
  title, brand, pack_size,
  summary, description,
  image_url, sku_id
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │ ① Attribute extraction                                   │
  │   product_form · brand_canonical · variant · color       │
  │   unit_value+uom · pack_count · origin · GS1 prefix      │
  └────────────────────────┬─────────────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │ ② Blocking (wide — sets the 95.9% recall ceiling)        │
  │   blocking_key = (GS1 prefix ∥ brand) × unit_value+uom   │
  │   ⚠️ origin stays out of the blocking key (adding it     │
  │      drops recall to 76.2%, see §2.3)                    │
  │   1.08M pairs ──▶ 9,523 pairs (−99.1%)                   │
  └────────────────────────┬─────────────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │ ③ In-block ML disambiguation                             │
  │   embedding similarity + pairwise GBDT classifier        │
  │   → correlation clustering (no naive transitive closure) │
  └────────────────────────┬─────────────────────────────────┘
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │ ④ Hard-constraint adjudication (strict — this is where   │
  │   the stakeholder's rules bite)                                 │
  │   product_form / color / unit / origin_resolved conflict │
  │   → split; different GS1 prefix → cannot-link            │
  └────────────────────────┬─────────────────────────────────┘
                           ▼
                  group_id / group_name
```

> ⚠️ **Implementation correction**: the "different GS1 prefix → cannot-link" step in the diagram causes collateral damage
> if implemented on a fixed-length prefix — in practice the same product carries several manufacturer prefixes (Dove `8720182`/`8718114`,
> Pears `8901030`/`5017306`, 棕欖 (Palmolive) `4710168`/`4892368`), and the GS1 company prefix is itself variable-length. The production
> implementation deliberately does **not** treat the fixed-length prefix as a hard constraint (see the hard_veto comments in
> `prediction/engine/multipass.py`); a GS1 conflict is only a feature and a piece of diagnostic evidence.

**Why "wide first, strict second" rather than "one shot"**: putting every hard constraint into the blocking key looks simpler,
but measurement shows this drops the recall ceiling from 95.9% to 76.2% — because fields like origin are themselves missing or contradictory,
and cutting on them at the blocking stage sends large numbers of genuinely same-group items into different blocks **permanently**, beyond any later recovery.
**Recall wide, adjudicate strictly — the order cannot be reversed.**

**The one-sentence rationale**: the stakeholder's rules are fundamentally a set of **hard constraints** (mismatched origin / product form / colour / unit size = must be separated),
and hard constraints can only be expressed as **structured attributes**, never as a similarity score. But "two ways of naming the same thing" (`純天然山羊奶皂 - 檸檬香`
vs `Goat Soap 瘦羊皂檸檬味`) is a **semantic problem** that only ML can solve.
Neither half is optional — the measurements below back this up.

---

## 1. Why rules alone won't work, and neither will embeddings alone (with data)

This section is the **motivating argument** for the stakeholder: three measured results, each ruling out one shortcut.

### 1.1 ❌ Pure string matching: a 42.5% ceiling

Lower-bound recall estimated on the 193 silver-standard positive pairs built from barcodes (`docs/research/silver_standard.py`):

| Method | Predicted positive pairs | Silver-standard recall |
|---|---|---|
| Normalised title **exact match** | 166 | **28.0%** |
| Brand blocking + 2-gram Jaccard ≥ 0.8 | 337 | 28.5% |
| Brand blocking + 2-gram Jaccard ≥ 0.7 | 735 | 35.2% |
| Brand blocking + 2-gram Jaccard ≥ 0.6 | 1,355 | 40.9% |
| Brand blocking + 2-gram Jaccard ≥ 0.5 | 2,342 | **42.5%** |

Loosening the threshold from 0.8 to 0.5 buys **only 14pp of recall while the predicted pair count grows 7×** — precision is already collapsing,
and recall has not even reached half.

**And this is not because normalisation was not pushed far enough.** The normalisation ablation (FINDINGS §14) shows:

| Normalisation layer | Silver-standard recall |
|---|---|
| Raw title | 24.9% |
| +NFKC/lowercase | 24.9% |
| +strip marketing noise | 24.9% |
| +strip multi-pack markers | 24.9% |
| +soap synonym normalisation (番梘/石鹸/肥皂→皂) | 24.9% |
| +strip punctuation and whitespace | **28.0%** |
| +bag of words (order-independent) | 28.0% |

**Six layers of normalisation buy 3.1pp in total, and three of those layers contribute 0.0pp.** The differences are **semantic**,
structurally out of reach for string methods — this road ends here.

### 1.2 ❌ Pure embedding similarity: it makes big, well-hidden mistakes

The counter-example comes from the data itself (FINDINGS §13.3):

```
Goat Soap          GS1 manufacturer prefix 9349254   Australia 100g goat-milk soap 檸檬/燕麥/椰子/木瓜...
THE GOAT SKINCARE  GS1 manufacturer prefix 9329401   Australia 100g goat-milk soap 檸檬/燕麥/椰子/木瓜...
```

**These are two different Australian manufacturers** (different barcode prefixes) — competitors, not aliases.
But their category, origin, gram weight and set of scents overlap almost perfectly — **any semantic embedding will judge them highly similar**.

Clustering on similarity alone would wrongly merge these 92 items into a single blob, and:
- The error is well hidden (a manual spot check sees "they really are all Australian goat-milk soaps" and lets it through)
- The error is concentrated (92 items wrong at once; the hit to correctness comes in a block, not spread evenly)

→ **A manufacturer/brand-level hard constraint must backstop this; embeddings are only allowed to act inside the constraints.**

### 1.3 ❌ Labelling every row with an LLM: neither the cost nor the consistency works

Calling a large model once per row across 3 million rows is an order-of-magnitude cost problem; worse is **consistency** —
the same product can come back with a different spelling of `variant` in different batches, so the group_keys no longer match.
The right use of an LLM is **as a teacher for labelling and distillation**, not as the online grouping decision-maker.

---

## 2. Core design: group_key

### 2.1 Definition

```
group_key = hash(
    product_form,        # product form: soap_bar / soap_paper / body_wash / accessory ...
    brand_canonical,     # canonical brand (GS1 manufacturer prefix first, alias dictionary second)
    variant,             # variant: scent / ingredient / formula (檸檬 / 12%月桂油 / 無香)
    color,               # colour (only when colour is a variant dimension in this category)
    unit_value + uom,    # single-unit size (normalised to g / ml, oz converted, with tolerance)
    origin_resolved      # origin (⚠️ the *adjudicated* origin, not the raw field, see §2.3)
)
# ⚠️ pack_count is explicitly excluded — this is the stakeholder's rule, implemented directly
```

**Two keys, kept distinct** (a core distinction in this design):

| Key | Purpose | Composition |
|---|---|---|
| **blocking_key** | Generate candidate pairs; must be **wide** (better to over-recall than to miss) | `(GS1 manufacturer prefix ∥ brand_canonical) × unit_value+uom` |
| **group_key** | Fix the final group; must be **strict** (every one of the stakeholder's hard constraints lives here) | the full 6-tuple above |

Putting `origin` into the blocking_key drops recall from 94.3% to 76.2% (measured, see §2.3);
putting it into the group_key is mandatory (the stakeholder's rule). **Recall wide first, adjudicate strictly second** — the order cannot be reversed.

### 2.2 How each of the stakeholder's rules lands in code

| What the stakeholder said | How it is implemented | Where |
|---|---|---|
| A 3-bottle pack and a 6-bottle pack are the same group | `pack_count` is extracted but **kept out of the key**; it survives only as an attribute inside the group | Attribute extraction layer |
| Different origin → different group | `origin` goes into the key; multi-values such as `印度/印尼` are handled separately | Hard-constraint layer |
| Different product category / form → different group | `product_form` goes into the key, built as a **mutually exclusive** classification (a soap dish ≠ a soap bar) | Hard-constraint layer |
| Different colour → different group | `color` goes into the key, but only after deciding whether colour is a variant dimension in this category | Hard-constraint layer |
| Different size / ml / gram weight → different group | `unit_value+uom` goes into the key, with a **tolerance** (see §2.5) | Hard-constraint layer |

> ⚠️ **The origin rule cannot be implemented naively** — see the measured evidence in §2.3 below.

### 2.3 ⚠️ Origin must be a verification constraint, never a blocking key

This is the most counter-intuitive measured result of the whole exploration, and the one with the most decision value (FINDINGS §15).

Using the 193 silver-standard positive pairs to measure the **recall ceiling** of each blocking scheme
(if a genuinely same-group pair never lands in the same block, no downstream model, however strong, can recover it):

| Blocking scheme | In-block candidate pairs | **Blocking recall (ceiling)** |
|---|---|---|
| Brand | 25,308 | 95.9% |
| Brand × unit size | 11,764 | 94.3% |
| **Brand × unit size × origin** | 10,544 | **76.2%** ⚠️ |
| **GS1 manufacturer prefix ∥ brand × unit size** | **9,523** | **95.9%** ✅ |

**Adding origin to the blocking key drops recall from 94.3% to 76.2%.** Of the 46 pairs lost, **35 were lost to an "origin mismatch"** —
but read one by one, not a single one is a genuine difference in origin:

```
[蛇牌   | None ] Snake brand - 薰衣草舒緩涼感香皂 100g
[蛇牌   | 泰國 ] 薰衣草味涼感抗菌香皂 100g | 8852086002797     ← same barcode, one side left origin blank

[PELICAN | 日本/韓國/台灣/紐西蘭/中國 ] 臀部護理肥皂(桃之香) 80g
[PELICAN | 日本 ]                    去黑色素去角質美臀pp蜜桃香皂 80g   ← one side filled in multiple values

[加信氏 | 印尼 ] 加信氏 香皂 4PCS（6297000323060）
[加信氏 | 中國 ] 加信氏 皇室香皂110g 4個裝                      ← same barcode, merchants filed conflicting origins
```

**The stakeholder's rule is exactly right at the level of product semantics, but the `origin` field itself is unreliable** (5.6% missing, multi-valued, merchants contradicting one another).
So the rule has to land like this:

| Case | Handling | Rationale |
|---|---|---|
| Both sides present and equal | Pass | — |
| **One side missing** | **Treat as a wildcard; do not block** | Missing ≠ different; otherwise every item with no origin filled in gets stranded as a singleton |
| One side is multi-valued `日本/韓國/…` | Pass if the sets intersect | — |
| The two sides conflict | Adjudicate on barcode/GS1 first; if still in conflict take the merchant majority, set `origin_conflict=true` and send for manual review | What a merchant types is not ground truth |

### 2.4 Blocking scheme: GS1 manufacturer prefix ∥ brand × unit size

The best scheme takes the highest recall and the lowest cost at the same time:

- **95.9% recall** (= the brand-only ceiling; nothing is lost)
- **Only 9,523 candidate pairs** (62% fewer than the brand scheme, and still 19% fewer than brand × unit size)
- Largest block is 88, so no oversized block drags down the pairwise stage
- From 1.08M theoretical pairs down to 9,523 — **a 99.1% reduction**

Why: the GS1 manufacturer prefix automatically bridges brand aliases (`PELICAN` ↔ `Pelican For Back`, `梨牌` ↔ `Unilever`),
and unit size then cuts the blocks finer — **use the barcode to solve brand, use parsing to solve unit size**.

**95.9% is the recall ceiling of this route.** The remaining 4.1% is lost mostly to unit-size parsing failures
(85.5% coverage today) → **raising unit-size parsing coverage is the only lever that lifts the ceiling, and should be funded first**.

⚠️ That ceiling was measured on the **silver standard**, which is skewed towards resale listings with messy titles but complete barcodes (§5.2).
Blocking recall on the true population is **very likely below 95.9%**, and this must be stated when reporting.

### 2.5 Three details that must be handled

1. **Tolerance**: handmade soaps are often listed as `120g±10%`, `100g (+/-10g)`, `100-110g`.
   Rule: `|a-b| / max(a,b) <= 0.10` counts as the same unit size, but the relaxation applies **only within the same brand and variant**.
2. **Unit conversion**: `3oz` = 85g; everything must be normalised to g/ml before comparison (oz spellings really do occur in the data).
3. **Mixed bundles**: `力士香皂 盈潤煥采 + 幽蓮魅膚 105g*5` puts different scents in one box, so there is **no single unit**.
   Mark `is_mixed_bundle=true`, put it in its own group and keep it out of normal matching (→ see the open questions in §7).

---

## 3. Execution plan

### ① Data preparation and normalisation

| Item | Content |
|---|---|
| **Goal** | Turn dirty text into clean, computable text |
| **What** | ① Full-width/half-width and case normalisation (NFKC; no Traditional↔Simplified conversion — the normalisation ablation proves it contributes nothing to recall, and attribute extraction works directly on the original glyphs) ② Strip the 33.5% of marketing-noise phrases (平行進口/行貨/隨機發貨/到期日/EXP/送皂網…) ③ Extract and strip unit-size tokens ④ Set non-brand values (內地直送/優質生活百貨) to NULL |
| **Output** | `normalized.parquet` (every column keeps both a raw and a norm version, so anything can be traced back) |
| **Acceptance** | Unit-size parsing coverage ≥ 90% (85.5% today, verified reachable at ≈91.9%); non-brand-value cleanup recall ≥ 95% (100 rows checked by hand); barcode extraction precision 100% (guaranteed by the EAN-13 check digit) |

> The path from 85.5% to ~91.9% unit-size parsing is already worked out (FINDINGS §16):
> add `description` as a source (+3.4pp), add `片/粒/枚/pcs` to the set of valid uom (+1.4pp),
> and route accessories through the `product_form=accessory` branch so they do not count as failures (+1.6pp).
> The remaining **8.1% is merchants genuinely not stating a unit size** (e.g. `梨香小蒼蘭香皂`); that is missing data, not something to keep engineering against —
> fall back instead: `unit=UNKNOWN`, merge only with other UNKNOWNs of the same brand and variant, and flag low confidence.

> ⚠️ **The acceptance criterion deliberately avoids "end-to-end recall improvement".** Measured (FINDINGS §14):
> stacking all 6 normalisation layers moves the silver-standard recall of exact title matching only from 24.9% to 28.0%,
> and **three of those layers — stripping marketing noise, stripping multi-pack markers, and synonym normalisation — each contribute 0.0pp**.
> The value of normalisation is that it **makes downstream attribute extraction cleaner**, not that it lifts recall by itself — grading normalisation on recall misdirects the investment.

### ② Human gold-standard evaluation

| Item | Content |
|---|---|
| **Goal** | Produce statistically defensible correctness / completeness |
| **What** | ① Generate the stratified labelling queue with its **order pre-randomised**, per §5.3 ② Ship a two-track labelling tool (track A for estimation / track B for improvement, strictly separated) ③ You label in queue order and can stop whenever you like ④ The tool reports the current B³ metrics + bootstrap 95% CI + "how many more groups until the CI closes to ±3pp" at any time |
| **Output** | Labelling tool + `label_queue.csv` + metrics report (with the trade-off curve and the per-stratum table) |
| **Acceptance** | Both metrics carry a 95% CI; deliver the correctness–completeness **trade-off curve** rather than a single point; report Cohen's κ (labelling solo, re-label 30 groups after a gap to measure intra-rater agreement) |

> 💡 **Consistency matters even with a single labeller**: after a few days, **re-label** the 30 groups you labelled first and compute intra-rater κ.
> If you cannot agree with yourself (κ<0.7), the definition of "what counts as the same product" is still not pinned down —
> no model metric is trustworthy at that point, and the open questions in §7 must be settled first.

### ③ Scaling to 3 million rows

| Item | Content |
|---|---|
| **Goal** | Show the method runs site-wide at a controllable cost |
| **What** | ① **Shard by leaf category**: grouping happens only inside a category cluster (note: the categories themselves are dirty, so category merging comes first, see §6) ② Run attribute extraction as batch inference on a distilled small model ③ Shard the ANN index with incremental updates (a newly listed SKU only needs retrieval, not a rebuild) ④ Use **content addressing** for group_id (see §6) so increments stay cheap ⑤ **Stratified validation**: stratify by category, price band and merchant size and report metrics for each — this catches covariate shift early |
| **Acceptance** | A full run completes; on ≥ 5 categories with very different styles (food / beauty / appliances / apparel / household) the metrics do not fall off a cliff |

---

## 4. Metric definitions

The stakeholder asked for correctness and completeness. Both words have to be translated into unambiguous mathematical definitions.

### 4.1 Recommended definition: B-Cubed (B³)

For each item $i$, let $C(i)$ be the group the algorithm assigns it and $G(i)$ the group it truly belongs to:

$$
\text{correctness (B}^3\text{ precision)} = \frac{1}{N}\sum_{i=1}^{N} \frac{|C(i) \cap G(i)|}{|C(i)|}
\qquad
\text{completeness (B}^3\text{ recall)} = \frac{1}{N}\sum_{i=1}^{N} \frac{|C(i) \cap G(i)|}{|G(i)|}
$$

**Why B³ rather than pairwise P/R:**

| Metric | Problem |
|---|---|
| **pairwise P/R** | **Over-weights large clusters** (an n-member cluster contributes $\binom{n}{2}$ pairs). In this data 54.7% of blocks are singletons, and a singleton cluster contributes 0 pairs, so it **never enters the pairwise denominator** → an algorithm that shatters everything into singletons scores a pairwise precision of 100% (undefined / perfect), which is pure illusion |
| **cluster exact match** | Too harsh: one wrong member in a group zeroes it out, so no progress is visible while improving |
| **ARI / V-measure** | Statistically elegant, but **not explainable to the business** — they cannot answer a question like "how clean are our groups?" |
| **B³** | Every **item** carries the same weight regardless of cluster size; singleton clusters participate normally; and precision/recall map directly onto what the stakeholder called "correctness" and "completeness" ✅ |

**Pairwise P/R must still be reported as a secondary metric**, because it is more sensitive to large clusters being wrongly merged (exactly the risk in §1.2).

### 4.2 Supporting metrics (to catch what B³ misses)

| Metric | What it catches |
|---|---|
| Share of singleton groups | Whether the algorithm is being lazy and refusing to merge (a signal that completeness is artificially low) |
| Largest group size / group-size distribution | Whether an avalanche of wrong merges has happened |
| Share of cross-merchant groups | Whether grouping is actually creating business value (merging within one merchant is useless for price comparison) |
| Hard-constraint violation rate | Whether items of different origin / product form / gram weight got merged — **this must be 0** |
| group_name readability | See §6.2 |

---

## 5. How to build the evaluation sets (with no ground truth)

### 5.1 Three tiers of evaluation set

| Tier | Source | Size | Purpose | Limitation |
|---|---|---|---|---|
| **Silver standard** | Same barcode → same group | 193 pairs (175 cross-merchant) | Recall lower bound, weak supervision for training | **Systematically biased**, see 5.2 |
| **Gold standard** | Human labelling | See 5.3 | **The only basis for the final metrics** | Expensive |
| **Adversarial set** | Hand-built hard cases (same brand different scent, same category different manufacturer, multi-pack vs single) | ~200 pairs | Regression testing, so improvements do not break known hard cases | Cannot produce population-level metrics |

### 5.2 ⚠️ Selection bias in the silver standard (must be raised with the stakeholder proactively)

Measured differences between items with and without a barcode:

| Feature | With barcode | Without barcode | Diff |
|---|---|---|---|
| Name contains marketing noise | **49.5%** | 28.7% | **+20.8pp** |
| description non-empty | 59.0% | 70.7% | **−11.8pp** |
| pack_size non-empty | 32.2% | 39.5% | −7.3pp |
| brand non-empty | 92.3% | 90.6% | +1.7pp |

Items with a barcode look **systematically more like parallel-import / grey-market resale listings**.
The silver standard is therefore **not a random sample of the population**: it can serve as a recall lower bound, but **must not be the basis of the final report**.

### 5.3 Sampling design for the gold standard (tuned for "labelled by one person, quantity unknown in advance")

> Premise: you do the labelling yourself, the total is not fixed in advance, and you may stop at any moment.
> This constraint **changes the sampling design** — the classic "pick n up front, label until it is full" scheme does not apply.

#### ⚠️ Trap one: labelling the most uncertain cases first biases the estimator

The intuition is to label whatever the model is least sure about first (active learning).
But if **that same batch is used to compute the metrics**, the estimator is biased — you have systematically over-sampled the hard cases,
the correctness you compute lands below the true value, and the size of the bias is unknowable.

#### ✅ The fix: two labelling tracks, each batch doing its own job

| Track | Sampling | Purpose | Usable for metrics? |
|---|---|---|---|
| **Track A (estimation)** | Stratified **random** sampling, **order shuffled in advance** | Compute correctness / completeness | ✅ The only track that may be reported |
| **Track B (improvement)** | Uncertainty sampling / boundary cases / adversarial set | Find bugs, add training data | ❌ Must never enter the metrics |

**The key design in track A: pre-randomised order.**

The labelling list is shuffled and frozen at generation time. As a result:

> **Whether you stop at item 50 or item 400, the labelled portion is always a valid random sample.**
> The estimator stays unbiased, and the confidence interval narrows monotonically as labelling proceeds.

That is what solves the unknown quantity — **there is no need to decide up front how much to label**.

#### Stratification

Stratify by the **groups** the algorithm outputs (the labelling unit is a group, not a pair — reading a whole group is faster than reading a pair, and it yields the B³ numerator directly):

| Stratum | Why it gets its own stratum |
|---|---|
| Group size 1 (singleton) | High share, and the main hiding place for completeness misses |
| Group size 2–3 | The most common case |
| Group size 4–10 | — |
| Group size > 10 | High-risk zone for wrong merges; needs over-sampling |
| Cross-merchant or not | Only cross-merchant groups carry business value; look at them separately |
| Contains a barcode or not | Quantifies the silver standard's bias (§5.2) |

Sample each stratum with **unequal probability** (over-sample the high-risk strata), then weight by $1/\pi_i$ when estimating to recover the population → unbiased.

#### Suggested labelling tool

A small command-line or web tool doing three things:

1. Generate a fixed list from the strata above plus a random order, written to `label_queue.csv`
2. Show all members of one group at a time (name + image + unit size + origin); the labeller marks which ones do not belong to this group
3. **A "show current estimate" button, available at any time** — it prints immediately:
   ```
   137 groups labelled (track A)
   B³ correctness   0.912  [95% CI: 0.871, 0.944]
   B³ completeness  0.783  [95% CI: 0.721, 0.838]
   per-stratum contribution table / how many more groups to close the CI to ±3pp
   ```

You always have an estimate up to wherever you have reached, so you can judge "is this enough?" at any point.

### 5.4 How labelling volume maps to confidence intervals

At p≈0.9 (bootstrapped by group, averaging 3 members per group):

| Groups labelled | 95% CI half-width on B³ precision | Good enough for |
|---|---|---|
| ~50 | ±8–9pp | Order of magnitude only — is it around 80% or around 95%? |
| ~100 | ±6pp | Telling which of two versions is clearly better |
| ~200 | ±4pp | Reporting "correctness is about 90%" |
| **~400** | **±3pp** | Supporting a claim like "improved from 88% to 92%" |

**Recommendation**: label 100 groups first to get the order of magnitude, then decide whether to continue.
If the two versions' CIs already do not overlap, there is no need to label all 400.

completeness is far more expensive than correctness (it requires searching backwards for what should have merged but did not);
restrict the search to the embedding top-20 candidates and **state explicitly that this is an upper-bound estimate of recall**.

## 6. group_id and group_name

### 6.1 group_id: content-addressed, for stability and increment-friendliness

```python
group_id = "G_" + sha1(canonical_key_string)[:12]
# canonical_key_string = "form=soap_bar|brand=goat_soap|variant=lemon_myrtle|
#                         color=green|unit=100|uom=g|origin=AU"
```

**Why not an auto-increment ID**: auto-increment IDs drift on re-runs and increments — the same group is G_00042 today and G_00117 tomorrow,
which is unusable downstream. Content addressing guarantees that **the same attribute combination always yields the same id**, and adding new SKUs does not disturb existing ids.

> ⚠️ **Implementation update**: the attribute-hash scheme has a sting in the tail — change attribute extraction and every id changes with it. The production implementation switched to
> **member fingerprint + lineage inheritance** (`prediction/lib/group_ids.py`): a new group takes its id from a hash of its member set, and **inherits** an old id
> by greedy one-to-one matching against the previous run's output at ≥50% member overlap; splits and re-merges are recorded in `group_lineage.csv`.
> The effect is the same (stable across re-runs, increment-friendly), and evolving the semantic key or the extraction rules no longer rewrites already-published ids.
> The attribute hash in this section is kept as the original design draft.

⚠️ The cost: change attribute extraction (say `variant` goes from `檸檬` to `檸檬香桃木`) and the id changes.
→ A `group_id → previous_group_id` migration table has to be maintained.

### 6.2 group_name: generated from a template, human-readable

```
{brand_canonical} {variant} {product_form} {unit_value}{uom} ({origin})
```

Example: `Goat Soap 檸檬香桃木 山羊奶皂 100g (澳洲)` (lemon myrtle goat-milk soap, Australia)

**How to evaluate group_name quality** (the stakeholder asked for this column but never defined how to judge it — that is a trap):
- Readability: sample 100 names and judge by hand whether each uniquely identifies the product → pass rate
- Uniqueness: whether different group_ids ever carry the same group_name (must be 0)
- Stability: the rate at which names change across re-runs
- **Do not** use text metrics such as BLEU/ROUGE — there is no reference answer, so those numbers are meaningless

---

## 7. Open questions awaiting a decision (the business side must call these)

| # | Question | Why it must be settled first | Suggested default |
|---|---|---|---|
| 1 | **Mixed-flavour bundles** (`105g*5`, five scents in one box) — one group or split apart? | The stakeholder's "normalise multi-packs to the unit" rule breaks down here (there is no single unit) | Its own group, flagged `is_mixed_bundle` |
| 2 | **Free-gift differences** (`100g` vs `100g 送起泡網袋`) — same group? | Affects a batch of items | Same group (the unit is identical) |
| 3 | **Old vs new packaging** (`新舊包裝隨機發貨`) — same group? | Very common in the data | Same group |
| 4 | **Parallel import vs official channel** — same group? | Same origin but a different channel, with possible price/warranty differences | Same group, but keep a `channel` attribute |
| 5 | **Multiple origins** (`印度/印尼`, `中國/台灣`) — how are they handled? | 4 of the 39 origin values are multi-valued | Treat as an origin value in its own right; do not split |
| 6 | Under which categories is **colour** a variant dimension? | For soap, `藍色/綠色` (blue/green) is really a proxy for scent; for apparel, colour is a genuine variant | Configure per category |
| 7 | When one merchant lists the same item repeatedly, does merging those count as a "successful grouping"? | Affects the share-of-cross-merchant-groups business-value metric | Count it in the metrics, but report it separately |

---

## 8. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| **Barcodes used for both training and evaluation → circular reasoning** | Inflated metrics; the failure only surfaces in production | Split strictly by barcode hash; only the human gold standard counts for the final metrics |
| **Sampling labels only from large clusters** | Overstates correctness (large clusters are usually one merchant's repeat listings, which are easy calls anyway); the "should have merged but did not" misses become invisible | Stratified sampling must cover singleton and small groups; use reverse sampling for completeness |
| **Singleton clusters inflate pairwise precision** | A degenerate algorithm that merges nothing scores very well | Use B³ as the headline metric, and report the singleton-group share alongside it |
| **Thresholds tuned on 1472 rows stop working at 3 million** | The metrics collapse after the site-wide rollout | Make **per-category stratified validation** mandatory before scaling; make thresholds category-adaptive (set from quantiles of the within-category similarity distribution rather than a global constant) |
| **Handing the stakeholder a single number** | Hides the correctness/completeness trade-off, and leaves no way to discuss which setting to dial in | Deliver a **curve** (P–R points across thresholds) plus one recommended operating point, with the reasoning behind the recommendation |
| **Embeddings merge competing products** (§1.2) | Block-shaped errors that spot checks easily miss | GS1 manufacturer-prefix cannot-link hard constraint; adversarial-set regression tests |
| **The categories themselves are dirty** (14.3% non-soap items sit inside the soap category: antiperspirants, body lotions, shampoos, scented candles, soap racks) | Sharding by category sends the same product to different shards → it can never be merged | product_form takes priority over category_code; shard by form, not by the merchant's category |
| **Image-fetching cost** | Downloading and encoding 3 million images is heavy engineering | Use images only for targeted adjudication of candidate pairs whose text signal is ambiguous, never across the whole catalogue (image download verified: needs a UA + Referer, and HEAD returns 404) |
| **GTINs are assigned per tradable unit** | A multi-pack carries **its own barcode, independent of the single unit**. Treating "different barcode" as a negative label systematically teaches the model that "a 3-pack ≠ a 1-pack", in direct conflict with the stakeholder's rule | Negative-sample construction must add two filters: `pack_count` equal + unit size equal (measured: without them precision is understated at 36.7%; with them, 87.0%). Also keep the **pack-invariance regression test**; anything below 100% blocks the release |
| **A GS1 prefix identifies a company, not a brand** | One manufacturer prefix can cover several of a company's brands (e.g. `8901030` appears under both Unilever and 梨牌/Pears). Used as the canonical brand name, Dove and Pears could be fused into one brand | Use the GS1 prefix only as a **blocking key and as cannot-link evidence** (blocking is wide, so it is safe); when using it as a canonical brand name, if one prefix maps to several brands with close vote counts, fall back to not using it, or switch to the 9-digit prefix |
| **Singleton groups inflate the metrics** | A singleton group's B³ precision is always 1.0, and 80.8% of the groups in this data are singletons → a degenerate algorithm that merges nothing looks perfect | The evaluator must always report the "excluding singleton groups" comparison figure alongside |

---

## 9. Delivery scope

### 9.1 Remaining deliverables

| # | Deliverable | Content |
|---|---|---|
| 1 | ✅ **Grouping output** | `output/soap_sku_data_method_3_semantic.xlsx` — the original ten columns + `group_id` and `group_name` |
| 2 | ✅ **Code** | Normalisation and attribute extraction + multi-channel recall (blocking) + pairwise GBDT + constrained clustering, in five representations |
| 3 | 🔶 **Metrics report** | ✅ Threshold trade-off curves on the silver-standard basis (one per method, five in all) ｜ ⬜ B³ + CI on the human gold standard |
| 4 | 🔶 **Explainability material** | ✅ GBDT feature-importance table + diagnostic-feature self-evidence table ｜ ⬜ Case-by-case attribution for 20 positives + 20 negatives |
| 5 | ⬜ **Labelling tool** | The three-part toolkit was built once and then deleted, since no human labelling ever happened; the design is in [IMPLEMENTATION.md](IMPLEMENTATION.md) |
| 6 | 🔶 **Scalability argument** | ✅ The variant vocabulary is already mined automatically (zero hand-written soap dictionary) ｜ ⬜ Run it on 1 additional category |

### 9.2 Why this scope (the stakeholder has a statistics background)

- **There is real ML**: a pairwise GBDT classifier, not a pile of rules
- **Explainable**: GBDT was chosen over a deep model precisely so the **feature-importance table** can be laid out in front of the stakeholder
  — "these two items were judged same-group mainly because the GS1 prefix matches (weight 0.31) + the unit size matches (0.24) + the variant term matches (0.19)"
- **Uncertainty is quantified**: every metric carries a CI, and the deliverable is a trade-off curve rather than a single point
- **Scalability is evidenced**: item 6 exists specifically to answer the "can this reach 3 million rows?" challenge

### 9.3 Explicitly out of scope

- Distilling an LLM into a small model (at POC stage, labelling 2–3k rows with an LLM is enough; no distillation)
- Image embeddings (downloads are verified to work, but fetching and encoding 3 million images is a separate engineering project)
- Incremental updates / sharded indexes / a full site-wide run
- A learned classifier for product_form (currently the rule-based version with three pitfalls fixed, see `core/extract.py`)
