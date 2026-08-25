# HKTVmall SKU Grouping Implementation

The current recommended version is **`method_5_hybrid_fusion`**, with the operating point
**auto-selected** by the release gate (`multipass.choose_threshold`, which picks `0.95` on the
current data). It combines multi-channel candidate recall, OpenAI embeddings, local hashed
TF-IDF, product-photo evidence, a pairwise GBDT, hard constraints, and cluster-level constrained
clustering.

All five methods run the same multi-channel recall pipeline; the only difference is how a product
is represented. The number is the method's identity, and the mapping lives in
`paths.REPRESENTATION`.

## Quick Reproduction

Run all of the following from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 -m prediction.lib.model
python3 -m unittest discover -s qa -p 'test_*.py' -v
python3 -m prediction.run_method_5_hybrid_fusion
python3 -m prediction.regression_checks
python3 -m prediction.evaluate_all
```

If the embedding cache is already present on this machine, the recommended result reproduces
fully offline:

```bash
python3 -m prediction.run_method_5_hybrid_fusion --offline
```

The recommended deliverables live in `output/debug/method_5_hybrid_fusion/`:

- `groups.csv`: the stable `group_id` / `group_name` for every SKU
- `report.json`: threshold curve, honest silver-standard diagnostics, and data hashes
- `model.json`: the complete GBDT, reloadable
- `candidate_pairs.csv`: candidate source and model score, for debugging
- `group_lineage.csv`: how this run's group IDs inherit from the previous result

The unified comparison across methods lives in `output/metrics.csv` and `output/debug/metrics.json`.

## Methods Run and Results

Every recall / error-rate number uses the honest diagnostic track, which disables exact-EAN
retrieval and must-link, avoiding the circular argument of defining the silver standard by
barcode and then hitting that same silver standard with the same barcode.

The five methods (item-for-item consistent with [output/metrics.csv](../output/metrics.csv)):

| Method | Silver-standard recall | Silver negative-pair wrong merges | case-control sample precision | Pack invariance | Adversarial regression | Release gate | Time | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `method_1_lexical` | 54.9% | 2.5% | 86.9% | 98.8% | 7/7 | Pass | 8.9s | 0 |
| `method_2_local_tfidf` | 54.9% | 2.2% | 88.3% | **100.0%** | 7/7 | Pass | 10.7s | 0 |
| `method_3_semantic` | 57.5% | 2.3% | 88.1% | 98.8% | 7/7 | Pass | 10.3s | 310,737 |
| `method_4_semantic_image` | 62.7% | 2.3% | 89.0% | 98.8% | 7/7 | Pass | 11.1s | 310,737 |
| **`method_5_hybrid_fusion`** | **69.4%** | 2.5% | **89.3%** | **100.0%** | **7/7** | **Pass** | 12.4s | 310,737 |

Every operating point above is auto-selected by the release gate (`release_gate_auto`): 0.98 for
m1/m3, 0.99 for m2, 0.95 for m4/m5. Method 5 was once pinned by hand at 0.98 (more conservative:
61.7% recall, 1.24% wrong merges, 93.7% precision), then returned to automatic selection by a
business decision — the trade-off is discussed in the "Method 5" section below.

### Two Optional Switches

The defaults are exactly the configuration in the table above. Behavior changes only when a
switch is passed, and the artifacts are written under `output/experiments/`, overwriting neither
the deliverables nor the `group_id` lineage.

```bash
python3 -m prediction.run_method_5_hybrid_fusion --no-form   # disable product_form
python3 -m prediction.run_method_5_hybrid_fusion --ann       # switch to IVF approximate retrieval
```

The impact of **`--no-form`** varies by method; it is not a uniform cost:

| Method | Completeness | Adversarial regression | Release gate |
|---|---|---|---|
| `method_1_lexical` | 54.9% → **63.2%** | 7/7 | Pass |
| `method_2_local_tfidf` | 54.9% → 57.0% | 7/7 | Pass |
| `method_3_semantic` | 57.5% → 62.7% | **7/7 → 4/7** | **Fail** |
| `method_4_semantic_image` | 62.7% → **65.8%** | 7/7 | Pass |
| `method_5_hybrid_fusion` | 61.7% → **55.4%** | 7/7 | Pass |

(This table was measured with method 5 at its old pinned operating point of 0.98; see the
configuration record in ablation_form_ann.txt.)

`product_form` is a hand-written priority dictionary (14 classes of regex), finalized only after
220 hard cases were checked by hand across the 1472 rows. Hand-writing one per category is
unmaintainable when scaling to the hundreds of categories on the full site, which is why this
switch exists. But the measurements show that "drop it" is not the answer: completeness actually
goes up for four of the methods when it is disabled (form is a hard veto, so a misfire blocks
genuinely identical products), while `method_3_semantic`'s adversarial regression falls to 4/7 —
form was blocking exactly those 3 hand-verified errors. The right path is to replace the
dictionary with a learned classifier, trained on the rules' high-confidence output plus
LLM-labelled hard cases.

**`--ann`** did not move a single digit of the four metrics on any of the five methods, even
though it only recovers 92.4% of the exact top-20 — the neighbours it misses would have been
filtered out by the hard-attribute checks or fallen below the threshold anyway. On 1472 rows it
is in fact slower (the fixed cost of building the k-means index); the payoff depends on scale, as
the comparisons per query drop from `n` to roughly `nprobe × √n`.

| Data size | Exact | IVF | Saved |
|---:|---:|---:|---:|
| 1,472 | 1,471 comparisons | ~309 comparisons | 79% |
| 100,000 | 99,999 comparisons | ~2,531 comparisons | 97.5% |
| 3,000,000 | 3,000,000 comparisons | ~13,856 comparisons | **99.5%** |

Evidence and full data in [docs/research/ablation_form_ann.txt](research/ablation_form_ann.txt).

The release gate is fixed at: silver negative-pair wrong merges `<=2.5%`, pack invariance `>=98%`,
and all 7 adversarial regressions passing; among the methods that clear it, prefer the one with
higher sample precision within 2pp of the best honest silver recall. All five methods clear the
gate, and `method_5_hybrid_fusion` is the current best operating point. When the data cannot be
sent to an external service, `method_2_local_tfidf` remains the best fully offline option; among
the networked methods, method 5 also reaches 100% pack invariance.

The token column still reports the **real cold-start cost** even when the local cache is hit —
that is what the method actually costs; the `api_cache_hit` column says whether this particular
run paid it again.

### Method 4: Product-Photo Signal

All the evidence is in [docs/research/image_signal.txt](research/image_signal.txt), and the probe
is re-runnable (`python3 docs/research/image_signal.py`). Three things worth remembering:

**1. "More similar image ⇒ more likely the same product" is wrong on this data.** A silver-standard
negative pair is defined as "same brand, same unit size, same pack count, disjoint barcodes" — in
practice, different scents from the same product line sharing one packaging design. The 32x32
greyscale structural similarity AUC is **0.419, below random**: the easily confused pairs look
more alike than the genuinely identical ones. Using image similarity directly as positive
evidence actively works against you.

**2. The genuinely clean signal is photo near-duplication.** When different merchants resell the
same product they often reuse the manufacturer's photo verbatim; after re-upload the CDN filename
differs but the pixels are almost identical. dhash Hamming distance:

| Radius | Positive hits | Negative false positives | Precision ratio |
|---:|---:|---:|---:|
| ≤0 | 13.0% | 0 / 645 | no false positives |
| **≤2** | **17.1%** | **2 / 645 (0.3%)** | **55x** |
| ≤4 | 17.6% | 13 / 645 (2.0%) | 8.7x |

Radius 2 is the operating point: 4pp more recall than 0, at the cost of 2 false positives; from
radius 4 on, the precision ratio collapses.

**3. The gain comes entirely from the scoring features, not from a recall channel.** Ablation
(honest track, threshold 0.98):

| Configuration | Candidate recall | Silver recall | Negative-pair wrong merges | Sample precision |
|---|---:|---:|---:|---:|
| Method 3 baseline | 98.4% | 57.5% | 2.5% | 87.4% |
| + image recall channel only | 98.4% | 57.5% | 2.5% | 87.4% |
| + image features only | 98.4% | **59.6%** | **1.2%** | **93.5%** |

Image near-duplication as a sixth recall channel **moved not one metric** — every new pair it
picked up was filtered out by `broad_compatible`'s hard-attribute checks. So the production path
does not enable that channel, and `candidates.generate`'s `image_pairs` parameter remains an
optional hook (it may be useful on data with sparser text).

The implementation gives the GBDT three features (`image_dhash_similarity` / `image_hue_cosine` /
`image_structure_cosine`) and lets the model learn the direction itself; when either side has no
image it is fed a `-1` sentinel rather than 0, so the trees can split "no image" out on its own.
1466 of the 1469 URLs downloaded successfully; the other 3 are genuine 404s. Only the descriptor
vectors are cached (about 4KB per URL); the original images never touch disk.


These are not human gold-standard results. The positive/negative ratio behind `case-control
precision` is constructed artificially and must not be read as overall production precision; the
silver standard likewise covers only the biased subpopulation that carries barcodes. Final
correctness/completeness numbers require the human annotation track described below.

## Method 5: Pairwise Late Fusion of Semantic, Lexical, and Image Signals

Method 4's honest candidate recall already reaches 98.4%, yet its final silver recall is only
62.7%, which means the bulk of the loss happens in candidate scoring and clustering, not in
recall. Method 5 therefore leaves the candidate and clustering layers untouched and adds a single
independent 4,096-dimensional hashed char/word TF-IDF cosine to the pairwise representation:

- `semantic_cosine` handles cross-spelling semantic correspondences such as 瘦羊皂 and 山羊奶皂
  (two ways of writing the same Australian Goat Soap goat-milk soap);
- `local_tfidf_cosine` preserves the rare literal evidence — scent, model number, internal short
  codes — that a semantic vector tends to smooth away;
- image dhash / hue / structure keep supplying photo-reuse and packaging-difference evidence;
- the GBDT sees all five scores separately rather than a fixed weighted average, so the direction
  of the interactions is learned out-of-fold.

Method 5's operating point is auto-selected by the release gate at `0.95`: silver recall 69.4%
(134/193), silver negative wrong merges 2.48% (16/645, one event of headroom left against the
2.5% gate), sample precision 89.3%, pack invariance 85/85.

For the record: method 5 was once pinned by hand at the more conservative `0.98` — wrong merges
8/645 (8 events of headroom), sample precision 93.7%, at the cost of silver recall dropping to
61.7% (15/193 fewer positive pairs). 0.95 and 0.98 are two in-gate operating points on the same
curve; choosing between them is a business judgement, not a technical conclusion. To return to
the conservative point, register it in `paths.FIXED_OPERATING_POINTS`, and the report's
`threshold_policy` will attest to the configuration.
In the production model, `local_tfidf_cosine` earns 16.4% of the gain importance.

## Pipeline Shared by All Five Methods

1. The attribute layer extracts brand, unit size, product form, origin, EAN, pack count, and variant signals.
2. The candidate layer takes the union of several independent recall channels: EAN, legacy block hints,
   brand + compatible unit size, pack-stripped exact title, and local or semantic vector top-k. A single
   missing field no longer blocks a pair permanently.
3. The representation layer picks lexical, local TF-IDF, semantic, image, or method 5's late fusion,
   depending on the method; every vector input excludes `pack_count`, matching the definition that a
   different pack count is still the same product.
4. The pairwise GBDT uses out-of-fold predictions to evaluate barcode families it has already seen; the production model is serialized in full.
5. Known conflicts in product form, unit size, origin, mixed sets, and variants are hard `cannot-link`; a missing value is not a conflict.
6. Exact EAN, identical pack-stripped titles, and strictly pack-invariant pairs are `must-link`, handled before any probabilistic edge.
7. Cluster-level clustering checks every cross-member hard constraint and counts missing cross edges as 0, so a single bad edge cannot cause a transitive avalanche.
8. `group_id` is decoupled from the semantic key; existing groups inherit their ID by member overlap, and new groups use a full member fingerprint with lineage recorded.

## Human Gold-Standard Evaluation (Designed, Not Yet Implemented)

⚠️ This repo has **no** code for human gold-standard evaluation. A version once existed (sampling,
an annotation CLI, B³ estimation) and was deleted because no human annotation was ever produced;
recover it from git history when needed:
`git log --diff-filter=D -- human_evaluation`. The design below is what that version implemented — follow it when rebuilding.

- **Correctness track**: apply a simple random permutation without replacement over all groups,
  fixing the order at the moment the queue is generated. That way, whether annotation stops at the
  50th item or the 400th, the annotated portion is an SRSWOR prefix of the group population and can
  be used directly to report metrics.
- **Completeness track**: a random SKU anchor plus the top-30 candidates from its block, with the annotator picking out identical products that were missed.
- **Estimator**: a Horvitz-Thompson total estimate of item-weighted B³ from the SRSWOR prefix and
  the known SKU total, with confidence intervals bootstrapped over groups plus a finite-population
  correction (members within a group are not independent).
- **Always report the "excluding singleton groups" control numbers alongside**: about 78% of the
  groups in this data are singletons, a singleton's B³ precision is always 1.0, and a degenerate
  "merge nothing" algorithm would look perfect on the overall metrics.
- **Lock the queue and the grouping file with SHA-256**; once the grouping result changes, the old queue must be invalidated and cannot be annotated alongside the new one.
- **The risk-oversampled track is only for finding bugs** and must never be mixed into the official metrics.
- Recall can only confirm genuinely identical products inside a finite candidate pool, so it must be reported explicitly as an **upper bound** on actual completeness.

Until this machinery exists, every number in `output/metrics.csv` is only a diagnostic on the
barcode silver standard — and that silver standard covers only 18.5% of the products, leaving the
errors in the remaining 81.5% completely invisible to the metrics.

## OpenAI Embeddings

`prediction/lib/embeddings.py` uses `text-embedding-3-small`, batched requests, 256-dimensional
output, and a content-addressed cache; `.env` and `.cache/` are both in `.gitignore`, and the key
is never printed. This experiment sent 1,472 product texts in 3 requests, for 310,737 input tokens.

```bash
python3 -m prediction.run_method_5_hybrid_fusion
```

This sends the pack-stripped product title, brand, product form, unit size, and a truncated
summary to an external API. Re-running with the same input hits the local cache, so there is no
repeat charge and no repeat upload.

## Scaling to 3 Million SKUs

The current exact top-k uses batched matrix multiplication, bringing memory down from `O(n²)` to
`O(batch*n)`, but the computation is still `O(n²)`. In production, partition by category/region
first and replace top-k with FAISS, HNSW, or an existing vector service.
`candidates.generate(..., semantic_neighbors=...)` already provides an entry point for external ANN
results; downstream scoring and constrained clustering only ever handle sparse candidate edges. For
large brands, the brand + unit size channel should also land on a partitionable retrieval index
rather than a Cartesian product within the brand.

## Code Map

```text
paths.py               the project's single path configuration

core/                  domain modules shared by preprocessing and prediction
├── dataset.py         reads the SKU workbook
├── silver.py          barcode-derived silver standard and pack-invariance test set
└── norm/extract/variant/group.py   normalization, attribute extraction, variants, cluster keys

prediction/lib/        machinery used only by prediction
├── candidates.py      multi-channel candidate union and the ANN interface
├── embeddings.py      OpenAI embeddings and the local cache
├── local_vectors.py   local hashed TF-IDF
├── model.py           GBDT and full serialization
├── group_ids.py       stable IDs and lineage
└── features.py        pairwise features

prediction/run_method_N_*.py     entry point for each of the five methods
prediction/engine/multipass.py   the multi-channel recall pipeline shared by all five
prediction/lib/images.py         product-photo download, cache, and descriptor vectors
prediction/lib/ann.py            IVF approximate nearest neighbour (optional, off by default)
prediction/runner.py             the output contract shared by all five entry points
prediction/engine/               implementations of the five methods
prediction/evaluate_all.py       unified method comparison and the selection gate
prediction/regression_checks.py  7 hand-verified adversarial cases
qa/                              automated tests
```
