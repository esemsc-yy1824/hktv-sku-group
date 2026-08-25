<h1 align="center">Welcome to hktv-sku-group 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.9-blue.svg" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-55%20passing-brightgreen.svg" />
  <a href="#-further-reading" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/esemsc-yy1824/hktv-sku-group/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>

> Group the SKUs listed by many merchants into real products, assigning every SKU a stable `group_id` and a human-readable `group_name`.

Brand, product form, variant, unit size and origin together define product identity — **pack count does not**. A 3-pack and a 6-pack of the same soap are one product.

Five methods are implemented and measured against the same evaluation harness. The recommended one, `method_5_hybrid_fusion`, reaches **69.4% silver-standard recall at 89.3% sample precision**, with 2.48% wrong merges and 100% pack invariance across 1,472 SKUs.

### 🏠 [Homepage](https://github.com/esemsc-yy1824/hktv-sku-group)

## Prerequisites

- python >=3.9 (CI runs 3.12)
- numpy >=2.0,<3
- openpyxl >=3.1,<4
- Pillow >=10,<13

## Install

```sh
python3 -m pip install -r requirements.txt
```

## Usage

```sh
# 1. Preprocess once. Emits the silver-standard labels and the frozen pack-invariance test set.
python3 -m preprocessing.preprocess_data

# 2. Run the recommended method.
python3 -m prediction.run_method_5_hybrid_fusion

# 3. Compare every method that has been run.
python3 -m prediction.evaluate_all
```

To run all five methods:

```sh
python3 -m prediction.run_method_1_lexical
python3 -m prediction.run_method_2_local_tfidf
python3 -m prediction.run_method_3_semantic
python3 -m prediction.run_method_4_semantic_image   # downloads 1,469 product images on first run
python3 -m prediction.run_method_5_hybrid_fusion
python3 -m prediction.evaluate_all
```

Everything except the two network methods runs offline. Once the embedding and image caches exist, those run offline too:

```sh
python3 -m prediction.run_method_5_hybrid_fusion --offline
```

## Run tests

```sh
python3 -m unittest discover -s qa -p 'test_*.py' -v
```

Two further checks verify an installation end to end:

```sh
python3 -m prediction.lib.model        # GBDT self-test
python3 -m prediction.regression_checks # 7 adversarial checks
```

## 📊 Results

Item-for-item consistent with `output/metrics.csv`:

| Method | Recall | Wrong merges | Precision | Pack invariance | Adversarial | Gate | Time | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `method_1_lexical` | 54.9% | 2.5% | 86.9% | 98.8% | 7/7 | Pass | 8.9s | 0 |
| `method_2_local_tfidf` | 54.9% | 2.2% | 88.3% | **100.0%** | 7/7 | Pass | 10.7s | 0 |
| `method_3_semantic` | 57.5% | 2.3% | 88.1% | 98.8% | 7/7 | Pass | 10.3s | 310,737 |
| `method_4_semantic_image` | 62.7% | 2.3% | 89.0% | 98.8% | 7/7 | Pass | 11.1s | 310,737 |
| **`method_5_hybrid_fusion`** | **69.4%** | 2.5% | **89.3%** | **100.0%** | 7/7 | Pass | 12.4s | 310,737 |

Every number uses the **honest track**, which disables exact-EAN retrieval and must-link. Without that, the silver standard would be defined by barcode and then scored using the same barcode — a circular argument.

## ⚙️ How it works

All five methods run the same three-step pipeline. The only thing that changes is how a product is represented as a vector, which is a switch inside the engine — the method number is the identity, the representation is the parameter.

**Step 1 — Shortlist.** Comparing all 1.08 million pairs is slow and error-prone, so five recall channels run in parallel and a hit on any one admits the pair:

| Channel | How it decides |
|---|---|
| Same barcode | Identical EAN-13, the strongest evidence available |
| Brand + unit size | Same brand at the same weight |
| Cleaned title | Identical string once pack and size markers are stripped |
| Vector top-k | The 20 nearest neighbours in the method's representation |
| Legacy blocking | Manufacturer identity, size value and unit all equal |

Running five in parallel fixes the classic single-filter failure: one blank or mistyped field — origin, say — would otherwise strand a listing so it never pairs with anything.

**Step 2 — Score.** Every shortlisted pair is scored 0–1 by a gradient-boosted decision tree (180 trees, depth 3). Trees are the right tool at this size: a few hundred training pairs, wildly different feature scales, and a need to read the feature importances out loud.

**Step 3 — Group.** Pairs above the threshold link into groups by average linkage. Hard vetoes — different unit size, conflicting origin, clearly different scent — block a merge outright, and identical barcodes force one.

### Naming

`group_name` is guaranteed **one-to-one with `group_id`**: two different groups never print the same name. Three layers are applied *after* grouping finishes, so none of them can move the clustering result:

1. A variant word unique among same-named sibling groups takes the variant slot.
2. Failing that, a word no sibling group has, taken from the residual of the member titles.
3. When the text genuinely cannot tell them apart, append `#hex6` from the `group_id`.

Across 1,472 rows this took 139 duplicated names down to 0, with 6 reaching the fallback. It also fixed the variant slot printing n-gram fragments (`橄欖油乳·欖油乳木·油乳木果` → `橄欖油乳木果·蘆薈`). `qa/test_group_names.py` enforces both uniqueness and freedom from fragments.

## 🔬 The five methods

| Method | Representation | Network |
|---|---|---|
| `method_1_lexical` | Lexical features only, no vectors | No |
| `method_2_local_tfidf` | Local hashed character/word TF-IDF | No |
| `method_3_semantic` | OpenAI `text-embedding-3-small`, 256d | Yes (`--offline` available) |
| `method_4_semantic_image` | The above + primary product image | Yes (`--offline` available) |
| **`method_5_hybrid_fusion`** | Semantic + local TF-IDF + image, late fusion | Yes (`--offline` available) |

### Choosing the operating point

The threshold is **selected automatically** by the release gate (`multipass.choose_threshold`): first clear wrong merges `<= 2.5%` and pack invariance `>= 98%`, then take the threshold with the highest silver recall. This currently picks 0.98 for methods 1 and 3, 0.99 for method 2, and 0.95 for methods 4 and 5.

Method 5 was once pinned by hand at the more conservative 0.98 (recall 61.7%, wrong merges 1.24%, precision 93.7%); a business decision returned it to automatic selection. The trade-off is 8 versus 16 wrong-merge events against +7.7pp recall. It can be re-pinned at any time in `paths.FIXED_OPERATING_POINTS`.

### Why method 4 uses images, and what it does *not* assume

Method 4 uses the primary product image as extra evidence. It explicitly **does not assume "more similar image, more likely same product"** — measurement says the opposite. The silver-standard negative pairs are same-brand, same-size products differing only in fragrance, sharing one packaging design, and structural similarity scores an AUC of just **0.419**, worse than random.

What actually helps is near-duplicate detection: different merchants reusing the same manufacturer photo. When the dhash matches exactly, 13.0% of positives are hit with **0 false positives** across 645 negatives. The three image features are handed to the GBDT to weigh for itself.

### Why method 5 fuses late

Method 5 targets method 4's bottleneck — the candidate was recalled, but the pairwise decision still split it — without adding an expensive new model. It keeps two independent scores inside the same GBDT: semantic cosine aligns different spellings, while local TF-IDF cosine preserves rare surface tokens such as fragrance, model number and short codes. Both are late-fused with the three image features. The newly added local cosine takes 16.4% of the model's gain importance.

## 🚩 Optional flags

The defaults *are* the current configuration; passing a flag is what changes behaviour.

| Flag | Default | Effect |
|---|---|---|
| `--offline` | Off | Use the embedding and image caches only, never the API |
| `--no-form` | Off (form enabled) | Disable `product_form` |
| `--ann` | Off (exact retrieval) | Replace exact top-k with IVF approximate nearest neighbours |
| `--llm-extract` | Off (regex only) | LLM structured extraction overrides the regexes (methods 3/4/5 only) |

### `--llm-extract`

The LLM only reads; the code verifies. The evidence must be a substring of the original text and the value must appear inside that evidence, and anything failing validation keeps the regex result. It fixes three classes of the regex misreading a number's role: the formulation percentage in `12%月桂油` (12% laurel oil) now enters the signature and the hard veto, `4PCS` is no longer taken as a unit size, and mixed sets are judged by component configuration (`0M+` is no longer misread).

Measured on 1,472 rows, each method compared before and after at one fixed threshold (method 5 at its then-pinned 0.98):

- method 5 recall 61.7% → **64.8%**, wrong merges unchanged at 1.24%, adversarial 7/7 unchanged
- method 4 wrong merges 2.33% → **1.40%**, recall unchanged
- method 3 recall 57.5% → 55.4% (−4 borderline pairs)

Deliverable-level errors were eliminated: groups mixing laurel-oil percentages 9 → **0**, 力士 (Lux) groups mixing set configurations 1 → **0**, and the `0M+` two-pack now merges correctly.

Silver negatives are built from an attribute snapshot taken *before* the override, so both extraction modes are compared on the same denominator. Cold start costs about 1.39M tokens; behind the content-addressed cache, `--offline` reproduces it.

> ⚠️ **Before re-running `--llm-extract`:** it exposed 6 mislabelled pairs inside the frozen pack-invariance test set (20% and 30% laurel oil treated as different pack counts of the same product — what the number-stripping bug left behind when the labels were generated). Against the original denominator of 85 pairs, "after" scores only 92.9%; with those 6 removed (79 pairs), before and after are both 100%. **The frozen file and the production artefacts are unchanged** — correcting the test set changes the release measurement basis and needs explicit approval. Until then, those 6 pairs make every method fail the pack-invariance gate and automatic selection degrades with it, picking absurd thresholds like 0.50. Pin the operating point temporarily in `paths.FIXED_OPERATING_POINTS` for comparison experiments.

### `--no-form`

The impact **varies by method**; it is not a uniform cost:

| Method | Completeness | Adversarial | Gate |
|---|---|---|---|
| `method_1_lexical` | 54.9% → **63.2%** | 7/7 | Pass |
| `method_2_local_tfidf` | 54.9% → 57.0% | 7/7 | Pass |
| `method_3_semantic` | 57.5% → 62.7% | **7/7 → 4/7** | **Fail** |
| `method_4_semantic_image` | 62.7% → **65.8%** | 7/7 | Pass |
| `method_5_hybrid_fusion` | 61.7% → **55.4%** | 7/7 | Pass |

Four methods gain completeness once form is disabled — form is a hard veto, and a misfire keeps genuinely identical products apart. But `method_3_semantic` drops to 4/7 adversarial: what form was blocking is precisely those 3 manually verified errors. So "form is an unmaintainable hand-written dictionary" does not mean "just remove it"; the right path is to replace it with a learned classifier. (Measured while method 5 was pinned at 0.98; the conclusion is independent of the operating point.)

### `--ann`

Does not move **a single digit** of the four metrics on any method, even though it recovers only 92.4% of the exact top-20. On 1,472 rows it is in fact *slower*, because of the fixed cost of building the k-means index. The payoff appears at scale: comparisons per query fall from `n` to about `nprobe × √n`, a saving of 99.5% at 3 million rows.

The flag state of every run is recorded under the `config` key of `report.json`.

## 📏 How results are measured

| Instrument | What it is | Size |
|---|---|---|
| Silver standard | Positive pairs derived from identical barcodes | 193 pairs |
| Silver negatives | Same-brand, same-size pairs differing by variant | 645 pairs |
| Pack invariance | Frozen pairs that differ only in pack count | 85 pairs |
| Adversarial checks | Hand-verified cases that must not regress | 7 checks |

**The pack-invariance test set is frozen** into `data/labels/pack_invariance_pairs.csv` during preprocessing; evaluation reads the file and never recomputes it. The reason is worth stating: the pairs are drawn by bucketing on variant signatures, so the denominator derives from the variant vocabulary. Any change that shrinks the vocabulary shrinks the denominator with it, kicks the newly failing pairs out of the test set, and the metric prints 100% regardless. We watched it happen — one change to `min_brand_count` and the denominator quietly went from 85 to 84. Frozen, any change is an explicit file change, visible in `git diff`.

**Release rule:** wrong merges on silver negatives `<= 2.5%`, pack invariance `>= 98%`, and all seven adversarial checks passing. Within 2pp of the best honest silver recall, prefer the method with higher sample precision.

> The silver standard and case-control numbers are **diagnostic metrics, not a human gold standard**. The silver standard covers only the 18.5% of SKUs carrying a usable barcode, which biases it toward well-maintained listings. Turning these into a real correctness figure requires hand-labelling roughly 100 groups.

## 📦 Outputs

| Path | Contents |
|---|---|
| `output/soap_sku_data_<method>.xlsx` | The original ten columns untouched, plus `group_id` and `group_name` |
| `output/metrics.csv` | Unified metrics for every method, ratios rounded to 4dp |
| `output/debug/metrics.json` | The same metrics at full precision |
| `output/debug/<method>/groups.csv` | Stable `group_id` / `group_name` per SKU |
| `output/debug/<method>/report.json` | Threshold curve, honest diagnostics, data hashes, flag state |
| `output/debug/<method>/model.json` | The complete GBDT, reloadable |
| `output/debug/<method>/candidate_pairs.csv` | Candidate source and model score, for debugging |
| `output/debug/<method>/group_lineage.csv` | How this run's group IDs inherit from the previous result |
| `output/experiments/` | Artefacts from non-default flags — never overwrites the deliverables |

All five methods write `report.json` under one shared set of key names, and `evaluate_all` reads only those structured fields.

## 🔁 Reproducing the published results

The deck in `slides/` is generated from committed artefacts. Everything it reports comes from the pipeline itself — no document in `docs/` is required:

| Deck content | Produced by |
|---|---|
| Five-method comparison table | `output/metrics.csv`, via `run_method_*` then `evaluate_all` |
| LLM before/after table | The same entry points with `--llm-extract`, landing in `output/experiments/*__llm/` |
| Threshold curve | The `threshold_curve` key of each `output/debug/<method>/report.json` |
| Adversarial 7/7 | `python3 -m prediction.regression_checks` |

The 95% Wilson intervals and the p-values on the LLM slide were computed separately and are not regenerated by any script in this repository.

## 📁 Project structure

```text
hktv-sku-group/
├── paths.py                 the single path configuration for the whole project
├── core/                    domain modules shared by preprocessing and prediction
│   ├── dataset.py           reads the SKU workbook
│   ├── silver.py            barcode-derived silver standard + pack-invariance test set
│   └── norm/extract/variant/group.py
├── data/
│   ├── raw/                 the single raw dataset
│   ├── processed/           model input after unified preprocessing
│   └── labels/              silver-standard labels + frozen pack-invariance test set
├── preprocessing/           the single data-preprocessing entry point
├── prediction/
│   ├── run_method_N_*.py    standalone entry point per method
│   ├── evaluate_all.py      unified metric comparison
│   ├── runner.py            output contract shared by all five entry points
│   ├── regression_checks.py seven adversarial regression checks
│   ├── engine/multipass.py  the recall → score → cluster pipeline shared by all five methods
│   ├── engine/training.py   context building, label factory, leak-free fold splits
│   └── lib/                 features, candidates, embeddings, images, GBDT, ANN
├── output/                  predictions, metrics, and per-method debug artefacts
├── qa/                      long-lived offline regression tests
├── docs/                    design record, data-exploration findings, research probes
├── slides/                  the method-and-results deck, as a self-contained HTML presentation
└── .github/workflows/       runs QA automatically on every push
```

Dependencies point one way: `preprocessing → core ← prediction`. Silver-standard construction and workbook reading both live in `core/`, and preprocessing depends on no prediction method, so "where the labels came from" never gets tangled with "how a particular model computes". `qa/test_project_layout.py` enforces this direction.

## 🔒 Configuration and data safety

- `.env` stays on the local machine and is excluded by `.gitignore`; `.env.example` holds variable names only.
- GitHub Actions runs only the offline QA on synthetic data, and never calls the OpenAI API.
- Raw product text and the embedding cache should not be shared externally without data-governance sign-off.

## 📚 Further reading

| Document | Contents |
|---|---|
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | The recommended implementation in depth, with per-switch ablations |
| [docs/TECHNICAL_PLAN.md](docs/TECHNICAL_PLAN.md) | Problem definition, `group_key` design, metric definitions, risk register |
| [docs/data_exploration.md](docs/data_exploration.md) | Field fill rates, barcode availability, and dirty-data patterns across the 1,472 raw rows |
| [docs/research/](docs/research/) | One-off probe scripts and their recorded output — the measured evidence behind each design decision |

## Author

👤 **Yiyu Yang**

- Github: [@esemsc-yy1824](https://github.com/esemsc-yy1824)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check the [issues page](https://github.com/esemsc-yy1824/hktv-sku-group/issues).

## Show your support

Give a ⭐️ if this project helped you!

***
_README format based on [readme-md-generator](https://github.com/kefranabg/readme-md-generator)._
