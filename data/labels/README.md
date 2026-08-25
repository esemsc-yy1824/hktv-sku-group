# About the labels

For now this holds only the silver standard, built automatically by rules over product barcodes and the like — not a human-confirmed ground truth:

- `silver_positive_pairs.csv`: SKU pairs sharing a valid EAN-13.
- `silver_negative_pairs.csv`: SKU pairs with the same brand, same unit size, and same pack count, but disjoint barcodes.
- `preprocessing_manifest.json`: input/output hashes and record volumes for this preprocessing run.

If a gold standard confirmed by manual work ever exists, save it separately as `gold_*.csv`; do not overwrite the silver standard.

`pack_invariance_pairs.csv` is the **frozen** pack-invariance test set (same brand, same unit size, same product
form, same variant signature, compatible origin, differing only in pack count). Evaluation reads it and never
recomputes it: these pairs are drawn by bucketing on variant signatures, and recomputing would let the
denominator follow the variant vocabulary, so shrinking the vocabulary would violate the rule and improve the
metric at the same time.
See `core/silver.py:load_packcount_pairs` for details.
