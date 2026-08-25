# Data directory

- `raw/soap_sku_data.xlsx`: the single raw dataset; no program should overwrite it.
- `processed/soap_sku_data_preprocessed.xlsx`: the shared input to all five methods after unified preprocessing.
- `labels/silver_positive_pairs.csv`: silver-standard positive pairs sharing a valid EAN-13.
- `labels/silver_negative_pairs.csv`: silver-standard negative pairs with the same brand, unit size, and pack count but disjoint barcodes.
- `labels/preprocessing_manifest.json`: input/output hashes, row counts, and label volumes.

The silver standard is a proxy label with a barcode bias, not a human ground truth; it is physically isolated from the model input.
