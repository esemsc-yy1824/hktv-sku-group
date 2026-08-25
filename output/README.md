# Output directory

Day-to-day inspection only needs:

- `soap_sku_data_method_1_lexical.xlsx`
- `soap_sku_data_method_2_local_tfidf.xlsx`
- `soap_sku_data_method_3_semantic.xlsx`
- `soap_sku_data_method_4_semantic_image.xlsx`
- `soap_sku_data_method_5_hybrid_fusion.xlsx`
- `metrics.csv`

Every prediction workbook is the original ten columns plus `group_id` and `group_name`. `metrics.csv` compares
the methods on the honest diagnostic track, with silver-standard leakage turned off, and rounds ratios uniformly to 4 decimal places. Besides the quality metrics there are cost columns:

- `total_seconds` / `engine_seconds`: full processing time (including the xlsx export) and engine time
- `api_tokens` / `api_requests`: tokens and request count for the OpenAI embeddings. On a local cache hit these
  still report the **real cold-start cost** (that is what the method costs); `api_cache_hit=True` means this run spent nothing further

`debug/` holds regenerable artefacts, used only for troubleshooting and reproduction:

- `<method>/report.json`: structured metrics under one shared set of key names across the five methods; the only data source for `evaluate_all`.
- `<method>/{groups.csv,model.json,candidate_pairs.csv,group_lineage.csv}`: details, model, and candidate pairs.
- `metrics.json`: the full-precision comparison, the selection rule, and the best method.
