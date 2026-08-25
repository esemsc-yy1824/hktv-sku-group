# Design and Experiment Record

This folder holds the project's design documents, data-exploration findings, and one-off probe
scripts. **Every path and command here is aligned with the current layout**, so they can be run
as written. For the official run entry points, see the [README.md](../README.md) at the project root.

| File | Contents |
|---|---|
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | The recommended implementation (`method_5_hybrid_fusion`, operating point auto-selected against the release gate), reproduction commands, and the metric comparison across the five methods |
| [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md) | Problem definition, group_key design, metric definitions, and the risk list |
| [data_exploration.md](data_exploration.md) | Field fill rates, barcode availability, and dirty-data patterns across the 1472 raw rows |
| `research/` | 9 probe scripts plus their outputs, each same-named pair adjacent. `image_signal` is the evidence for method 4, `ablation_form_ann` for the two optional switches, and `model_choice` for the scorer selection (tree depth vs. logistic regression) |

The metrics in `IMPLEMENTATION.md` are item-for-item consistent with [output/metrics.csv](../output/metrics.csv).
`TECHNICAL_PLAN.md` is the design-and-definitions document, and every file path that appears in it points at a file that actually exists.

## Re-running the Probes

The scripts in `research/` run as-is, from any directory:

```bash
python3 docs/research/form_rules.py
```

They originally read a `raw.jsonl` export — its contents duplicated the source xlsx row for row,
so it was deleted during the reorganisation; they now uniformly read `data/raw/soap_sku_data.xlsx`
through `core.dataset.load`. Each one was verified individually: the outputs are byte-identical
before and after the data-source change.

One exception: `blocking_recall.py` depends on set iteration order, which is inherently unstable
across processes — the counts (46/193 missed, dominated by 35/8/3) are the same every run, but the
dictionary key order and the sampled examples vary. So its `blocking_recall.txt` is not
byte-reproducible; compare it on the numbers only.

The conclusions these probes originally reached have already been baked into the rules in `core/`
and into [data_exploration.md](data_exploration.md), so under normal circumstances there is no
need to re-run them.
