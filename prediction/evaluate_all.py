# -*- coding: utf-8 -*-
"""One comparison table across all five prediction methods.

The contract is the `metrics` key of the report.json each method writes: all five are read
from the same key, never by parsing the human-readable report -- one reworded sentence there
would silently scrape the wrong number, or crash outright.

Outputs:
  output/metrics.csv        side-by-side comparison; rates rounded to 4 dp, plus timing and tokens
  output/debug/metrics.json full precision + selection rule + best method
"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import (METHOD_DIRS, METHODS, METRICS_CSV, METRICS_JSON,               # noqa: E402
                   PREDICTION_WORKBOOKS, ensure_dirs, groups_csv)
from prediction.regression_checks import check, load                            # noqa: E402

# How the five methods describe their representation in the metrics.csv column
REPRESENTATION_LABELS = {
    'method_1_lexical': 'lexical features',
    'method_2_local_tfidf': 'local hashed char/word TF-IDF',
    'method_3_semantic': 'OpenAI text-embedding-3-small (256d)',
    'method_4_semantic_image': 'OpenAI 256d + product-photo dhash/hue/structure',
    'method_5_hybrid_fusion': ('OpenAI 256d + local hashed TF-IDF + '
                               'product-photo late fusion'),
}

RATES = ('honest_silver_recall', 'silver_negative_merge_rate',
         'case_control_precision', 'pack_invariance', 'candidate_recall')

FIELDS = ['method', 'output_file', 'representation', 'threshold', 'honest_silver_recall',
          'silver_negative_merge_rate', 'case_control_precision', 'pack_invariance',
          'candidate_recall', 'adversarial_pass', 'adversarial_total', 'release_gate',
          'total_seconds', 'engine_seconds', 'api_tokens', 'api_requests', 'api_cache_hit']

# Release gate
MAX_NEGATIVE_MERGE_RATE = 0.025
MIN_PACK_INVARIANCE = 0.98
RECALL_TOLERANCE = 0.02
SELECTION_RULE = ('negative merge <= 2.5%, pack invariance >= 98%, all adversarial '
                  'checks pass; among methods within 2pp of the best honest silver '
                  'recall, maximize case-control precision')


def read_report(method):
    """Read one method's report.json."""
    report_path = METHOD_DIRS[method] / 'report.json'
    if not report_path.exists():
        raise FileNotFoundError(
            '%s is missing; run the matching prediction/run_0N_*.py first' % report_path)
    with report_path.open(encoding='utf-8') as handle:
        report = json.load(handle)
    if 'metrics' not in report:
        raise ValueError('%s has no metrics key; the output contract does not line up' % report_path)
    return report


def regression_score(method):
    rows, gid = load(str(groups_csv(method)))
    results = check(rows, gid)
    return sum(bool(ok) for _, ok, _ in results), len(results)


def build_row(method):
    report = read_report(method)
    metrics = report['metrics']
    threshold = report.get('selected_threshold')
    timing = report.get('timing') or {}
    # On a cache hit, usage still holds the tokens the cold start actually spent -- that is
    # the real cost of the method.
    usage = ((report.get('embedding') or {}).get('usage')) or {}
    passed, total = regression_score(method)
    row = {
        'method': method,
        'output_file': PREDICTION_WORKBOOKS[method].name,
        'representation': REPRESENTATION_LABELS[method],
        'threshold': '' if threshold is None else threshold,
        'honest_silver_recall': metrics['silver_recall'],
        'silver_negative_merge_rate': metrics['silver_negative_merge_rate'],
        'case_control_precision': metrics['case_control_precision'],
        'pack_invariance': metrics['pack_invariance'],
        'candidate_recall': metrics['candidate_recall'],
        'adversarial_pass': passed,
        'adversarial_total': total,
        'total_seconds': timing.get('total_seconds', ''),
        'engine_seconds': timing.get('engine_seconds', ''),
        'api_tokens': int(usage.get('total_tokens', 0)),
        'api_requests': int(usage.get('requests', 0)),
        # True means this run read the local cache and spent no tokens again
        'api_cache_hit': bool((report.get('embedding') or {}).get('cache_hit', False)),
    }
    row['release_gate'] = bool(
        row['silver_negative_merge_rate'] <= MAX_NEGATIVE_MERGE_RATE
        and row['pack_invariance'] >= MIN_PACK_INVARIANCE
        and passed == total
    )
    return row


def main():
    ensure_dirs()
    rows = [build_row(method) for method in METHODS]

    eligible = [r for r in rows if r['release_gate']]
    if not eligible:
        raise SystemExit('no method passed the release gate; cannot pick a best method')
    best_recall = max(r['honest_silver_recall'] for r in eligible)
    near_best = [r for r in eligible
                 if r['honest_silver_recall'] >= best_recall - RECALL_TOLERANCE]
    best = max(near_best, key=lambda r: (r['case_control_precision'],
                                         -r['silver_negative_merge_rate'],
                                         r['honest_silver_recall']))

    # The CSV is for humans: round every rate to 4 dp, so no 0.011000000000000001 noise.
    with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore',
                                lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (round(v, 4) if k in RATES and isinstance(v, float) else v)
                for k, v in row.items()
            })

    # The JSON keeps full precision, for reproduction and downstream use.
    payload = {
        'selection_rule': SELECTION_RULE,
        'best_method': best['method'],
        # paths.review_method() reads this key so the adversarial regression follows the
        # evaluation verdict.
        'best_method_key': best['method'],
        'caveat': ('silver/case-control metrics are diagnostics, not human-labelled '
                   'population correctness or completeness'),
        'methods': rows,
    }
    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print('  %-24s %8s %8s %9s %9s %7s %6s %8s %10s' % (
        'method', 'recall', 'negmerge', 'sample-P', 'pack-inv', 'adv', 'gate',
        'seconds', 'tokens'))
    for r in rows:
        print('  %-24s %7.1f%% %7.1f%% %8.1f%% %8.1f%% %3d/%-3d %6s %7.1fs %10s' % (
            r['method'], r['honest_silver_recall'] * 100,
            r['silver_negative_merge_rate'] * 100,
            r['case_control_precision'] * 100, r['pack_invariance'] * 100,
            r['adversarial_pass'], r['adversarial_total'],
            'PASS' if r['release_gate'] else 'FAIL',
            r['total_seconds'] or 0.0,
            '{:,}'.format(r['api_tokens']) if r['api_tokens'] else '0'))
    print('\nBest:', best['method'])
    print('Wrote:', METRICS_CSV)
    print('Wrote:', METRICS_JSON)
    return payload


if __name__ == '__main__':
    main()
