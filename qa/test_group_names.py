# -*- coding: utf-8 -*-
"""Uniqueness and readability regressions for group_name, plus an anti-cheat check on
the pack-invariance denominator.

Both are "the metric lies to you" defects that eyeballing the deliverable will never
catch:
  1. Different group_ids printing the same group_name -- a reader assumes the grouping
     is wrong. docs/TECHNICAL_PLAN.md §Uniqueness requires this count to be 0.
  2. The pack-invariance denominator is derived from the variant vocabulary, so shrinking
     the vocabulary both breaks the rule and makes the metric look better.
"""
import csv
import unittest
from pathlib import Path

from core.group import compose_name, disambiguate_names
from core.variant import stitch_windows, variant_label
from paths import METHOD_DIRS, METHODS, PACK_PAIRS_PATH


def _delivered(method):
    path = METHOD_DIRS[method] / 'groups.csv'
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


class GroupNameUniquenessTests(unittest.TestCase):
    def test_every_delivered_method_has_unique_names(self):
        """In the deliverable, group_name must correspond one-to-one with group_id."""
        checked = 0
        for method in METHODS:
            rows = _delivered(method)
            if rows is None:
                continue
            checked += 1
            by_name = {}
            for r in rows:
                by_name.setdefault(r['group_name'], set()).add(r['group_id'])
            shared = {n: ids for n, ids in by_name.items() if len(ids) > 1}
            self.assertEqual(shared, {},
                             '%s: %d group_names used by 2+ group_ids' % (method, len(shared)))
        self.assertGreater(checked, 0, 'no groups.csv available to check')

    def test_no_sliding_window_fragments_in_names(self):
        """The variant slot must not print overlapping n-gram fragments of one phrase
        (橄欖油乳·欖油乳木·油乳木果)."""
        for method in METHODS:
            rows = _delivered(method)
            if rows is None:
                continue
            for name in {r['group_name'] for r in rows}:
                parts = [p for p in name.split('·') if len(p) >= 3]
                for a, b in zip(parts, parts[1:]):
                    self.assertNotEqual(a[1:], b[:-1],
                                        '%s: sliding-window fragment in name -> %s' % (method, name))


class StitchTests(unittest.TestCase):
    def test_overlapping_windows_are_stitched(self):
        got = stitch_windows(frozenset({'橄欖油乳', '欖油乳木', '油乳木果'}),
                             '法國有機橄欖油乳木果皂')
        self.assertEqual(got, ['橄欖油乳木果'])

    def test_unattested_merge_is_refused(self):
        """A stitched phrase that never appears in the text must not be merged -- this
        stops coincidental overlap from coining words."""
        got = stitch_windows(frozenset({'檸檬味', '味道好'}), '檸檬味 和 味道好')
        self.assertEqual(sorted(got), ['味道好', '檸檬味'])

    def test_label_prefers_discriminating_term(self):
        sig = frozenset({'橄欖油乳', '欖油乳木', '油乳木果', '蘆薈'})
        self.assertEqual(variant_label(sig, '橄欖油乳木果 蘆薈', prefer={'蘆薈'}, limit=1),
                         '蘆薈')

    def test_label_does_not_echo_other_slots(self):
        """The origin is already in the parentheses; the variant slot must not repeat it."""
        got = variant_label(frozenset({'法國', '蘆薈'}), '法國 蘆薈', exclude=['法國'])
        self.assertEqual(got, '蘆薈')


class DisambiguationTests(unittest.TestCase):
    def _member(self, sku, title, brand='B'):
        return {'sku_id': sku, 'brand_canonical': brand, 'form': 'soap_bar',
                'unit_value': 100.0, 'unit_uom': 'g', 'origin': frozenset({'X'}),
                'raw': {'sku_name_chi': title, 'brand_name_chi': brand}}

    def test_identical_text_falls_back_to_id_suffix(self):
        """When the text truly cannot separate two groups, #hex6 is the fallback -- a
        duplicate name is never acceptable."""
        groups = {'k1': [self._member('s1', 'B 香皂 100g')],
                  'k2': [self._member('s2', 'B 香皂 100g')]}
        sigs = {'s1': frozenset(), 's2': frozenset()}
        ids = {'k1': 'G5_aaaaaa111111', 'k2': 'G5_bbbbbb222222'}
        names = disambiguate_names(groups, sigs, ids)
        self.assertEqual(len(set(names.values())), 2, names)
        self.assertTrue(all('#' in n for n in names.values()), names)

    def test_distinguishing_text_is_surfaced(self):
        groups = {'k1': [self._member('s1', 'B 蘆薈香皂 100g')],
                  'k2': [self._member('s2', 'B 玫瑰香皂 100g')]}
        sigs = {'s1': frozenset(), 's2': frozenset()}
        ids = {'k1': 'G5_aaaaaa111111', 'k2': 'G5_bbbbbb222222'}
        names = disambiguate_names(groups, sigs, ids)
        self.assertIn('蘆薈', names['k1'])
        self.assertIn('玫瑰', names['k2'])
        self.assertFalse(any('#' in n for n in names.values()), names)

    def test_deterministic_across_calls(self):
        groups = {'k%d' % i: [self._member('s%d' % i, 'B %s香皂 100g' % v)]
                  for i, v in enumerate(['蘆薈', '玫瑰', '薰衣草', '蜂蜜'])}
        sigs = {'s%d' % i: frozenset() for i in range(4)}
        ids = {'k%d' % i: 'G5_%06d000000' % i for i in range(4)}
        first = disambiguate_names(groups, sigs, ids)
        self.assertEqual(first, disambiguate_names(groups, sigs, ids))

    def test_compose_name_shape(self):
        self.assertEqual(compose_name('LUX', '玫瑰', '香皂', '80g', ['印尼']),
                         'LUX 玫瑰 香皂 80g (印尼)')
        self.assertEqual(compose_name(None, '', '香皂', '80g', []), '香皂 80g')


class PackInvarianceDenominatorTests(unittest.TestCase):
    """The pack-invariance denominator must come from a frozen file, never be recomputed.

    🚨 Recomputing makes the denominator follow the variant vocabulary: shrink the
    vocabulary -> the denominator shrinks -> newly failing pairs drop out of the test set
    -> the metric still prints 100%. Observed once in practice: 85 -> 84. Any change to
    the denominator should surface as an explicit file change (visible in git diff), not
    as a silent metric improvement.
    """

    def test_frozen_file_exists_and_is_well_formed(self):
        self.assertTrue(Path(PACK_PAIRS_PATH).exists(),
                        'frozen pack-invariance set missing: run preprocessing.preprocess_data')
        with open(PACK_PAIRS_PATH, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        self.assertEqual({'sku_a', 'sku_b', 'label', 'evidence'}, set(rows[0]))
        pairs = {(r['sku_a'], r['sku_b']) for r in rows}
        self.assertEqual(len(pairs), len(rows), 'duplicate pairs in the frozen file')

    def test_reports_denominator_matches_frozen_file(self):
        from core.silver import load_packcount_pairs
        import json
        frozen = len(load_packcount_pairs(PACK_PAIRS_PATH))
        checked = 0
        for method in METHODS:
            path = METHOD_DIRS[method] / 'report.json'
            if not path.exists():
                continue
            checked += 1
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['metrics']['pack_invariance_n'], frozen,
                             '%s: pack-invariance denominator differs from frozen file' % method)
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
