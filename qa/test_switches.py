# -*- coding: utf-8 -*-
"""Behaviour tests for the two optional switches: the defaults must match today's
behaviour, and each switch must actually do something.

No network, no full pipeline run -- this only checks that the switches wire the signal
in or cut it out.
"""
import unittest

import numpy as np

from core import group as G
from core.dataset import load
from paths import RAW_DATASET_PATH
from prediction.engine import multipass as M
from prediction.lib import ann as ANN


class FormSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load(RAW_DATASET_PATH)[:400]

    def test_default_keeps_real_product_form(self):
        attrs, _ = G.build_attributes(self.rows)
        forms = {a['form'] for a in attrs.values()}
        self.assertIn('soap_bar', forms)
        self.assertGreater(len(forms), 1, 'the default must extract more than one form')

    def test_no_form_blanks_every_product_form(self):
        attrs, _ = G.build_attributes(self.rows, use_form=False)
        self.assertEqual({a['form'] for a in attrs.values()}, {'UNKNOWN'})
        self.assertEqual({a['form_conf'] for a in attrs.values()}, {0.0})

    def test_no_form_disables_the_form_veto(self):
        """Two products of different forms: vetoed by default, not vetoed once disabled."""
        on, _ = G.build_attributes(self.rows)
        pairs = [(a, b) for a in on.values() for b in on.values()
                 if a['form'] != b['form'] and 'UNKNOWN' not in (a['form'], b['form'])]
        self.assertTrue(pairs, 'the sample should contain pairs with differing forms')
        a, b = pairs[0]
        self.assertTrue(M.hard_veto(a, b), 'default: a form conflict must be vetoed')
        off, _ = G.build_attributes(self.rows, use_form=False)
        self.assertFalse(M.hard_veto(off[a['sku_id']], off[b['sku_id']]),
                         'once disabled, form is no longer a conflict')


class AnnSwitchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(240, 32)).astype(np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.ids = ['s%03d' % i for i in range(len(self.matrix))]

    def test_exact_top_k_shape_and_order(self):
        out = ANN.exact_top_k(self.matrix, self.ids, k=10)
        self.assertEqual(len(out), len(self.ids))
        for sid, hits in out.items():
            self.assertEqual(len(hits), 10)
            self.assertNotIn(sid, [o for o, _ in hits], 'a vector cannot be its own neighbour')
            scores = [s for _, s in hits]
            self.assertEqual(scores, sorted(scores, reverse=True), 'must be sorted by similarity')

    def test_ivf_returns_fewer_comparisons_than_exhaustive(self):
        _, meta = ANN.top_k(self.matrix, self.ids, k=10, nprobe=4)
        self.assertLess(meta['avg_compared'], meta['exhaustive_compared'])
        self.assertGreater(meta['saved_fraction'], 0.0)
        self.assertEqual(meta['kind'], 'ivf_flat_numpy')

    def test_ivf_recovers_most_of_exact_top_k(self):
        approx, _ = ANN.top_k(self.matrix, self.ids, k=10, nprobe=6)
        recall = ANN.recall_against_exact(approx, ANN.exact_top_k(self.matrix, self.ids, k=10))
        self.assertGreater(recall, 0.5, 'approximate search must not lose most neighbours')

    def test_ivf_is_deterministic(self):
        a, _ = ANN.top_k(self.matrix, self.ids, k=5, seed=3)
        b, _ = ANN.top_k(self.matrix, self.ids, k=5, seed=3)
        self.assertEqual(a, b, 'the same seed must reproduce bit for bit')


class ReportRecordsConfigTests(unittest.TestCase):
    def test_every_method_report_records_the_switch_state(self):
        """report.json must state for itself which configuration produced the run."""
        import json
        from paths import METHODS, METHOD_DIRS
        for name in METHODS:
            path = METHOD_DIRS[name] / 'report.json'
            if not path.exists():
                continue
            with path.open(encoding='utf-8') as f:
                cfg = json.load(f).get('config')
            with self.subTest(method=name):
                self.assertIsNotNone(cfg, '%s has no config record' % name)
                self.assertIs(cfg['use_form'], True, 'committed artefacts must be the default config')
                self.assertIs(cfg['use_ann'], False, 'committed artefacts must be the default config')


if __name__ == '__main__':
    unittest.main()
