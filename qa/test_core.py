# -*- coding: utf-8 -*-
"""Focused regression tests for Method 5 infrastructure."""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prediction.lib import candidates as C                              # noqa: E402
from prediction.lib.group_ids import assign                    # noqa: E402
from prediction.lib.model import GBDT                          # noqa: E402
from prediction.engine.multipass import constrained_clusters, hard_veto  # noqa: E402


def attr(sku_id, title=None, brand='brand', form='SOAP_BAR', unit=100.0,
         origin=frozenset({'HK'}), pack=1, ean=frozenset()):
    return {
        'sku_id': sku_id, 'form': form, 'unit_value': unit, 'unit_uom': 'G',
        'origin': origin, 'pack_count': pack, 'ean': ean,
        'gs1': None,
        'brand_key': brand, 'brand_id': brand, 'brand_canonical': brand,
        'raw': {
            'sku_name_chi': title or ('soap ' + sku_id),
            'summary_chi': '', 'brand_name_chi': brand,
        },
    }


class Method5Tests(unittest.TestCase):
    def test_method_5_adds_local_cosine_after_semantic_cosine(self):
        """Late fusion must keep two independent scores, not average away interpretability."""
        from prediction.engine.multipass import feature_vector

        a, b = attr('a'), attr('b')
        ctx = {
            'sigs_feat': {'a': frozenset(), 'b': frozenset()},
            'resid': {'a': frozenset({'alpha'}), 'b': frozenset({'beta'})},
            'idf': {}, 'n': 2, 'veto': set(), 'kvt': set(),
        }
        semantic = {'a': np.asarray([1.0, 0.0]), 'b': np.asarray([0.0, 1.0])}
        local = {'a': np.asarray([1.0, 0.0]), 'b': np.asarray([1.0, 0.0])}
        values = feature_vector(a, b, ctx, semantic,
                                auxiliary_embedding_by_sku=local)
        self.assertEqual(values[-2:], [0.0, 1.0])

    def test_hard_veto_only_rejects_known_conflicts(self):
        a = attr('a', form='UNKNOWN', unit=None, origin=frozenset())
        b = attr('b', form='SOAP_BAR', unit=None, origin=frozenset({'US'}))
        self.assertFalse(hard_veto(a, b))
        self.assertTrue(hard_veto(attr('c', unit=100), attr('d', unit=200)))

    def test_candidate_union_recovers_embedding_only_pair(self):
        attrs = {
            'a': attr('a', title='alpha', brand='one'),
            'b': attr('b', title='beta', brand='two'),
            'c': attr('c', title='gamma', brand='three'),
        }
        vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        info = C.generate(attrs, vectors, embedding_top_k=1, embedding_min_cos=0.8,
                          include_ean=False, include_legacy_hint=False)
        self.assertIn(('a', 'b'), info)
        self.assertIn('semantic_topk', info[('a', 'b')]['sources'])

    def test_must_link_is_applied_before_sparse_average_linkage(self):
        attrs = {x: attr(x) for x in ('a', 'b', 'c')}
        scores = {('a', 'b'): 1.0, ('b', 'c'): 0.99}
        groups = constrained_clusters(attrs, scores, 0.98, must_links={('a', 'b')})
        member_sets = {frozenset(m['sku_id'] for m in members) for members in groups.values()}
        self.assertIn(frozenset({'a', 'b'}), member_sets)

    def test_group_id_survives_added_listing(self):
        first_groups = {'x': [attr('a'), attr('b')]}
        first, _ = assign(first_groups)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'groups.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['sku_id', 'group_id'])
                w.writeheader()
                for sid in ('a', 'b'):
                    w.writerow({'sku_id': sid, 'group_id': first['x']})
            second, _ = assign({'y': [attr('a'), attr('b'), attr('c')]}, path)
        self.assertEqual(first['x'], second['y'])

    def test_gbdt_round_trip_predictions(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(80, 3))
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float)
        model = GBDT(n_trees=12, seed=4).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.json')
            model.save(path, {'purpose': 'roundtrip-test'})
            restored = GBDT.load(path)
        np.testing.assert_allclose(model.predict_proba(X), restored.predict_proba(X),
                                   rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
