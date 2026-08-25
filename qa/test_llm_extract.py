# -*- coding: utf-8 -*-
"""Offline behaviour tests for --llm-extract: no network, synthetic LLM output.

Three things are guarded:
  1. grounding -- if the evidence is not a substring of the source text, or the value
     does not appear in the evidence, the field must be discarded;
  2. conservative overrides -- the regex unit is cleared only when three conditions hold
     together (the LLM explicitly reports no unit, the regex misread the pack count as
     pc, and the two numbers are equal); that is the fix for Bug 2, and dropping any one
     of the three is not allowed;
  3. two new hard vetoes -- conflicting formulation percentages, and bundles with
     different component compositions.
"""
import unittest

from prediction.engine.multipass import hard_veto, _mixed_bundle
from prediction.lib import llm_extract as L


def row(sku_id, title, pack='', summary=''):
    return {'sku_id': sku_id, 'sku_name_chi': title,
            'pack_size_chi': pack, 'summary_chi': summary}


def attr(sku_id, unit=(100.0, 'g'), pack=1, form='soap_bar'):
    return {'sku_id': sku_id, 'form': form,
            'unit_value': unit[0], 'unit_uom': unit[1], 'unit_source': 'name',
            'origin': frozenset(), 'pack_count': pack, 'ean': frozenset(),
            'raw': {'sku_name_chi': 'x'}}


class ValidateTests(unittest.TestCase):
    def test_ungrounded_evidence_is_rejected(self):
        v = L.validate({'unit': {'value': 110, 'uom': 'g', 'evidence': '不存在的话'},
                        'pack_count': {'value': None, 'evidence': None},
                        'formulation': [], 'bundle_components': None},
                       '標題: 加信氏 皇室香皂110g 4個裝')
        self.assertIsNone(v['unit'])
        self.assertIn('unit', v['rejected'])

    def test_value_must_appear_in_evidence(self):
        """A value the LLM invents that is absent from the evidence (soap sheets,
        value=1) must be discarded and the regex result kept."""
        v = L.validate({'unit': {'value': 1, 'uom': '片', 'evidence': '香皂片紙'},
                        'pack_count': {'value': 20, 'evidence': '[20片裝]'},
                        'formulation': [], 'bundle_components': None},
                       '標題: [20片裝] 香皂片紙')
        self.assertIsNone(v['unit'])
        self.assertEqual(v['pack_count'], 20)

    def test_chinese_numeral_grounds_pack(self):
        v = L.validate({'unit': {'value': None, 'uom': None, 'evidence': None},
                        'pack_count': {'value': 2, 'evidence': '兩件裝'},
                        'formulation': [], 'bundle_components': None},
                       '標題: 兩件裝 香皂')
        self.assertEqual(v['pack_count'], 2)

    def test_formulation_and_bundle_grounding(self):
        v = L.validate({'unit': {'value': None, 'uom': None, 'evidence': None},
                        'pack_count': {'value': None, 'evidence': None},
                        'formulation': [{'name': '月桂油', 'pct': 12, 'evidence': '12%月桂油'},
                                        {'name': '幻觉油', 'pct': 9, 'evidence': '9%幻觉油'}],
                        'bundle_components': [{'name': '盈潤煥采', 'count': 3},
                                              {'name': '幽蓮魅膚', 'count': 2}]},
                       '標題: 12%月桂油 盈潤煥采*3+幽蓮魅膚*2')
        self.assertEqual(v['formulation'], frozenset({('月桂油', 12.0)}))
        self.assertEqual(v['bundle_key'],
                         frozenset({('盈潤煥采', 3), ('幽蓮魅膚', 2)}))

    def test_bundle_counts_are_ratio_normalized(self):
        """3件裝(1,1,1) and 6件裝(2,2,2) are the same configuration -- pack invariance
        applies to bundles too."""
        v = L.validate({'unit': {'value': None, 'uom': None, 'evidence': None},
                        'pack_count': {'value': None, 'evidence': None},
                        'formulation': [],
                        'bundle_components': [{'name': '原味', 'count': 2},
                                              {'name': '玫瑰', 'count': 2}]},
                       '標題: 原味 玫瑰 6件裝')
        self.assertEqual(v['bundle_key'], frozenset({('原味', 1), ('玫瑰', 1)}))


class OverrideTests(unittest.TestCase):
    def _apply(self, a, record, title):
        attrs = {a['sku_id']: a}
        sigs = {a['sku_id']: frozenset()}
        sigs_feat = {a['sku_id']: frozenset()}
        stats = L.apply_overrides(attrs, sigs, sigs_feat,
                                  {a['sku_id']: record}, [row(a['sku_id'], title)])
        return a, sigs, sigs_feat, stats

    def test_pcs_unit_is_cleared_only_with_all_three_conditions(self):
        """4PCS: regex unit=4pc, LLM explicitly unit=null with pack=4 -> clear the unit
        and set the pack count."""
        a, _, _, stats = self._apply(
            attr('s1', unit=(4.0, 'pc')),
            {'unit': {'value': None, 'uom': None, 'evidence': None},
             'pack_count': {'value': 4, 'evidence': '4PCS'},
             'formulation': [], 'bundle_components': None},
            '加信氏 香皂 4PCS')
        self.assertIsNone(a['unit_value'])
        self.assertEqual(a['pack_count'], 4)
        self.assertEqual(stats.get('unit_cleared_pc'), 1)

    def test_real_gram_unit_is_not_cleared_by_llm_null(self):
        a, _, _, _ = self._apply(
            attr('s2', unit=(100.0, 'g')),
            {'unit': {'value': None, 'uom': None, 'evidence': None},
             'pack_count': {'value': 2, 'evidence': '兩件裝'},
             'formulation': [], 'bundle_components': None},
            '兩件裝 香皂 100g')
        self.assertEqual(a['unit_value'], 100.0)

    def test_formulation_tokens_join_both_signatures(self):
        a, sigs, sigs_feat, _ = self._apply(
            attr('s3'),
            {'unit': {'value': None, 'uom': None, 'evidence': None},
             'pack_count': {'value': None, 'evidence': None},
             'formulation': [{'name': '月桂油', 'pct': 12, 'evidence': '12%月桂油'}],
             'bundle_components': None},
            '法國12%月桂油阿勒頗手工皂')
        self.assertIn('12%月桂油', sigs['s3'])
        self.assertIn('12%月桂油', sigs_feat['s3'])
        self.assertEqual(a['formulation'], frozenset({('月桂油', 12.0)}))


class VetoTests(unittest.TestCase):
    def test_formulation_conflict_is_a_hard_veto(self):
        a, b = attr('a', unit=(200.0, 'g')), attr('b', unit=(200.0, 'g'))
        a['formulation'] = frozenset({('月桂油', 12.0)})
        b['formulation'] = frozenset({('月桂油', 40.0)})
        self.assertTrue(hard_veto(a, b))
        b['formulation'] = frozenset({('月桂油', 12.0), ('橄欖油', 88.0)})
        self.assertFalse(hard_veto(a, b), 'same pct + one extra listed ingredient is no conflict')

    def test_bundle_key_composition_differs_is_a_veto(self):
        a, b = attr('a'), attr('b')
        a['bundle_key'] = frozenset({('盈潤煥采', 3), ('幽蓮魅膚', 2)})
        b['bundle_key'] = frozenset({('舒緩潔淨', 3), ('幽蓮魅膚', 2)})
        self.assertTrue(hard_veto(a, b))
        b['bundle_key'] = frozenset({('盈潤煥采', 3), ('幽蓮魅膚', 2)})
        self.assertFalse(hard_veto(a, b))

    def test_llm_bundle_key_replaces_plus_heuristic(self):
        """`0M+` age marker: the heuristic misreads it as a bundle; once the LLM returns
        None it no longer does."""
        a = attr('a', pack=2)
        a['raw'] = {'sku_name_chi': '*優惠兩件裝* 山羊奶手工皂 100g 0M+'}
        self.assertTrue(_mixed_bundle(a), 'without the LLM the heuristic does misfire (as today)')
        a['bundle_key'] = None
        self.assertFalse(_mixed_bundle(a), 'after the LLM verdict an age marker is not a bundle')


if __name__ == '__main__':
    unittest.main()
