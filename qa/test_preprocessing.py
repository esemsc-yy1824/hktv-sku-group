# -*- coding: utf-8 -*-
"""Regression checks for the single preprocessing pipeline."""
import unittest

from preprocessing.preprocess_data import build_processed_records, validate_processed_records
from core.extract import ean13_valid, extract_form, extract_unit


class PreprocessingTests(unittest.TestCase):
    def test_ean13_check_digit(self):
        self.assertTrue(ean13_valid('9349254002011'))
        self.assertFalse(ean13_valid('9349254002012'))

    def test_unit_and_pack_count_are_separate(self):
        row = {'sku_name_chi': 'Goat Soap 山羊奶皂 100g x 6',
               'pack_size_chi': '', 'summary_chi': '', 'description_chi': ''}
        self.assertEqual(extract_unit(row)[:3], (100.0, 'g', 6))

    def test_gift_does_not_turn_soap_into_accessory(self):
        row = {'sku_name_chi': '阿勒頗手工皂 200g（送起泡袋）', 'summary_chi': ''}
        self.assertEqual(extract_form(row)[0], 'soap_bar')

    def test_processed_records_preserve_unique_skus(self):
        rows = [
            {'category_code': 'x', 'category_path': 'soap', 'sku_id': 'A0001_S_1',
             'sku_name_chi': 'Test soap 100g', 'brand_name_chi': 'Test',
             'pack_size_chi': '', 'origin_chi': '香港', 'summary_chi': '',
             'description_chi': '', 'main_image_url': ''},
        ]
        records, _, _ = build_processed_records(rows)
        validate_processed_records(rows, records)
        self.assertEqual(records[0]['product_form'], 'soap_bar')


if __name__ == '__main__':
    unittest.main()
