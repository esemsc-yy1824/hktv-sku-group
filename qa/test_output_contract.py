# -*- coding: utf-8 -*-
"""Checks for the common five-method prediction output contract."""
import csv
import tempfile
import unittest
from pathlib import Path

from prediction.runner import validate_group_mapping


class OutputContractTests(unittest.TestCase):
    def write_rows(self, rows):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / 'groups.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['sku_id', 'group_id', 'group_name'])
            w.writeheader(); w.writerows(rows)
        return tmp, path

    def test_complete_unique_mapping_passes(self):
        tmp, path = self.write_rows([
            {'sku_id': 'a', 'group_id': 'g1', 'group_name': 'A'},
            {'sku_id': 'b', 'group_id': 'g2', 'group_name': 'B'},
        ])
        try:
            self.assertEqual(len(validate_group_mapping(path, ['a', 'b'])), 2)
        finally:
            tmp.cleanup()

    def test_duplicate_sku_fails(self):
        tmp, path = self.write_rows([
            {'sku_id': 'a', 'group_id': 'g1', 'group_name': 'A'},
            {'sku_id': 'a', 'group_id': 'g1', 'group_name': 'A'},
        ])
        try:
            with self.assertRaises(ValueError):
                validate_group_mapping(path)
        finally:
            tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
