# -*- coding: utf-8 -*-
"""Behaviour tests for image descriptors and near-duplicate detection.

No network access; synthetic images only.
"""
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prediction.lib import images


def png_bytes(pixels):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode='RGB').save(buf, format='PNG')
    return buf.getvalue()


def gradient(width=64, height=64, tint=(0, 0, 0)):
    x = np.linspace(0, 255, width, dtype=np.float32)
    base = np.tile(x, (height, 1))
    rgb = np.stack([np.clip(base + tint[i], 0, 255) for i in range(3)], axis=-1)
    return rgb


class DescriptorTests(unittest.TestCase):
    def test_descriptor_shapes_and_normalization(self):
        desc = images.descriptor(png_bytes(gradient()))
        self.assertEqual(desc['dhash'].shape, (images.DHASH_BITS,))
        self.assertEqual(desc['dhash'].dtype, bool)
        self.assertEqual(desc['hue'].shape, (images.HUE_BINS,))
        self.assertEqual(desc['structure'].shape,
                         (images.STRUCTURE_SIDE * images.STRUCTURE_SIDE,))
        for key in ('hue', 'structure'):
            norm = float(np.linalg.norm(desc[key]))
            self.assertTrue(abs(norm - 1.0) < 1e-5 or norm == 0.0, (key, norm))

    def test_same_image_is_bit_identical(self):
        raw = png_bytes(gradient())
        a, b = images.descriptor(raw), images.descriptor(raw)
        np.testing.assert_array_equal(a['dhash'], b['dhash'])
        np.testing.assert_allclose(a['structure'], b['structure'], atol=0)

    def test_missing_side_returns_sentinel_not_zero(self):
        """A missing side must be a sentinel, not 0 -- the model reads 0 as 'very unlike'."""
        feats = images.pair_features('a', 'b', {})
        self.assertEqual(feats, [images.MISSING] * len(images.FEATURE_NAMES))
        self.assertLess(images.MISSING, 0.0)

    def test_identical_photos_are_near_duplicates(self):
        raw = png_bytes(gradient())
        desc = images.descriptor(raw)
        by_sku = {'a': desc, 'b': images.descriptor(raw)}
        self.assertEqual(images.near_duplicate_pairs(by_sku, max_hamming=0), {('a', 'b')})
        self.assertEqual(images.pair_features('a', 'b', by_sku)[0], 1.0)

    def test_build_uses_cache_and_never_downloads_when_disabled(self):
        rows = [{'sku_id': 's1', 'main_image_url': 'https://example.invalid/x.png'}]
        with tempfile.TemporaryDirectory() as tmp:
            by_sku, meta = images.build(rows, Path(tmp), allow_download=False)
            self.assertEqual(by_sku, {})
            self.assertEqual(meta['n_downloaded'], 0)
            self.assertEqual(meta['n_failed'], 1)
            self.assertFalse(meta['download_allowed'])


if __name__ == '__main__':
    unittest.main()
