# -*- coding: utf-8 -*-
"""Download, cache and compact descriptors for product hero images.

⚠️ Measured conclusion (probe in docs/research/image_signal.py, results in
docs/research/image_signal.txt): "the more alike the images, the more likely the same
product" is **false** on this data. A silver-standard negative pair is defined as "same
brand, same unit size, same pack count, disjoint barcodes" -- which in practice means
different scents from one product line, sharing a single packaging design. Over the full
193 positive / 645 negative pairs, 32x32 greyscale structural similarity reaches only
AUC 0.419, **worse than random**: the easily confused pairs are structurally more alike
than the genuinely same-product pairs.

What does work is **photo near-duplication**: merchants reselling the same item routinely
reuse the manufacturer's photo, so the re-uploaded file gets a different CDN name but
near-identical pixels. On an exact dhash match, positives hit 13.0% with 0/645 false
positives; relaxing to Hamming <=2 gives 17.1% of positives at 2/645.

So this module emits three descriptors and lets the GBDT decide how to use them instead
of prescribing a direction:

  dhash      64-bit difference hash -- near-duplicate detection, high-precision evidence
  hue        32-bin hue histogram (saturation x value weighted) -- scent variants differ
             mainly in hue
  structure  32x32 greyscale, mean removed -- layout/outline, marginal AUC < 0.5, the
             model learns the direction

The images themselves are never written to disk; only the three descriptors are cached
(about 4KB per URL).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import time
import urllib.error
import urllib.request

import numpy as np

DHASH_BITS = 64
HUE_BINS = 32
STRUCTURE_SIDE = 32
USER_AGENT = 'Mozilla/5.0'
TIMEOUT = 25

# Hamming radius for the near-duplicate recall channel. Measured on the full silver
# standard (193 positive / 645 negative):
#   <=0  pos hit 13.0%  neg false 0.0%   (0/645)
#   <=2  pos hit 17.1%  neg false 0.3%   (2/645)  <- chosen: +4.1pp recall, costs 2 pairs
#   <=4  pos hit 17.6%  neg false 2.0%   (13/645) precision ratio falls from 55x to 8.7x
NEAR_DUP_MAX_HAMMING = 2


def _l2(vector):
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm > 1e-6 else vector).astype(np.float32)


def descriptor(raw):
    """Decode image bytes into the three descriptors: dhash / hue / structure."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    image = image.convert('RGB')

    grey = np.asarray(
        image.convert('L').resize((STRUCTURE_SIDE, STRUCTURE_SIDE), Image.LANCZOS),
        dtype=np.float32).ravel()
    structure = _l2(grey - grey.mean())

    hsv = np.asarray(image.convert('HSV').resize((64, 64), Image.LANCZOS), dtype=np.float32)
    weight = (hsv[..., 1].ravel() / 255.0) * (hsv[..., 2].ravel() / 255.0)
    hist, _ = np.histogram(hsv[..., 0].ravel(), bins=HUE_BINS, range=(0, 256), weights=weight)
    hue = _l2(hist)

    small = np.asarray(image.convert('L').resize((9, 8), Image.LANCZOS), dtype=np.float32)
    dhash = (small[:, 1:] > small[:, :-1]).ravel()

    return {'dhash': dhash, 'hue': hue, 'structure': structure}


def _cache_path(cache_dir, url):
    return cache_dir / (hashlib.sha256(url.encode('utf-8')).hexdigest()[:20] + '.npz')


def _load_cached(cache_dir, url):
    path = _cache_path(cache_dir, url)
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return {'dhash': data['dhash'], 'hue': data['hue'], 'structure': data['structure']}
    except Exception:
        return None


def _download(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def build(rows, cache_dir, allow_download=True, max_workers=12):
    """Return ``(sku_id -> descriptor, meta)``. Already-cached URLs are not re-downloaded.

    With allow_download=False it reads the cache only and never touches the network --
    the same contract as semantic's --offline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_by_sku = {r['sku_id']: (r.get('main_image_url') or '').strip() for r in rows}
    urls = sorted({u for u in url_by_sku.values() if u})

    started = time.perf_counter()
    by_url = {}
    missing = []
    for url in urls:
        cached = _load_cached(cache_dir, url)
        if cached is None:
            missing.append(url)
        else:
            by_url[url] = cached
    n_cached = len(by_url)

    failures = {}
    if missing and allow_download:
        def work(url):
            return url, descriptor(_download(url))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for future in concurrent.futures.as_completed(
                    [pool.submit(work, url) for url in missing]):
                try:
                    url, desc = future.result()
                except Exception as exc:            # one bad image must not sink the run
                    failures[str(exc)] = failures.get(str(exc), 0) + 1
                    continue
                np.savez_compressed(_cache_path(cache_dir, url), **desc)
                by_url[url] = desc

    by_sku = {sid: by_url[url] for sid, url in url_by_sku.items() if url in by_url}
    meta = {
        'n_rows': len(rows),
        'n_urls': len(urls),
        'n_with_descriptor': len(by_sku),
        'n_cache_hit': n_cached,
        'n_downloaded': len(by_url) - n_cached,
        'n_failed': len(urls) - len(by_url),
        'failures': failures,
        'download_allowed': bool(allow_download),
        'seconds': round(time.perf_counter() - started, 3),
        'dhash_bits': DHASH_BITS,
        'hue_bins': HUE_BINS,
        'structure_side': STRUCTURE_SIDE,
    }
    return by_sku, meta


def _bits_to_key(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes()


def near_duplicate_pairs(by_sku, max_hamming=NEAR_DUP_MAX_HAMMING):
    """SKU pairs that are near-duplicates by dhash.

    With max_hamming=0 the dhashes are bucketed exactly; above 0 the 64 bits are split
    into 4 bands for a coarse banding pass and every surviving pair is then verified by
    Hamming distance -- which avoids the O(n^2) scan.
    """
    items = [(sid, desc['dhash'].astype(bool)) for sid, desc in sorted(by_sku.items())]
    pairs = set()
    if max_hamming <= 0:
        buckets = {}
        for sid, bits in items:
            buckets.setdefault(_bits_to_key(bits), []).append(sid)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
        return pairs

    band = DHASH_BITS // 4
    buckets = {}
    for index, (sid, bits) in enumerate(items):
        for b in range(4):
            key = (b, _bits_to_key(bits[b * band:(b + 1) * band]))
            buckets.setdefault(key, []).append(index)
    seen = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = (members[i], members[j])
                if key in seen:
                    continue
                seen.add(key)
                a, b = items[members[i]], items[members[j]]
                if int(np.count_nonzero(a[1] != b[1])) <= max_hamming:
                    pairs.add((a[0], b[0]) if a[0] < b[0] else (b[0], a[0]))
    return pairs


# Sentinel fed to the model when an image is missing. -1 rather than 0, so that a tree
# can split "no image" out on its own.
MISSING = -1.0
FEATURE_NAMES = ('image_dhash_similarity', 'image_hue_cosine', 'image_structure_cosine')


def pair_features(sku_a, sku_b, by_sku):
    """The three image features. If either side has no image, all of them are the sentinel."""
    a, b = by_sku.get(sku_a), by_sku.get(sku_b)
    if a is None or b is None:
        return [MISSING] * len(FEATURE_NAMES)
    dhash_similarity = 1.0 - float(np.count_nonzero(
        a['dhash'].astype(bool) != b['dhash'].astype(bool))) / DHASH_BITS
    return [dhash_similarity,
            float(np.dot(a['hue'], b['hue'])),
            float(np.dot(a['structure'], b['structure']))]
