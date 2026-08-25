# -*- coding: utf-8 -*-
"""Text normalization.

Rationale (research/README.md §14): normalization on its own barely improves recall
(six layers of it buy only 3.1pp); its value is in giving downstream attribute extraction
cleaner input. So we do only what is needed to make extraction more accurate, and do not
chase normalization as an end in itself.
"""
import re
import unicodedata

# Full-width -> half-width (NFKC covers most of it; this fills in what NFKC leaves alone)
_EXTRA = str.maketrans({
    '﹒': '·', '·': '·', '，': ',', '；': ';', '：': ':',
    '（': '(', '）': ')', '【': '[', '】': ']', '｜': '|',
    '～': '~', '－': '-', '＋': '+', '／': '/', '＊': '*',
    '✥': ' ', '✿': ' ', '♥': ' ', '❤': ' ', '★': ' ', '☆': ' ',
})

# Marketing noise (FINDINGS §5)
NOISE = re.compile(
    r'平行進?口(貨品|貨|商品|產品)?|平行进口'
    r'|原裝行貨|香港行貨|原裝正貨|行貨|正貨'
    r'|現貨|熱賣|包郵|官方正品|正品保證|放心選購'
    r'|新舊(包裝|版本)隨機發?(貨|出)?|不同版本包裝隨機發?貨?|包裝隨機發貨'
    r'|新舊包裝交替出貨|新舊包裝隨機|隨機發放|隨機出貨|隨機發貨|包裝隨機|隨機派發'
    r'|順豐|市集世界|日本市集|內地直送'
    r'|到期日?\s*[:：]?\s*[\d/\.\-年月日]+'
    r'|EXP\s*[:：]?\s*[\d/\.\-年月]+'
    r'|最佳使用期\s*[:：]?\s*\S+'
    r'|最少至\s*[\d\-/]+'
    r'|No\s*Box'
    r'|\[?Par(allel)?\s*Import\]?',
    re.I)

# Free-gift phrases (FINDINGS §17.1: leaving them in misclassifies 47 soaps as accessory)
GIFT = re.compile(
    r'[（(\[【]?\s*[送附贈赠]\s*(起泡[網网袋]|皂[網网袋盒]|皂碟)\S*\s*[)）\]】]?'
    r'|免費送\S*'
    r'|隨機贈送\S*'
    r'|附\s*起泡[網网袋]'
    r'|送\s*皂[網网]\s*\d*\s*個?',
    re.I)

# Tolerance / approximate-weight notations
TOLERANCE = re.compile(r'[\(（]?\s*\+?/?[-−]\s*\d+\s*%?\s*g?\s*[,，]?\s*(手工製作|手工製造|handmade)?\s*[\)）]?', re.I)

_WS = re.compile(r'\s+')


def nfkc(s):
    """NFKC + full-width to half-width + lowercase + whitespace collapse. Meaning preserved."""
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', str(s)).translate(_EXTRA)
    return _WS.sub(' ', s).strip()


def strip_noise(s):
    """Strip marketing-noise phrases."""
    return _WS.sub(' ', NOISE.sub(' ', s)).strip()


def strip_gift(s):
    """Strip free-gift phrases. Must be called before product_form classification."""
    return _WS.sub(' ', GIFT.sub(' ', s)).strip()


def clean_title(s):
    """Clean title for attribute extraction: NFKC + gift removal + noise removal.

    Note that digits and units are kept — spec extraction still needs them.
    """
    s = strip_noise(strip_gift(nfkc(s)))
    # Stripping can leave empty brackets behind
    s = re.sub(r'[\[\(【（]\s*[\]\)】）]', ' ', s)
    return _WS.sub(' ', s).strip()


def fold(s):
    """Fold into a comparison key: keep only Chinese characters, letters, digits and percent."""
    return re.sub(r'[^\w一-鿿%]+', '', nfkc(s).lower())
