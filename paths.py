# -*- coding: utf-8 -*-
"""The one path configuration for the whole project.

Preprocessing, the five prediction methods, the unified evaluation and the manual
labelling tools all take their paths from here, so renaming a directory does not
mean combing through every script.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data ---
RAW_DATASET_PATH = PROJECT_ROOT / 'data' / 'raw' / 'soap_sku_data.xlsx'
PROCESSED_DATASET_PATH = (PROJECT_ROOT / 'data' / 'processed'
                          / 'soap_sku_data_preprocessed.xlsx')
LABELS_DIR = PROJECT_ROOT / 'data' / 'labels'
# The pack-invariance test set has to be frozen into a file: its denominator is derived
# from the variant vocabulary, so recomputing it would give the metric a chance to
# improve itself by shrinking the test set (see core/silver.load_packcount_pairs).
PACK_PAIRS_PATH = LABELS_DIR / 'pack_invariance_pairs.csv'

# --- Output ---
OUTPUT_DIR = PROJECT_ROOT / 'output'
DEBUG_DIR = OUTPUT_DIR / 'debug'
METRICS_CSV = OUTPUT_DIR / 'metrics.csv'
METRICS_JSON = DEBUG_DIR / 'metrics.json'

# --- Local caches: paid embeddings / LLM extraction, never committed ---
CACHE_DIR = PROJECT_ROOT / '.cache' / 'embeddings'
IMAGE_CACHE_DIR = PROJECT_ROOT / '.cache' / 'images'
LLM_EXTRACT_CACHE_DIR = PROJECT_ROOT / '.cache' / 'llm_extract'

# All five methods share one multi-channel recall pipeline; they differ only in how a
# product is represented. The number is a method's official identity, while
# representation is the engine's internal representation switch -- do not conflate them.
METHODS = ('method_1_lexical', 'method_2_local_tfidf', 'method_3_semantic',
           'method_4_semantic_image', 'method_5_hybrid_fusion')

REPRESENTATION = {
    'method_1_lexical': 'lexical',
    'method_2_local_tfidf': 'local_tfidf',
    'method_3_semantic': 'semantic',
    'method_4_semantic_image': 'semantic_image',
    'method_5_hybrid_fusion': 'hybrid_fusion',
}

METHOD_LABELS = {
    'method_1_lexical': 'Method 1 multi-channel recall + lexical features',
    'method_2_local_tfidf': 'Method 2 multi-channel recall + local TF-IDF vectors',
    'method_3_semantic': 'Method 3 multi-channel recall + OpenAI semantic vectors',
    'method_4_semantic_image': 'Method 4 semantic vectors + product image signals',
    'method_5_hybrid_fusion': 'Method 5 semantic + local lexical + product image late fusion',
}

# Explicitly approved fixed production operating points (method -> threshold). Currently
# empty: all five methods pick their point automatically off the honest threshold curve
# against the release gate (multipass.choose_threshold). Method 5 was once pinned to a
# more conservative 0.98 (silver negative merges 8/645 -- 8 fewer bad merges than the
# automatically selected 0.95, and also about 15 fewer silver recall pairs), then went
# back to automatic selection as a business decision; the 0.98 vs 0.95 trade-off is
# recorded in docs/IMPLEMENTATION.md and on the threshold slide. The mechanism stays, so
# pinning a method again later is just a matter of registering it here.
FIXED_OPERATING_POINTS = {}

METHOD_DIRS = {name: DEBUG_DIR / name for name in METHODS}

PREDICTION_WORKBOOKS = {
    name: OUTPUT_DIR / ('soap_sku_data_%s.xlsx' % name) for name in METHODS
}

# Which method's results the adversarial regression defaults to before evaluate_all runs.
FALLBACK_REVIEW_METHOD = 'method_5_hybrid_fusion'


EXPERIMENTS_DIR = OUTPUT_DIR / 'experiments'


def config_suffix(use_form=True, use_ann=False, use_llm=False):
    """Directory suffix for non-default configs; the default config returns ''."""
    parts = []
    if not use_form:
        parts.append('noform')
    if use_ann:
        parts.append('ann')
    if use_llm:
        parts.append('llm')
    return '__'.join(parts)


def method_dir(method, use_form=True, use_ann=False, use_llm=False):
    """Debug directory for one method.

    Non-default configurations are written under output/experiments/ and never overwrite
    the production artefacts -- which matters: group_id is content-addressed and inherits
    from the previous run by member overlap, so a single experiment run would rewrite the
    ID lineage in the production directory and change downstream group_ids for no reason.
    """
    suffix = config_suffix(use_form, use_ann, use_llm)
    if not suffix:
        return METHOD_DIRS[method]
    return EXPERIMENTS_DIR / ('%s__%s' % (method, suffix))


def prediction_workbook(method, use_form=True, use_ann=False, use_llm=False):
    """A method's delivery workbook. Experiment configs are saved aside too, never
    overwriting the deliverable."""
    suffix = config_suffix(use_form, use_ann, use_llm)
    if not suffix:
        return PREDICTION_WORKBOOKS[method]
    return (EXPERIMENTS_DIR / ('%s__%s' % (method, suffix))
            / PREDICTION_WORKBOOKS[method].name)


def groups_csv(method):
    """A method's sku_id -> group_id/group_name detail (default configuration only)."""
    return METHOD_DIRS[method] / 'groups.csv'


def review_method():
    """Return the best method chosen by evaluate_all.

    The adversarial regression and the manual labelling queue used to hard-code semantic
    -- once the evaluation picked a different method, these tools were still inspecting
    the old method's results. They now follow the conclusion in metrics.json and only
    fall back to FALLBACK_REVIEW_METHOD when no evaluation has been run.
    """
    import json
    try:
        payload = json.loads(METRICS_JSON.read_text(encoding='utf-8'))
        method = payload['best_method_key']
    except (OSError, ValueError, KeyError):
        return FALLBACK_REVIEW_METHOD
    return method if method in METHOD_DIRS else FALLBACK_REVIEW_METHOD


def review_groups_csv():
    return groups_csv(review_method())


def ensure_dirs():
    for path in (*METHOD_DIRS.values(), OUTPUT_DIR, CACHE_DIR, IMAGE_CACHE_DIR,
                 LLM_EXTRACT_CACHE_DIR, PROCESSED_DATASET_PATH.parent, LABELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
