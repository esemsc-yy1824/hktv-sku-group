# -*- coding: utf-8 -*-
"""Stable group identifiers and snapshot hashes.

The cluster fingerprint is deliberately not the public identifier.  Existing
IDs are reconciled by member overlap, so adding a listing does not rename the
real-world product for downstream users.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import os


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _members(group):
    return frozenset(m['sku_id'] for m in group)


def _new_id(members, used):
    digest = hashlib.sha256('\0'.join(sorted(members)).encode()).hexdigest()
    n = 16
    while True:
        gid = 'G5_' + digest[:n]
        if gid not in used:
            return gid
        n += 2


def load_previous(path):
    if not path or not os.path.exists(path):
        return {}
    out = collections.defaultdict(set)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['group_id']].add(row['sku_id'])
    return {gid: frozenset(ids) for gid, ids in out.items()}


def assign(groups, previous_path=None, overlap_threshold=0.50):
    """Return ``(group_key -> stable_id, lineage_rows)``.

    Reuse an old ID when at least half of the smaller cluster survives.  Greedy
    one-to-one matching prevents a split from assigning one public ID twice.
    """
    previous = load_previous(previous_path)
    new_members = {key: _members(ms) for key, ms in groups.items()}
    choices = []
    for key, new in new_members.items():
        for old_id, old in previous.items():
            inter = len(new & old)
            if not inter:
                continue
            containment = inter / min(len(new), len(old))
            jaccard = inter / len(new | old)
            if containment >= overlap_threshold:
                choices.append((containment, jaccard, inter, key, old_id))
    choices.sort(key=lambda x: (-x[0], -x[1], -x[2], str(x[3]), x[4]))

    assigned, claimed = {}, set()
    lineage = []
    for containment, jaccard, inter, key, old_id in choices:
        if key in assigned or old_id in claimed:
            continue
        assigned[key] = old_id
        claimed.add(old_id)
        lineage.append({
            'group_id': old_id, 'previous_group_id': old_id,
            'relation': 'continued', 'overlap': inter,
            'jaccard': round(jaccard, 6),
        })

    used = set(previous) | set(assigned.values())
    for key in sorted(groups, key=lambda k: sorted(new_members[k])):
        if key in assigned:
            continue
        gid = _new_id(new_members[key], used)
        used.add(gid)
        assigned[key] = gid
        parents = []
        for old_id, old in previous.items():
            inter = len(new_members[key] & old)
            if inter:
                parents.append((inter, old_id))
        parents.sort(reverse=True)
        lineage.append({
            'group_id': gid,
            'previous_group_id': ';'.join(x[1] for x in parents),
            'relation': 'new' if not parents else 'split_or_reclustered',
            'overlap': parents[0][0] if parents else 0,
            'jaccard': 0.0,
        })
    return assigned, lineage
