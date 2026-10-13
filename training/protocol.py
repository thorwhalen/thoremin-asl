"""The evaluation protocol as data: which clip or image is train, val or test, for which
model, and why. Written once, before any training, from the extracted landmark records.

Two tasks: ``letters`` (the 24 static letters; the static model) and ``motion`` (J, Z, the
seven words and ``none``; the motion model). Roles:

- ``train``: eligible for training. Which train sources a candidate actually uses is chosen
  by validation only (leave-one-source-out, PopSign val), never by a test score.
- ``val``: for model and threshold selection (PopSign's val signers).
- ``test``: held out from every model. A source in test is in no training set of that task.
- ``excluded``: not used, with the reason (e.g. a byte-identical copy of a test clip).

Rules (see protocol.md for the reasoning):

letters  train: aslense (letter clips), aslnow, asl_hg, asl_alphabet_gestures, om192006,
                studiogalt, american_hand_signs
         test : marxulia (all; ``sample60`` = the shipped pipeline's sample), signalphaset_static,
                signalphaset_dynamic letter clips
motion   train: popsign corpus-train, popsign_negatives (other PopSign signs, corpus-train
                signers, as ``none``), aslense (J, Z, words; letter clips as ``none``),
                studiogalt (J, Z, bye, hello, please; letters as ``none``)
         val  : popsign corpus-val
         test : popsign corpus-test, asl_citizen (all splits; ``corpus_split=test`` is the
                headline), wlasl, signalphaset_dynamic (letter clips as ``none``)
         excluded: aslense clips that are byte-identical to (or share a file name with) an
                asl_citizen or wlasl clip; wlasl 'goodbye' (another gloss)

Outputs, in ``--out``: ``manifest.csv`` (one row per record and task) and
``duplicates.csv`` (the Aslense copies found and what they copy).

usage: asl.py protocol --raw RAW --lm LANDMARKS --out EVAL_DIR
"""
import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict

from asl_common import STATIC_LETTERS, iter_records
from motion_common import MOTION_SIGNS

LETTER_TRAIN = ('aslense', 'aslnow', 'asl_hg', 'asl_alphabet_gestures', 'om192006', 'studiogalt', 'american_hand_signs')
LETTER_TEST = ('marxulia', 'signalphaset_static', 'signalphaset_dynamic')
MOTION_SOURCES = ('popsign', 'aslense', 'studiogalt', 'asl_citizen', 'wlasl', 'signalphaset_dynamic')


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_copies(raw):
    """Aslense clips that are byte-identical to, or named like, a test-only clip:
    ``{aslense_id: (test_source, test_id, how)}``."""
    import sources as S
    test = {}
    for src in ('asl_citizen', 'wlasl'):
        for it in S.items(src, raw):
            p = it['loc'][1]
            test[_md5(p)] = (src, it['id'], 'md5')
            test.setdefault(('name', os.path.basename(p)), (src, it['id'], 'file name'))
    out = {}
    for it in S.items('aslense', raw):
        p = it['loc'][1]
        hit = test.get(_md5(p)) or test.get(('name', os.path.basename(p)))
        if hit:
            out[it['id']] = hit
    return out


def roles(rec, copies):
    """``[(task, role, note)]`` for one landmark record."""
    src, lab, split = rec['source'], rec['label'], rec.get('split')
    out = []
    if lab in STATIC_LETTERS:
        if src in LETTER_TRAIN:
            out.append(('letters', 'train', ''))
        elif src in LETTER_TEST:
            note = 'sample60' if rec.get('sample60') else ''
            out.append(('letters', 'test', note))
    if src == 'popsign_negatives':  # other PopSign signs, train signers only: negatives
        return [('motion', 'train' if split == 'train' else 'excluded', 'none')]
    is_sign = lab in MOTION_SIGNS
    is_none = lab in STATIC_LETTERS
    if src in MOTION_SOURCES and (is_sign or is_none) and rec['kind'] == 'video':
        cls = lab if is_sign else 'none'
        if src == 'popsign':
            out.append(('motion', {'train': 'train', 'val': 'val', 'test': 'test'}[split], cls))
        elif src == 'aslense':
            if rec['id'] in copies:
                s, i, how = copies[rec['id']]
                out.append(('motion', 'excluded', f'copy of {s} {i} ({how})'))
            else:
                out.append(('motion', 'train', cls))
        elif src == 'studiogalt':
            out.append(('motion', 'train', cls))
        else:  # asl_citizen, wlasl, signalphaset_dynamic
            out.append(('motion', 'test', cls))
    elif src == 'wlasl':
        out.append(('motion', 'excluded', f'gloss {lab!r} is not one of ours'))
    if src == 'aslense' and rec['id'] in copies and lab in STATIC_LETTERS:
        out = [(t, 'excluded', 'copy of a test-only clip') if t == 'letters' else (t, r, n) for t, r, n in out]
    return out


SOURCES_ALL = ('aslense', 'aslnow', 'popsign', 'popsign_negatives', 'signalphaset_dynamic', 'signalphaset_static', 'asl_hg',
               'american_hand_signs', 'asl_alphabet_gestures', 'om192006', 'studiogalt', 'marxulia', 'asl_citizen', 'wlasl')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raw', required=True)
    ap.add_argument('--lm', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    copies = find_copies(a.raw)
    with open(os.path.join(a.out, 'duplicates.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['aslense_id', 'copy_of_source', 'copy_of_id', 'matched_by'])
        for k, (s, i, how) in sorted(copies.items()):
            w.writerow([k, s, i, how])
    counts = defaultdict(Counter)
    signers = defaultdict(set)
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'id', 'label', 'signer', 'corpus_split', 'task', 'role', 'note', 'hand_frames'])
        for src in SOURCES_ALL:
            for r in iter_records(a.lm, src):
                nh = sum(1 for _, h in r['frames'] if h)
                for task, role, note in roles(r, copies):
                    w.writerow([src, r['id'], r['label'], r.get('signer') or '', r.get('split') or '', task, role, note, nh])
                    key = (task, role, src)
                    counts[key][r['label']] += 1
                    if r.get('signer'):
                        signers[key].add(r['signer'])
    print(f'{len(copies)} aslense clips are copies of test-only clips')
    rows = list(csv.DictReader(open(os.path.join(a.out, 'manifest.csv'))))
    near = near_duplicates(a.lm, rows)
    with open(os.path.join(a.out, 'near_duplicates.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['test_source', 'test_id', 'train_source', 'train_id', 'mean_abs_diff_palm'])
        for (s, k), (ts, ti, d) in sorted(near.items()):
            w.writerow([s, k, ts, ti, round(d, 4)])
    print(f'{len(near)} letter test stills nearly identical to a train hand:', dict(Counter((s, ts) for (s, _), (ts, _, _) in near.items())))
    for (task, role, src), c in sorted(counts.items()):
        print(f'{task:8s} {role:8s} {src:22s} {sum(c.values()):6d} items, {len(signers[(task, role, src)]) or "?"} signers, '
              f'{len(c)} labels')
    return 0


if __name__ == '__main__':
    sys.exit(main())


def _shape_vec(hand):
    """21 image points relative to the wrist, in palm lengths (x, y only, unmirrored)."""
    import numpy as np
    p = np.asarray(hand[2], np.float64).reshape(21, 3)[:, :2]
    p = p - p[0]
    return (p / (np.linalg.norm(p[9]) or 1.0)).reshape(-1)


def near_duplicates(lm, manifest_rows, *, test_sources=('marxulia', 'signalphaset_static'), threshold=0.02):
    """Letter test stills whose hand shape matches a train hand of the same letter almost
    exactly (mean absolute point difference below ``threshold`` palm lengths): a sign that the
    same photo is in both corpora. Returns ``{(source, id): (train_source, train_id, dist)}``."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from asl_common import pick_hand
    role = {(r['source'], r['id']): r['role'] for r in manifest_rows if r['task'] == 'letters'}
    train, test = defaultdict(list), defaultdict(list)
    for src in {s for s, _ in role}:
        for rec in iter_records(lm, src, set(STATIC_LETTERS)):
            ro = role.get((src, rec['id']))
            if ro not in ('train', 'test') or (ro == 'test' and src not in test_sources):
                continue
            hands = [pick_hand(h) for _, h in rec['frames'] if h]
            if not hands:
                continue
            (train if ro == 'train' else test)[rec['label']].append((src, rec['id'], _shape_vec(hands[len(hands) // 2])))
    out = {}
    for lab, te in test.items():
        tr = train.get(lab)
        if not tr:
            continue
        nn = NearestNeighbors(n_neighbors=1, metric='manhattan').fit(np.array([v for *_, v in tr]))
        d, i = nn.kneighbors(np.array([v for *_, v in te]))
        for (s, k, _), dd, ii in zip(te, d[:, 0] / 42, i[:, 0]):
            if dd < threshold:
                out[(s, k)] = (tr[ii][0], tr[ii][1], float(dd))
    return out
