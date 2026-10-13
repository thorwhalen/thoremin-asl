"""Score a static (letters) model and a motion model on the protocol's held-out test sets,
exactly as the browser would read them, and write one JSON of results.

Letters (static model alone, as before): top-1 and top-3 over the test hands, per-letter
recall and the top confusions, per test set:
  ``marxulia60``           the shipped pipeline's Marxulia sample (60 per letter, seed 0)
  ``marxulia``             every Marxulia image
  ``signalphaset_static``  SignAlphaSet stills (never trained on)
  ``signalphaset_clips``   SignAlphaSet letter videos, held frames, majority vote per clip

Motion (the motion model on top of THIS static model's posteriors): each clip is replayed
through the recogniser's buffer and window scoring (``motion_common.clip_scores``). A sign
clip counts as detected when the most confident sign over the clip is the right one AND
reaches the threshold; a ``none`` clip (a held letter) counts as a false sign when any sign
reaches the threshold. Test sets: ``popsign_test``, ``asl_citizen_test``, ``asl_citizen_all``,
``wlasl``, ``signalphaset_dynamic``; ``popsign_val`` with ``--val`` (selection only).
Clips that the SHIPPED models were trained on (Aslense copies of ASL Citizen and WLASL clips,
``duplicates.csv``) are flagged, and every motion set is also reported without them
(``<set>/clean``).

usage: asl.py evaluate --lm LANDMARKS --eval EVAL_DIR --static static.json --motion motion.json --name baseline
       asl.py evaluate --compare EVAL_DIR/results-baseline.json EVAL_DIR/results-candidate.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from asl_common import featurize, held_frames, iter_records, load_model, model_predict
from motion_common import MOTION_SIGNS, clip_scores, frame_records

THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def read_manifest(eval_dir):
    return list(csv.DictReader(open(os.path.join(eval_dir, 'manifest.csv'))))


def load_set(lm, rows):
    """The landmark records for manifest rows (keeps each row's note)."""
    want = defaultdict(dict)
    for r in rows:
        want[r['source']][r['id']] = r
    out = []
    for src, ids in want.items():
        labels = {r['label'] for r in ids.values()}
        for rec in iter_records(lm, src, labels):
            if rec['id'] in ids:
                rec['_note'] = ids[rec['id']]['note']
                out.append(rec)
    return out


def letter_sets(manifest):
    t = [r for r in manifest if r['task'] == 'letters' and r['role'] == 'test']
    return {
        'marxulia60': [r for r in t if r['source'] == 'marxulia' and r['note'] == 'sample60'],
        'marxulia': [r for r in t if r['source'] == 'marxulia'],
        'signalphaset_static': [r for r in t if r['source'] == 'signalphaset_static'],
        'signalphaset_clips': [r for r in t if r['source'] == 'signalphaset_dynamic'],
    }


def motion_sets(manifest, *, val=False):
    t = [r for r in manifest if r['task'] == 'motion']
    sets = {
        'popsign_test': [r for r in t if r['source'] == 'popsign' and r['role'] == 'test'],
        'asl_citizen_test': [r for r in t if r['source'] == 'asl_citizen' and r['role'] == 'test' and r['corpus_split'] == 'test'],
        'asl_citizen_all': [r for r in t if r['source'] == 'asl_citizen' and r['role'] == 'test'],
        'wlasl': [r for r in t if r['source'] == 'wlasl' and r['role'] == 'test'],
        'signalphaset_dynamic': [r for r in t if r['source'] == 'signalphaset_dynamic' and r['role'] == 'test'],
    }
    if val:
        sets['popsign_val'] = [r for r in t if r['source'] == 'popsign' and r['role'] == 'val']
    return sets


def eval_letters(static, recs, *, vote=False):
    """Top-1/top-3 of the static model. A still is one hand; with ``vote`` a clip's held frames
    vote (majority of per-frame top-1) and top-3 ranks their mean posterior."""
    labels = [l.lower() for l in static['labels']]
    y, pred, rank, items = [], [], [], 0
    for r in recs:
        if r['label'] not in labels:
            continue
        items += 1
        hands = held_frames(r) if vote else held_frames(r)[:1]
        if not hands:
            continue
        p = model_predict(static, np.array([featurize(h, r['w'], r['h'], static['features']) for h in hands]))
        pred.append(Counter(p.argmax(1)).most_common(1)[0][0])
        rank.append(np.argsort(-p.mean(0))[:3])
        y.append(labels.index(r['label']))
    if not y:
        return {'items': items, 'with_hand': 0}
    y, pred, rank = np.array(y), np.array(pred), np.array(rank)
    top3 = (rank == y[:, None]).any(1)
    per = {labels[k].upper(): [int((pred[y == k] == k).sum()), int((y == k).sum())] for k in sorted(set(y))}
    conf = Counter((labels[a].upper() + '>' + labels[b].upper()) for a, b in zip(y, pred) if a != b)
    return {'items': items, 'with_hand': int(len(y)), 'top1': float((pred == y).mean()), 'top3': float(top3.mean()),
            'per_letter': per, 'confusions': conf.most_common(12)}


def eval_motion(static, motion, recs, *, contaminated=frozenset()):
    """Per-clip outcomes, then summaries for all clips and for the clips the shipped models never saw."""
    labels = motion['labels']
    none = labels.index('none')
    clips = []
    for r in recs:
        cls = r['_note']
        recs_ = frame_records(static, r)
        S = clip_scores(motion, r, recs_)
        if len(S):
            sign = S.copy()
            sign[:, none] = -1
            k = np.unravel_index(sign.argmax(), sign.shape)
            best, p = labels[k[1]].lower(), float(sign.max())
        else:
            best, p = None, 0.0
        clips.append({'source': r['source'], 'id': r['id'], 'cls': cls, 'best': best, 'p': p,
                      'contaminated': (r['source'], r['id']) in contaminated})

    def summary(cs):
        out = {'clips': len(cs)}
        for th in THRESHOLDS:
            per, fp, conf = {}, [], Counter()
            for c in cs:
                if c['cls'] == 'none':
                    fp.append(c['p'] >= th)
                    continue
                ok = c['best'] == c['cls'] and c['p'] >= th
                per.setdefault(c['cls'], [0, 0])
                per[c['cls']][0] += ok
                per[c['cls']][1] += 1
                if not ok:
                    conf[f"{c['cls']}>{c['best'] if c['p'] >= th else 'none'}"] += 1
            n_sign = sum(v[1] for v in per.values())
            out[str(th)] = {'per_sign': per, 'signs_detected': (sum(v[0] for v in per.values()) / n_sign) if n_sign else None,
                                'none_clips': len(fp), 'none_false_sign': (float(np.mean(fp)) if fp else None),
                                'confusions': conf.most_common(10)}
        return out

    clean = [c for c in clips if not c['contaminated']]
    return {'all': summary(clips), 'clean': summary(clean) if len(clean) != len(clips) else 'same', 'clips': clips}


def contaminated_ids(eval_dir):
    """(source, id) of test clips the shipped models trained on, via their Aslense copies."""
    p = os.path.join(eval_dir, 'duplicates.csv')
    return frozenset((r['copy_of_source'], r['copy_of_id']) for r in csv.DictReader(open(p))) if os.path.exists(p) else frozenset()


def evaluate(*, lm, eval_dir, static_path, motion_path, val=False, letters=True):
    manifest = read_manifest(eval_dir)
    static = load_model(static_path)
    res = {'static': static_path, 'motion': motion_path, 'letters': {}, 'motion_sets': {}}
    if letters:
        for name, rows in letter_sets(manifest).items():
            res['letters'][name] = eval_letters(static, load_set(lm, rows), vote=name == 'signalphaset_clips')
            r = res['letters'][name]
            print(f"letters {name:20s} n={r['with_hand']}/{r['items']} top1 {r.get('top1', 0):.3f} top3 {r.get('top3', 0):.3f}", flush=True)
    if motion_path:
        motion = load_model(motion_path)
        cont = contaminated_ids(eval_dir)
        for name, rows in motion_sets(manifest, val=val).items():
            res['motion_sets'][name] = eval_motion(static, motion, load_set(lm, rows), contaminated=cont)
            s = res['motion_sets'][name]['all']
            th = str(float(motion.get('meta', {}).get('threshold', 0.9)))
            print(f"motion  {name:20s} clips={s['clips']} @{th}: signs {s[th]['signs_detected']} none-false {s[th]['none_false_sign']} "
                  f"{ {k: f'{v[0]}/{v[1]}' for k, v in s[th]['per_sign'].items()} }", flush=True)
    return res


def _fmt(v, d=3):
    return '–' if v is None else f'{v:.{d}f}'


def compare(paths, *, threshold='0.9'):
    """Markdown tables comparing result files (first = baseline)."""
    R = [json.load(open(p)) for p in paths]
    names = [os.path.basename(p).replace('results-', '').replace('.json', '') for p in paths]
    lines = ['| Letters test set | n | ' + ' | '.join(f'{n} top-1 | {n} top-3' for n in names) + ' |',
             '|---|---|' + '---|---|' * len(names)]
    for s in R[0]['letters']:
        lines.append(f"| {s} | {R[0]['letters'][s]['with_hand']} | " + ' | '.join(
            f"{_fmt(r['letters'].get(s, {}).get('top1'))} | {_fmt(r['letters'].get(s, {}).get('top3'))}" for r in R) + ' |')
    lines += ['', f'Motion, per sign at threshold {threshold} (detected / clips):', '']
    for s in R[0]['motion_sets']:
        for part in ('all', 'clean'):
            blocks = [r['motion_sets'][s][part] for r in R]
            if any(b == 'same' for b in blocks):
                continue
            signs = sorted({k for b in blocks for k in b[threshold]['per_sign']}, key=lambda k: MOTION_SIGNS.index(k))
            lines.append(f"| {s}{'' if part == 'all' else ' (clips the shipped model never saw)'} | " + ' | '.join(names) + ' |')
            lines.append('|---|' + '---|' * len(names))
            for k in signs:
                lines.append(f'| {k} | ' + ' | '.join(
                    '{}/{}'.format(*b[threshold]['per_sign'].get(k, [0, 0])) for b in blocks) + ' |')
            lines.append('| all signs | ' + ' | '.join(_fmt(b[threshold]['signs_detected']) for b in blocks) + ' |')
            if blocks[0][threshold]['none_clips']:
                lines.append(f"| held letters with a false sign ({blocks[0][threshold]['none_clips']} clips) | " + ' | '.join(
                    _fmt(b[threshold]['none_false_sign']) for b in blocks) + ' |')
            lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lm')
    ap.add_argument('--eval', help='the eval dir (manifest.csv, duplicates.csv; results are written here)')
    ap.add_argument('--static')
    ap.add_argument('--motion')
    ap.add_argument('--name', help='results-<name>.json')
    ap.add_argument('--val', action='store_true', help='also score PopSign val (selection only)')
    ap.add_argument('--no-letters', action='store_true')
    ap.add_argument('--compare', nargs='+', help='result files to tabulate (first = baseline)')
    ap.add_argument('--threshold', default='0.9')
    a = ap.parse_args(argv)
    if a.compare:
        print(compare(a.compare, threshold=a.threshold))
        return 0
    res = evaluate(lm=a.lm, eval_dir=a.eval, static_path=a.static, motion_path=a.motion, val=a.val, letters=not a.no_letters)
    out = os.path.join(a.eval, f'results-{a.name}.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
