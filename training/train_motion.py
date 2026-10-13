"""The motion model (J, Z, the seven words, and ``none``): windows of the signing hand's
recent frames, featurised exactly as ``src/lib/recognizer/motion.ts`` does, on top of a given
static model's letter posteriors (so a motion model belongs with the static model it was
trained against), classified by a small MLP.

Training clips are the protocol's motion ``train`` rows; ``none`` windows come from held-letter
clips. Selection uses the protocol's ``val`` rows (PopSign's val signers) scored the way the
browser scores a clip (``evaluate.eval_motion``), plus, with ``--cv``, a 5-fold grouped
cross-validation of the train clips (the only check for J, Z and sorry, which PopSign lacks).

Options worth knowing:
  --balance        class weights: 'class' (every class the same total weight), 'sqrt', 'none'
  --none-per-clip  how many random windows each held-letter clip contributes to ``none``
  --aug            augmented copies of each clip of the rare signs (``--aug-labels``): a random
                   in-plane rotation (+-``--aug-rot`` degrees, posteriors recomputed) and a
                   random speed change (x0.75 .. x1.33)
  --still-none     synthetic held-letter windows from still images (``--still-sources``): the
                   same hand repeated over 1.5 s with landmark jitter; per source and letter

usage: asl.py train-motion --lm LANDMARKS --eval EVAL_DIR --static static.json [--cv] [--export motion.json]
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asl_common import STATIC_LETTERS, export_pipeline, iter_records, load_model
from motion_common import LABELS, MIN_FRAMES, MOTION_SIGNS, T_PATH, T_POST, WINDOWS, frame_records, window_features
from sources import LICENCES

DEFAULT_SOURCES = ('popsign', 'aslense', 'studiogalt', 'popsign_negatives')
SWEEP = (0.8, 0.9, 0.95, 0.98)


def manifest_rows(eval_dir, role, sources=None):
    out = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(eval_dir, 'manifest.csv'))):
        if r['task'] == 'motion' and r['role'] == role and (sources is None or r['source'] in sources):
            out[r['source']][r['id']] = r
    return out


def load_clips(lm, rows):
    """Landmark records for manifest rows, each with ``_cls`` (its motion class) and ``_group``."""
    out = []
    for src, ids in rows.items():
        labels = {r['label'] for r in ids.values()}
        for rec in iter_records(lm, src, labels):
            if rec['id'] in ids:
                rec['_cls'] = ids[rec['id']]['note']
                rec['_group'] = f"{src}/{rec.get('signer') or rec['id']}"
                out.append(rec)
    return out


def train_windows(recs, *, step=1 / 6, min_frames=MIN_FRAMES):
    """Every (D, end) window over a clip on a 1/6 s grid of ends: frames in (end - D, end]."""
    if len(recs) < min_frames:
        return
    ts = np.array([t for t, _ in recs])
    for D in WINDOWS:
        e = ts[0] + D * 0.5
        while e <= ts[-1] + 1e-6:
            idx = np.where((ts > e - D) & (ts <= e))[0]
            if len(idx) >= min_frames:
                yield D, e, [recs[i] for i in idx]
            e += step


def augment(rec, rng, *, rot, speed=(0.75, 1.33)):
    """A copy of a clip turned in the image plane about its first wrist and played faster or slower."""
    first = next((h for _, hs in rec['frames'] for h in hs), None)
    if first is None:
        return None
    w, h = rec['w'], rec['h']
    x0, y0 = first[2][0] * w, first[2][1] * h
    a = np.radians(rng.uniform(-rot, rot))
    c, s = np.cos(a), np.sin(a)
    f = np.exp(rng.uniform(np.log(speed[0]), np.log(speed[1])))
    frames = []
    for t, hands in rec['frames']:
        nh = []
        for hd in hands:
            p = np.asarray(hd[2], np.float64).reshape(21, 3).copy()
            x, y = p[:, 0] * w - x0, p[:, 1] * h - y0
            p[:, 0] = (x0 + c * x - s * y) / w
            p[:, 1] = (y0 + s * x + c * y) / h
            nh.append([hd[0], hd[1], p.reshape(-1).tolist(), hd[3]])
        frames.append([t / f, nh])
    return dict(rec, frames=frames)


def still_clip(rec, rng, *, seconds=1.5, fps=12, jitter=0.004):
    """A still's hand held for ``seconds``: the same landmarks each frame plus small noise."""
    hands = rec['frames'][0][1]
    if not hands:
        return None
    from asl_common import pick_hand
    hd = pick_hand(hands)
    base = np.asarray(hd[2], np.float64)
    drift = rng.normal(0, jitter, 2)
    frames = []
    for k in range(int(seconds * fps)):
        p = (base + rng.normal(0, jitter * 0.5, base.shape)).reshape(21, 3)
        p[:, 0] += drift[0] * k / (seconds * fps)
        p[:, 1] += drift[1] * k / (seconds * fps)
        frames.append([k / fps, [[hd[0], hd[1], p.reshape(-1).tolist(), hd[3]]]])
    return dict(rec, frames=frames, _cls='none', _group=f"{rec['source']}/{rec.get('signer') or 'all'}")


def build(static, clips, *, none_per_clip, cover=0.6, seed=0):
    """Window features, labels and groups for clips (each with ``_cls``)."""
    rng = random.Random(seed)
    X, y, g, kept = [], [], [], []
    for c in clips:
        recs = frame_records(static, c)
        ws = list(train_windows(recs))
        if not ws:
            continue
        kept.append(c)
        if c['_cls'] == 'none':
            ws = rng.sample(ws, min(len(ws), none_per_clip))
        span = max(recs[-1][0] - recs[0][0], 1e-3)
        for D, e, fr in ws:
            if c['_cls'] != 'none' and (fr[-1][0] - fr[0][0]) / span < cover:
                continue
            X.append(window_features(fr))
            y.append(LABELS.index(c['_cls']))
            g.append(c['_group'])
    return np.array(X), np.array(y), np.array(g), kept


def class_weights(y, mode, *, none_weight=1.0):
    """Per-window weights: classes balanced ('class'), half-balanced ('sqrt') or not ('none'),
    then the ``none`` class scaled by ``none_weight`` (more weight = fewer false signs)."""
    c = Counter(y.tolist())
    k = len(c)
    if mode == 'none':
        per = {cl: 1.0 for cl in c}
    else:
        per = {cl: (len(y) / (k * n)) if mode == 'class' else np.sqrt(len(y) / (k * n)) for cl, n in c.items()}
    per[LABELS.index('none')] = per.get(LABELS.index('none'), 1.0) * none_weight
    return np.array([per[v] for v in y])


def make(hidden, alpha):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden, alpha=alpha, max_iter=600, random_state=0, early_stopping=False))


def fit(X, y, w, hidden, alpha):
    clf = make(hidden, alpha)
    clf.fit(X, y, mlpclassifier__sample_weight=w)
    return clf


def as_model(clf, threshold):
    """The fitted pipeline as an in-memory exported model (what evaluate.py scores)."""
    sc, est = clf.steps[0][1], clf.steps[-1][1]
    full = np.zeros((len(LABELS),), bool)
    full[est.classes_] = True
    layers = [{'W': W.T, 'b': b} for W, b in zip(est.coefs_, est.intercepts_)]
    if not full.all():  # a class absent from this fold: give it -inf logits
        W, b = layers[-1]['W'], layers[-1]['b']
        W2 = np.zeros((len(LABELS), W.shape[1]))
        b2 = np.full(len(LABELS), -1e9)
        W2[est.classes_], b2[est.classes_] = W, b
        layers[-1] = {'W': W2, 'b': b2}
    return {'labels': [l.upper() if len(l) == 1 else l for l in LABELS], 'mean': sc.mean_, 'scale': sc.scale_,
            'layers': layers, 'meta': {'windows': list(WINDOWS), 'minFrames': MIN_FRAMES, 'threshold': threshold}}


def report(name, res, th):
    s = res['all'][th]
    per = ' '.join(f'{k} {v[0]}/{v[1]}' for k, v in sorted(s['per_sign'].items(), key=lambda kv: MOTION_SIGNS.index(kv[0])))
    nf = '' if s['none_false_sign'] is None else f" | none false {s['none_false_sign']:.3f} of {s['none_clips']}"
    print(f'  [{name}] @{th} signs {s["signs_detected"]:.3f}{nf} | {per}', flush=True)


def main(argv=None):
    from evaluate import eval_motion
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lm', required=True)
    ap.add_argument('--eval', required=True)
    ap.add_argument('--static', required=True, help='the static model whose posteriors feed the windows')
    ap.add_argument('--sources', default=','.join(DEFAULT_SOURCES))
    ap.add_argument('--hidden', default='64')
    ap.add_argument('--alpha', type=float, default=1e-2)
    ap.add_argument('--balance', default='sqrt', choices=('class', 'sqrt', 'none'))
    ap.add_argument('--none-per-clip', type=int, default=40)
    ap.add_argument('--none-weight', type=float, default=1.0)
    ap.add_argument('--aug', type=int, default=0)
    ap.add_argument('--aug-labels', default='j,z,sorry')
    ap.add_argument('--aug-rot', type=float, default=15.0)
    ap.add_argument('--still-none', type=int, default=0)
    ap.add_argument('--still-sources', default='asl_hg,asl_alphabet_gestures')
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--exclude', action='append', help='SOURCE:label,label (repeatable): drop these clips')
    ap.add_argument('--cv', action='store_true')
    ap.add_argument('--no-val', action='store_true')
    ap.add_argument('--export')
    a = ap.parse_args(argv)
    hidden = tuple(int(v) for v in a.hidden.split(',') if v)
    th = str(a.threshold)
    static = load_model(a.static)
    t0 = time.time()
    clips = load_clips(a.lm, manifest_rows(a.eval, 'train', set(a.sources.split(','))))
    for spec in a.exclude or []:  # e.g. aslense:yes,no,hello (an ablation)
        src, labs = spec.split(':')
        clips = [c for c in clips if not (c['source'] == src and c['label'] in labs.split(','))]
    rng = np.random.default_rng(0)
    aug_labels = set(a.aug_labels.split(','))
    extra = [x for c in clips if c['_cls'] in aug_labels for _ in range(a.aug) for x in [augment(c, rng, rot=a.aug_rot)] if x]
    if a.still_none:
        train_ids = defaultdict(set)
        for r in csv.DictReader(open(os.path.join(a.eval, 'manifest.csv'))):
            if r['task'] == 'letters' and r['role'] == 'train':
                train_ids[r['source']].add(r['id'])
        for src in a.still_sources.split(','):
            by = defaultdict(list)
            for r in iter_records(a.lm, src, set(STATIC_LETTERS)):
                if r['id'] in train_ids[src] and r['frames'][0][1]:
                    by[r['label']].append(r)
            for lab, rs in by.items():
                for i in rng.choice(len(rs), min(a.still_none, len(rs)), replace=False):
                    x = still_clip(rs[i], rng)
                    if x:
                        extra.append(x)
    X, y, g, _ = build(static, clips + extra, none_per_clip=a.none_per_clip)
    w = class_weights(y, a.balance, none_weight=a.none_weight)
    print(f'clips {Counter(c["_cls"] for c in clips)} (+{len(extra)} augmented/synthetic)')
    print(f'windows {X.shape} {Counter(LABELS[i] for i in y)} ({time.time() - t0:.0f}s)', flush=True)

    if a.cv:  # grouped CV on the train clips, scored per clip as the browser would
        res_clips = []
        groups = np.array([c['_group'] for c in clips])
        for k, (tr, te) in enumerate(GroupKFold(5).split(groups, groups=groups)):
            test_groups = set(groups[te])
            m = np.array([gg not in test_groups for gg in g])
            clf = fit(X[m], y[m], w[m], hidden, a.alpha)
            te_clips = [dict(c, _note=c['_cls']) for c in clips if c['_group'] in test_groups]
            res_clips += eval_motion(static, as_model(clf, a.threshold), te_clips)['clips']
        for t in SWEEP:
            by = defaultdict(lambda: [0, 0])
            fp, fp_other, conf = [], [], Counter()
            for c in res_clips:
                if c['cls'] == 'none' and c['source'] == 'popsign_negatives':
                    fp_other.append(c['p'] >= t)
                elif c['cls'] == 'none':
                    fp.append(c['p'] >= t)
                    if c['p'] >= t:
                        conf[c['best']] += 1
                else:
                    by[c['cls']][0] += c['best'] == c['cls'] and c['p'] >= t
                    by[c['cls']][1] += 1
            print(f'  [cv] @{t} ' + ' '.join(f'{k} {v[0]}/{v[1]}' for k, v in sorted(by.items(), key=lambda kv: MOTION_SIGNS.index(kv[0])))
                  + f' | held letters false {np.mean(fp):.3f} of {len(fp)} {dict(conf.most_common(4))}'
                  + (f' | other signs false {np.mean(fp_other):.3f} of {len(fp_other)}' if fp_other else ''), flush=True)

    clf = fit(X, y, w, hidden, a.alpha)
    if not a.no_val:
        val = load_clips(a.lm, manifest_rows(a.eval, 'val'))
        res = eval_motion(static, as_model(clf, a.threshold), [dict(c, _note=c['_cls']) for c in val])
        for t in ('0.8', '0.9'):
            report('val', res, t)
        for t in SWEEP[2:]:
            sig = [c for c in res['clips'] if c['cls'] != 'none']
            print(f"  [val] @{t} signs {np.mean([c['best'] == c['cls'] and c['p'] >= t for c in sig]):.3f}", flush=True)
    if a.export:
        size = export_pipeline(clf, labels=[l.upper() if len(l) == 1 else l for l in LABELS], features='asl-motion/1', path=a.export, meta={
            'windows': list(WINDOWS), 'tPost': T_POST, 'tPath': T_PATH, 'minFrames': MIN_FRAMES, 'threshold': a.threshold,
            'staticModel': os.path.basename(a.static), 'trainedOn': a.sources.split(','),
            'licences': {s: LICENCES[s][0] for s in a.sources.split(',') + (a.still_sources.split(',') if a.still_none else [])},
            'recipe': {'hidden': list(hidden), 'alpha': a.alpha, 'balance': a.balance, 'noneWeight': a.none_weight, 'nonePerClip': a.none_per_clip,
                       'aug': a.aug, 'augLabels': a.aug_labels, 'stillNone': a.still_none,
                       'stillSources': a.still_sources if a.still_none else None, 'exclude': a.exclude}})
        print(f'exported {a.export} ({size} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
