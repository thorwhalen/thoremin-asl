"""The static letter model: geo features (as the shipped one) from every train-eligible
corpus in the protocol's manifest, model selection by LEAVE-ONE-SOURCE-OUT validation
(train on all sources but one, score the held-out one; the mean over sources is the
selection score), and export in the ``thoremin-asl-mlp/1`` format.

No test set is read here: the test numbers come from ``evaluate.py`` on the exported model.

Options worth knowing:
  --sources        which train-eligible corpora to use (default: the five validation kept;
                   American Hand Signs is train-eligible but was dropped by validation)
  --cap            at most this many samples per (source, letter), so one big single-signer
                   corpus does not drown the rest (deterministic sample)
  --balance        'source': every source carries the same total weight (default); 'none'
  --rot            in-plane rotation augmentation, +- degrees (0 = off), ``--aug-copies`` copies
  --models         comma list of MODELS keys; --loso runs the validation for each

usage: asl.py train-static --lm LANDMARKS --eval EVAL_DIR [--loso] [--export static.json --export-model M]
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import handedness as H
from asl_common import STATIC_LETTERS, export_pipeline, featurize, held_frames, iter_records
from sources import LICENCES

MODELS = {
    'logreg_c0.01': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.01)),
    'logreg_c0.03': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.03)),
    'logreg_c0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1)),
    'logreg_c0.3': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.3)),
    'logreg_c1': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0)),
    'mlp64_a1': lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=1.0, max_iter=600, random_state=0)),
    'mlp64_a0.1': lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=0.1, max_iter=600, random_state=0)),
    'mlp128_a1': lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1.0, max_iter=600, random_state=0)),
    'mlp128_a3': lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=3.0, max_iter=600, random_state=0)),
    'mlp128_a0.3': lambda: make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=0.3, max_iter=600, random_state=0)),
}

DEFAULT_SOURCES = ('aslnow', 'asl_hg', 'asl_alphabet_gestures', 'om192006', 'studiogalt')
KEYPOINT_SOURCES = ('aslnow', 'om192006')


def train_rows(eval_dir, sources):
    rows = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(eval_dir, 'manifest.csv'))):
        if r['task'] == 'letters' and r['role'] == 'train' and r['source'] in sources:
            rows[r['source']].add(r['id'])
    return rows


def rotate_hand(hand, w, h, deg):
    """The hand turned by ``deg`` in the image plane about its wrist (pixel space, so the frame's
    aspect is respected); z is left as is."""
    p = np.asarray(hand[2], np.float64).reshape(21, 3).copy()
    x, y = p[:, 0] * w, p[:, 1] * h
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    x0, y0 = x[0], y[0]
    xr = x0 + c * (x - x0) - s * (y - y0)
    yr = y0 + s * (x - x0) + c * (y - y0)
    p[:, 0], p[:, 1] = xr / w, yr / h
    return [hand[0], hand[1], p.reshape(-1).tolist(), hand[3]]


def collect(lm, eval_dir, sources, *, cap=None, seed=0):
    """Samples ``(hand, w, h, label_index, source, group)`` from the manifest's train rows."""
    rows = train_rows(eval_dir, sources)
    hclf = H.fit(lm) if any(s in KEYPOINT_SOURCES for s in sources) else None
    rng = np.random.default_rng(seed)
    out = []
    for src in sources:
        recs = [r for r in iter_records(lm, src, set(STATIC_LETTERS)) if r['id'] in rows[src]]
        if src in KEYPOINT_SOURCES:
            H.label_hands(hclf, recs)
        by = defaultdict(list)
        for r in recs:
            for hd in held_frames(r):
                by[r['label']].append((hd, r['w'], r['h'], STATIC_LETTERS.index(r['label']), src,
                                       f"{src}/{r.get('signer') or (r['id'] if r['kind'] == 'video' else 'all')}"))
        for lab, xs in by.items():
            if cap and len(xs) > cap:
                xs = [xs[i] for i in sorted(rng.choice(len(xs), cap, replace=False))]
            out += xs
    return out


def featurise(samples, *, rot=0.0, copies=0, seed=0):
    rng = np.random.default_rng(seed)
    X, idx = [], []
    for i, (hd, w, h, *_rest) in enumerate(samples):
        X.append(featurize(hd, w, h, 'geo'))
        idx.append(i)
        for _ in range(copies if rot else 0):
            X.append(featurize(rotate_hand(hd, w, h, rng.uniform(-rot, rot)), w, h, 'geo'))
            idx.append(i)
    idx = np.array(idx)
    y = np.array([samples[i][3] for i in idx])
    src = np.array([samples[i][4] for i in idx])
    return np.array(X, np.float32), y, src


def weights(src, balance):
    if balance == 'none':
        return np.ones(len(src))
    names, counts = np.unique(src, return_counts=True)
    per = dict(zip(names, len(src) / (len(names) * counts)))
    return np.array([per[s] for s in src])


def fit(model, X, y, w):
    clf = MODELS[model]()
    clf.fit(X, y, **{f'{clf.steps[-1][0]}__sample_weight': w})
    return clf


def loso(model, X, y, src, w, *, train_sources, val_sources):
    """Leave-one-source-out: for each validation source, fit on the train sources minus it and
    score it. Every config is scored on the same validation sources (also those it does not
    train on), so means are comparable across source choices."""
    res = {}
    for s in val_sources:
        te = src == s
        tr = np.isin(src, [t for t in train_sources if t != s])
        if not te.any():
            continue
        clf = fit(model, X[tr], y[tr], w[tr])
        P = clf.predict_proba(X[te])
        cls = clf.classes_
        pred = cls[P.argmax(1)]
        top3 = (cls[np.argsort(-P, 1)[:, :3]] == y[te][:, None]).any(1)
        res[s] = (float((pred == y[te]).mean()), float(top3.mean()), int(te.sum()))
    return res


def loso_aug(model, X, y, src, w, X0, y0, src0, train_sources, val_sources):
    """``loso`` with augmented training rows and un-augmented validation rows."""
    res = {}
    for s in val_sources:
        te = src0 == s
        tr = np.isin(src, [t for t in train_sources if t != s])
        if not te.any():
            continue
        clf = fit(model, X[tr], y[tr], w[tr])
        P = clf.predict_proba(X0[te])
        cls = clf.classes_
        pred = cls[P.argmax(1)]
        top3 = (cls[np.argsort(-P, 1)[:, :3]] == y0[te][:, None]).any(1)
        res[s] = (float((pred == y0[te]).mean()), float(top3.mean()), int(te.sum()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lm', required=True)
    ap.add_argument('--eval', required=True, help='the eval dir holding manifest.csv')
    ap.add_argument('--sources', default=','.join(DEFAULT_SOURCES))
    ap.add_argument('--cap', type=int, default=400)
    ap.add_argument('--balance', default='source', choices=('source', 'none'))
    ap.add_argument('--rot', type=float, default=0.0)
    ap.add_argument('--aug-copies', type=int, default=1)
    ap.add_argument('--models', default='logreg_c0.1')
    ap.add_argument('--loso', action='store_true')
    ap.add_argument('--val-sources', default='aslnow,asl_hg,asl_alphabet_gestures,om192006',
                    help='the sources leave-one-source-out scores (each is held out in turn)')
    ap.add_argument('--export')
    ap.add_argument('--export-model')
    a = ap.parse_args(argv)
    sources = a.sources.split(',')
    val_sources = a.val_sources.split(',') if a.loso else []
    t0 = time.time()
    samples = collect(a.lm, a.eval, list(dict.fromkeys(sources + val_sources)), cap=a.cap)
    X, y, src = featurise(samples, rot=a.rot, copies=a.aug_copies)
    if a.rot:  # validation hands are never augmented copies
        X0, y0, src0 = featurise(samples)
    w = weights(src, a.balance)
    n = {s: int((src == s).sum()) for s in sources}
    print(f'{len(X)} samples {n} ({time.time() - t0:.0f}s)', flush=True)
    for m in a.models.split(',') if a.loso else []:
        t0 = time.time()
        r = loso_aug(m, X, y, src, w, X0, y0, src0, sources, val_sources) if a.rot else \
            loso(m, X, y, src, w, train_sources=sources, val_sources=val_sources)
        mean1 = np.mean([v[0] for v in r.values()])
        mean3 = np.mean([v[1] for v in r.values()])
        detail = ' '.join(f'{s} {v[0]:.3f}' for s, v in r.items())
        print(f'[loso] {m:12s} mean top1 {mean1:.3f} top3 {mean3:.3f} | {detail} ({time.time() - t0:.0f}s)', flush=True)
    if a.export:
        keep = np.isin(src, sources)
        X, y, src, w = X[keep], y[keep], src[keep], weights(src[keep], a.balance)
        m = a.export_model or a.models.split(',')[0]
        clf = fit(m, X, y, w)
        size = export_pipeline(clf, labels=[l.upper() for l in STATIC_LETTERS], features='geo', path=a.export, meta={
            'model': m, 'trainedOn': sources, 'cap': a.cap, 'balance': a.balance, 'rot': a.rot,
            'licences': {s: LICENCES[s][0] for s in sources}})
        print(f'exported {a.export} ({size} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
