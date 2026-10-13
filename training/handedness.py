"""Geometry -> MediaPipe handedness, for corpora that ship bare landmarks (ASLNow!, om192006).

The letter features mirror a left hand, so a hand without MediaPipe's Left/Right label must
get one. A logistic regression on the unmirrored geo features learns it from corpora whose
MediaPipe runs carry the label. The CLI only reports its grouped cross-validation.

usage: asl.py handedness <landmarks_dir> [--sources aslense,asl_hg,asl_alphabet_gestures]
"""
import argparse
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asl_common import geo_features, hand_frames, iter_records

DEFAULT_SOURCES = ('aslense', 'asl_hg', 'asl_alphabet_gestures')


def raw_points(img_flat, w, h):
    """Image landmarks as 3D pixels, NOT mirrored (the input of the handedness classifier)."""
    p = np.asarray(img_flat, dtype=np.float64).reshape(21, 3).copy()
    p[:, 0] *= w
    p[:, 1] *= h
    p[:, 2] *= w
    return p


def dataset(lm, sources=DEFAULT_SOURCES, *, every=3, per_source=20000):
    X, y, g = [], [], []
    for src in sources:
        n = 0
        for r in iter_records(lm, src):
            for t, hd in hand_frames(r)[::every]:
                X.append(geo_features(raw_points(hd[2], r['w'], r['h'])))
                y.append(hd[0] == 'Left')
                g.append(f"{src}/{r['signer'] or r['id']}")
                n += 1
            if n >= per_source:
                break
    return np.array(X), np.array(y), np.array(g)


def make():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def fit(lm, sources=DEFAULT_SOURCES):
    X, y, _ = dataset(lm, sources)
    return make().fit(X, y)


def label_hands(clf, recs):
    """Fill in the handedness of hands that lack one (in place); returns ``recs``."""
    todo = [(hd, r) for r in recs for _, hands in r['frames'] for hd in hands if hd[0] is None]
    if todo:
        X = np.array([geo_features(raw_points(hd[2], r['w'], r['h'])) for hd, r in todo])
        for (hd, _), left in zip(todo, clf.predict(X)):
            hd[0] = 'Left' if left else 'Right'
    return recs


def main(argv=None):
    from sklearn.model_selection import GroupKFold, cross_val_score
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('lm')
    ap.add_argument('--sources', default=','.join(DEFAULT_SOURCES))
    a = ap.parse_args(argv)
    X, y, g = dataset(a.lm, a.sources.split(','))
    print(len(X), 'hands, Left share', y.mean().round(3))
    print('grouped CV acc', cross_val_score(make(), X, y, groups=g, cv=GroupKFold(5)).round(3))


if __name__ == '__main__':
    sys.exit(main())
