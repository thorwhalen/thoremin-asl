"""Shared vocabulary, landmark-record loading and the letter featurisers of the ASL models.

The featurisers MUST match ``src/lib/recognizer/features.ts`` (this repository): the exported
weights mean nothing otherwise, and the parity fixture (``make_parity_fixture.py``) pins
the two together. ``geo`` is the shipped letter feature; ``world``, ``image``, ``both`` and
``geo+image`` remain for comparison.

A landmark record (written by ``extract.py``) is one clip or still::

    {source, id, label, signer, split, kind, w, h, fps, frames: [[t, hands], ...]}

and a hand is ``[handedness, score, img(63), world(63)]`` (image landmarks normalised to
0..1, world landmarks in metres; handedness ``None`` for corpora shipped as bare landmarks).
"""
import json
import os

import numpy as np

LETTERS = [chr(c) for c in range(ord('a'), ord('z') + 1)]
STATIC_LETTERS = [l for l in LETTERS if l not in ('j', 'z')]
WORDS = ['yes', 'no', 'hello', 'thankyou', 'please', 'sorry', 'bye']


# ---- landmark records -----------------------------------------------------------------

def iter_records(lm_root, source, labels=None):
    """Records of one corpus under ``<lm_root>/<source>/<label>.jsonl`` (failed extractions skipped)."""
    d = os.path.join(lm_root, source)
    if not os.path.isdir(d):
        return
    for name in sorted(os.listdir(d)):
        if not name.endswith('.jsonl') or (labels is not None and name[:-6] not in labels):
            continue
        for line in open(os.path.join(d, name)):
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut by an interrupted extraction
            if 'error' not in r:
                yield r


def pick_hand(hands):
    """The signing hand: the one whose wrist is highest in the image (smallest y)."""
    if not hands:
        return None
    return min(hands, key=lambda h: h[2][1])


def hand_frames(rec):
    """``[(t, hand)]`` for the frames of a record where a hand was found (the signing hand)."""
    return [(t, pick_hand(h)) for t, h in rec['frames'] if h]


def held_frames(rec, lo=0.3, hi=0.85):
    """The hands a letter is held in: a still's one hand, or a clip's frames between ``lo`` and
    ``hi`` of its hand-bearing frames (the shipped recipe: skips the hand rising and leaving)."""
    fr = [h for _, h in hand_frames(rec)]
    if len(fr) <= 1:
        return fr
    n = len(fr)
    return fr[int(n * lo):max(int(n * lo) + 1, int(n * hi))]


# ---- raw-coordinate featurisers (comparison only) --------------------------------------

def canon_world(hand):
    """World landmarks, wrist-relative, scaled by wrist->middle-MCP length, mirrored for Left."""
    w = np.asarray(hand[3], dtype=np.float32).reshape(21, 3)
    w = w - w[0]
    s = np.linalg.norm(w[9]) or 1.0
    w = w / s
    if hand[0] == 'Left':
        w[:, 0] = -w[:, 0]
    return w.reshape(-1)


def canon_image(hand, width, height):
    """Image landmarks in pixels (z scaled like x), wrist-relative, scaled, mirrored for Left."""
    p = np.asarray(hand[2], dtype=np.float32).reshape(21, 3).copy()
    p[:, 0] *= width
    p[:, 1] *= height
    p[:, 2] *= width
    p = p - p[0]
    s = np.linalg.norm(p[9, :2]) or 1.0
    p = p / s
    if hand[0] == 'Left':
        p[:, 0] = -p[:, 0]
    return p.reshape(-1)


# ---- geometric features (kind 'geo', the shipped one) ---------------------------------
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def image_points(hand, width, height):
    """Image landmarks as 3D points in pixels (z scaled like x), mirrored for a Left hand."""
    p = np.asarray(hand[2], dtype=np.float64).reshape(21, 3).copy()
    p[:, 0] *= width
    p[:, 1] *= height
    p[:, 2] *= width
    if hand[0] == 'Left':
        p[:, 0] = -p[:, 0]
    return p


def geo_features(P):
    """P: 21x3 canonical (mirrored) points in any consistent unit -> 66 shape features,
    invariant to position and scale. Mirrors ``geoFeatures`` in features.ts."""
    P = P - P[0]
    s = np.linalg.norm(P[9]) or 1.0
    P = P / s
    f = []
    for ch in FINGERS:  # flexion: cos of the angle at each of the 3 inner joints
        for j in range(1, 4):
            f.append(float(np.dot(_unit(P[ch[j]] - P[ch[j - 1]]), _unit(P[ch[j + 1]] - P[ch[j]]))))
    for a, b in [(1, 2), (2, 3), (3, 4)]:  # spread between adjacent fingers' proximal bones
        f.append(float(np.dot(_unit(P[FINGERS[a][2]] - P[FINGERS[a][1]]), _unit(P[FINGERS[b][2]] - P[FINGERS[b][1]]))))
    f.append(float(np.dot(_unit(P[3] - P[2]), _unit(P[6] - P[5]))))  # thumb vs index
    for i in range(5):
        for j in range(i + 1, 5):
            f.append(float(np.linalg.norm(P[TIPS[i]] - P[TIPS[j]])))
    for t in TIPS:
        f.append(float(np.linalg.norm(P[t])))
    for k in (5, 6, 9, 10, 13, 14, 17, 18):  # thumb tip to knuckles
        f.append(float(np.linalg.norm(P[4] - P[k])))
    up = _unit(P[9])
    across = _unit(P[5] - P[17])
    normal = _unit(np.cross(P[5], P[17]))
    f += up.tolist() + across.tolist() + normal.tolist()
    for t in TIPS:  # each finger's pointing direction (camera frame)
        f += _unit(P[t] - P[t - 3]).tolist()
    return np.asarray(f, dtype=np.float32)


def featurize(hand, width, height, kind='geo'):
    """The feature vector of ``kind`` for one hand in a ``width`` x ``height`` frame."""
    if kind == 'geo':
        return geo_features(image_points(hand, width, height))
    if kind == 'geo+image':
        return np.concatenate([geo_features(image_points(hand, width, height)), canon_image(hand, width, height)])
    if kind == 'world':
        return canon_world(hand)
    if kind == 'image':
        return canon_image(hand, width, height)
    if kind == 'both':
        return np.concatenate([canon_world(hand), canon_image(hand, width, height)])
    raise ValueError(f'unknown feature kind {kind!r}')


# ---- the exported model format (thoremin-asl-mlp/1) -----------------------------------

def export_pipeline(clf, *, labels, features, meta, path):
    """Write a fitted ``StandardScaler -> LogisticRegression | MLPClassifier`` pipeline in the
    format ``src/lib/recognizer/mlp.ts`` reads. A logistic regression is the zero-hidden-layer
    case. Returns the file size in bytes."""
    sc, est = clf.steps[0][1], clf.steps[-1][1]
    if hasattr(est, 'coef_'):
        layers = [{'W': est.coef_.round(5).tolist(), 'b': est.intercept_.round(5).tolist()}]
    else:
        layers = [{'W': W.T.round(5).tolist(), 'b': b.round(5).tolist()} for W, b in zip(est.coefs_, est.intercepts_)]
    present = [int(c) for c in est.classes_]
    if present != list(range(len(labels))):
        # a label with no training data: its output row gets a large negative bias (never wins)
        W, b = layers[-1]['W'], layers[-1]['b']
        W2 = [[0.0] * len(W[0]) for _ in labels]
        b2 = [-30.0] * len(labels)
        for row, c in enumerate(present):
            W2[c], b2[c] = W[row], b[row]
        layers[-1] = {'W': W2, 'b': b2}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump({'format': 'thoremin-asl-mlp/1', 'features': features, 'labels': labels,
               'mean': sc.mean_.round(5).tolist(), 'scale': sc.scale_.round(5).tolist(), 'layers': layers,
               'meta': meta}, open(path, 'w'), separators=(',', ':'))
    return os.path.getsize(path)


def load_model(path):
    return json.load(open(path))


def model_predict(m, X):
    """Class probabilities of an exported model for the rows of X (what mlp.ts computes)."""
    A = (np.atleast_2d(np.asarray(X, np.float64)) - np.asarray(m['mean'])) / np.where(np.asarray(m['scale']) == 0, 1, np.asarray(m['scale']))
    for i, L in enumerate(m['layers']):
        A = A @ np.asarray(L['W']).T + np.asarray(L['b'])
        if i < len(m['layers']) - 1:
            A = np.maximum(A, 0)
    A = A - A.max(1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(1, keepdims=True)
