"""Motion-sign window features and the browser's clip-level motion scoring.

``window_features`` MUST match ``windowFeatures`` in ``src/lib/recognizer/motion.ts``. A window
is the signing hand's frames over the last D seconds; its features are:
  - the static model's letter posterior, resampled at T_POST times  (handshape over time)
  - the wrist path, relative to the window's first wrist, in palm lengths, at T_PATH times
  - the index and pinky tips, relative to that first wrist, in palm lengths, at T_PATH times
  - the wrist path length (palm lengths) and the window's duration (s)
x is mirrored for a left hand, as in the static features.

``browser_windows`` replays a clip the way ``recognizer.ts`` buffers it (the frame-gap
filter, the 2 s history, the dropout reset) and ``scoreMotion`` scores it (every window
length ending at the newest frame), so a clip-level evaluation measures what the app does.
"""
import numpy as np

from asl_common import featurize, hand_frames, model_predict

T_POST, T_PATH = 4, 8
WINDOWS = (0.6, 0.9, 1.3, 1.8)
MOTION_SIGNS = ['j', 'z', 'yes', 'no', 'hello', 'thankyou', 'please', 'sorry', 'bye']
LABELS = MOTION_SIGNS + ['none']
# recognizer.ts defaults
MIN_FRAMES, HISTORY, MIN_FRAME_GAP, DROPOUT = 4, 2.0, 1 / 15, 0.35


def frame_records(static, rec):
    """Per-frame quantities the window featuriser needs (what the TS buffer stores), for the
    hand-bearing frames of a landmark record: ``[(t, {post, wrist, index, pinky, palm})]``."""
    fr = hand_frames(rec)
    if not fr:
        return []
    w, h = rec['w'], rec['h']
    post = model_predict(static, np.array([featurize(hd, w, h, static['features']) for _, hd in fr]))
    out = []
    for (t, hd), p in zip(fr, post):
        pts = np.asarray(hd[2], np.float64).reshape(21, 3)
        px = pts[:, 0] * w
        py = pts[:, 1] * h
        if hd[0] == 'Left':
            px = -px
        palm = float(np.hypot(px[9] - px[0], py[9] - py[0])) or 1.0
        out.append((t, {'post': p, 'wrist': (px[0], py[0]), 'index': (px[8], py[8]), 'pinky': (px[20], py[20]), 'palm': palm}))
    return out


def _resample(ts, vals, n):
    ts = np.asarray(ts)
    vals = np.asarray(vals, np.float64)
    tt = np.linspace(ts[0], ts[-1], n) if ts[-1] > ts[0] else np.full(n, ts[0])
    return np.stack([np.interp(tt, ts, vals[:, j]) for j in range(vals.shape[1])], 1)


def window_features(frames):
    """frames: list of (t, record), time-ascending, >= 2 entries."""
    ts = [t for t, _ in frames]
    R = [r for _, r in frames]
    palm = float(np.mean([r['palm'] for r in R])) or 1.0
    w0 = np.asarray(R[0]['wrist'])

    def rel(key):
        return [(np.asarray(r[key]) - w0) / palm for r in R]

    post = _resample(ts, [r['post'] for r in R], T_POST).reshape(-1)
    wrist = _resample(ts, rel('wrist'), T_PATH).reshape(-1)
    tips = np.concatenate([_resample(ts, rel('index'), T_PATH), _resample(ts, rel('pinky'), T_PATH)], 1).reshape(-1)
    W = np.asarray([r['wrist'] for r in R])
    path = float(np.sum(np.hypot(*np.diff(W, axis=0).T))) / palm if len(W) > 1 else 0.0
    return np.concatenate([post, wrist, tips, [path, ts[-1] - ts[0]]])


def browser_windows(rec, recs, *, windows=WINDOWS, min_frames=MIN_FRAMES):
    """Every window the browser scores while this clip plays: ``[(frame_index, D, frames)]``.

    ``recs`` are ``frame_records(static, rec)``; the frames without a hand in ``rec`` drive the
    dropout reset, as in ``recognizer.ts``."""
    hand_t = {t for t, _ in recs}
    by_t = dict(recs)
    history, out, last_hand_t = [], [], -np.inf
    for k, (t, hands) in enumerate(rec['frames']):
        if t not in hand_t:
            if t - last_hand_t > DROPOUT:
                history = []
            continue
        last_hand_t = t
        if not history or t - history[-1][0] >= MIN_FRAME_GAP - 1e-6:
            history.append((t, by_t[t]))
        history = [f for f in history if f[0] >= t - HISTORY]
        for D in windows:
            win = [f for f in history if f[0] > t - D - 1e-9]
            if len(win) >= min_frames:
                out.append((k, D, win))
    return out


def clip_scores(motion, rec, recs):
    """The browser's per-frame motion output over a clip: for each frame, the probabilities of
    the window whose best sign is most confident (``scoreMotion``). Returns an (n, classes)
    array, empty when no window fills."""
    ws = browser_windows(rec, recs, windows=tuple(motion.get('meta', {}).get('windows', WINDOWS)),
                         min_frames=motion.get('meta', {}).get('minFrames', MIN_FRAMES))
    if not ws:
        return np.zeros((0, len(motion['labels'])))
    P = model_predict(motion, np.array([window_features(w) for _, _, w in ws]))
    none = motion['labels'].index('none')
    sign = P.copy()
    sign[:, none] = -1
    best = {}
    for (k, _, _), p, s in zip(ws, P, sign.max(1)):
        if k not in best or s > best[k][0]:
            best[k] = (s, p)
    return np.array([best[k][1] for k in sorted(best)])
