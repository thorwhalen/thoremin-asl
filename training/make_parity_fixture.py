"""Write ``test/fixtures/parity.json``: a few real hands and one real motion
window, with the features and probabilities THIS (training) code computes for them. The TS
test recomputes them with the plugin's recogniser and must agree, which pins the two featurisers
together. Re-run it whenever a model is re-exported.

usage: asl.py parity-fixture --lm LANDMARKS --static static.json --motion motion.json --out parity.json [--source signalphaset_dynamic,studiogalt] [--window-source popsign]

The fixture is committed, so both sources must be permissively licensed (defaults: SignAlphaSet
and PopSign v1.0, both CC BY 4.0, and StudioGalt, CC0, for a left hand: SignAlphaSet's signer
signs right-handed, and the fixture must exercise the mirroring).
"""
import argparse
import json
import sys

import numpy as np

from asl_common import featurize, hand_frames, iter_records, load_model, model_predict
from motion_common import frame_records, window_features


def browser_hand(h, w, hh):
    pts = np.asarray(h[2]).reshape(21, 3)
    wp = np.asarray(h[3]).reshape(21, 3)
    return {'handedness': h[0],
            'keypoints': [{'x': float(p[0] * w), 'y': float(p[1] * hh), 'z': float(p[2])} for p in pts],
            'worldKeypoints': [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in wp]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lm', required=True)
    ap.add_argument('--static', required=True)
    ap.add_argument('--motion', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='signalphaset_dynamic,studiogalt', help='video corpora with MediaPipe handedness and world landmarks, for the letter hands (comma-separated, in order)')
    ap.add_argument('--window-source', default='popsign', help='the same, for the motion window (a YES clip)')
    a = ap.parse_args(argv)
    static, motion = load_model(a.static), load_model(a.motion)
    hands = []
    for gloss in ('f', 'h', 'y'):
        for rec in (r for src in a.source.split(',') for r in iter_records(a.lm, src, {gloss})):
            fr = hand_frames(rec)
            if len(hands) >= 2:  # the last hand must be of a handedness not yet in the fixture
                seen = {x['hand']['handedness'] for x in hands}
                fr = [(t, h) for t, h in fr if h[0] not in seen]
            if not fr:
                continue
            t, h = fr[len(fr) // 2]
            x = featurize(h, rec['w'], rec['h'], static['features'])
            hands.append({'gloss': gloss, 'width': rec['w'], 'height': rec['h'], 'hand': browser_hand(h, rec['w'], rec['h']),
                          'features': x.round(6).tolist(), 'post': model_predict(static, x)[0].round(6).tolist()})
            break
    rec = next(iter_records(a.lm, a.window_source, {'yes'}))
    frames = hand_frames(rec)[:14]
    sub = dict(rec, frames=[[t, [h]] for t, h in frames])
    wf = window_features(frame_records(static, sub))
    window = {'width': rec['w'], 'height': rec['h'],
              'frames': [{'t': t, 'hand': browser_hand(h, rec['w'], rec['h'])} for t, h in frames],
              'features': wf.round(6).tolist(), 'probs': model_predict(motion, wf)[0].round(6).tolist()}
    json.dump({'hands': hands, 'window': window}, open(a.out, 'w'), indent=1)
    print('wrote', a.out, len(hands), 'hands,', len(frames), 'window frames')
    return 0


if __name__ == '__main__':
    sys.exit(main())
