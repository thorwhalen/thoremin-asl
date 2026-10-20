"""Write ``test/fixtures/clips.json``: a few real clips as browser-shaped hand frames, for the
replay test of the whole recogniser (fusion, smoothing, hysteresis, latch over real streams).
It pins behaviour, not accuracy: the accuracy numbers come from the held-out test sets.

Only permissively licensed clips go in the fixture, since it is committed: by default letters
from SignAlphaSet (CC BY 4.0) and words from PopSign v1.0 (CC BY 4.0). Each spec is
``source:gloss:which``, where ``which`` is a record id or the index of the record in file
order. The default clips are ones the shipped pair reads correctly (a behaviour replay needs
the reading to happen); re-pick them when a re-exported model reads one differently.

usage: asl.py replay-fixture --lm LANDMARKS --out clips.json [SPEC ...]
"""
import argparse
import json
import sys

import numpy as np

from asl_common import iter_records

DEFAULT_SPECS = ('signalphaset_dynamic:b:B_clip5.avi', 'signalphaset_dynamic:l:L_clip2.avi', 'signalphaset_dynamic:y:Y_clip7.avi',
                 'popsign:yes:game/train/4a.8006-yes-2022_12_19_12_45_16.723-0.mp4',
                 'popsign:please:game/train/4a.8017-please-2023_01_15_16_26_40.768-0.mp4',
                 'popsign:thankyou:game/train/4a.7003-thankyou-2022_12_15_13_21_36.667-0.mp4')
ATTRIBUTION = {
    'signalphaset_dynamic': 'SignAlphaSet (Mendeley Data, doi:10.17632/8fmvr9m98w.3, CC BY 4.0)',
    'popsign': 'PopSign ASL v1.0 (Georgia Tech / NTID / Google, CC BY 4.0)',
    'studiogalt': 'StudioGalt Sign-Language-Mocap-Archive (CC0 1.0)',
}


def _browser_hand(h, w, hh):
    pts = np.asarray(h[2]).reshape(21, 3)
    wp = np.asarray(h[3]).reshape(21, 3)
    return {'handedness': h[0],
            'keypoints': [{'x': round(float(p[0] * w), 1), 'y': round(float(p[1] * hh), 1), 'z': round(float(p[2]), 4)} for p in pts],
            'worldKeypoints': [{'x': round(float(p[0]), 4), 'y': round(float(p[1]), 4), 'z': round(float(p[2]), 4)} for p in wp]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('specs', nargs='*', default=list(DEFAULT_SPECS))
    a = ap.parse_args(argv)
    clips, sources = [], set()
    for spec in a.specs:
        source, gloss, which = spec.split(':', 2)
        if source not in ATTRIBUTION:
            sys.exit(f'{source}: not a permissively licensed source this fixture may carry')
        recs = list(iter_records(a.lm, source, {gloss}))
        c = recs[int(which)] if which.isdigit() else next((r for r in recs if r['id'] == which), None)
        if c is None:
            sys.exit(f'{spec}: no such record')
        frames = [{'t': t, 'hands': [_browser_hand(h, c['w'], c['h']) for h in hands]} for t, hands in c['frames']]
        clips.append({'gloss': gloss, 'source': source, 'video': c['id'], 'width': c['w'], 'height': c['h'], 'frames': frames})
        sources.add(source)
    json.dump({'source': '; '.join(ATTRIBUTION[s] for s in sorted(sources)) + '. MediaPipe hand landmarks.', 'clips': clips},
              open(a.out, 'w'), separators=(',', ':'))
    print('wrote', a.out, [(c['gloss'], len(c['frames'])) for c in clips])
    return 0


if __name__ == '__main__':
    sys.exit(main())
