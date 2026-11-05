"""Write ``test/fixtures/motion_letters.json.gz``: SignAlphaSet's J and Z clips (all ten each)
and one clip of every held letter, as browser-shaped hand frames, for the test that holds J and
Z to a detection rate and the held letters to a false-motion-letter rate.

SignAlphaSet (B. Garg et al., Mendeley Data, 2025, doi:10.17632/8fmvr9m98w.3) is CC BY 4.0, so
the landmarks may be committed with the attribution below; no thoremin-asl model is trained on
it, so the recogniser meets this signer for the first time. The landmarks were extracted at
30 fps by thoremin's ``scripts/video_to_landmarks.py`` (MediaPipe HandLandmarker, VIDEO mode)
into thoremin's app-data dir (``landmarks/signalphaset/<clip>.hands.ndjson.gz``), from the
clips read out of the dataset's ``ASL_dynamic.zip`` by HTTP range requests.

usage: python training/make_motion_letters_fixture.py [LANDMARK_DIR]
"""
import glob
import gzip
import json
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.local/share/thoremin/landmarks/signalphaset')
OUT = os.path.join(os.path.dirname(__file__), '..', 'test', 'fixtures', 'motion_letters.json.gz')
ATTRIBUTION = 'SignAlphaSet (B. Garg, M. Kasar, A. Kashyap, A. Vats, G. Sharma, A. Hange; Mendeley Data, 2025, doi:10.17632/8fmvr9m98w.3), CC BY 4.0. MediaPipe hand landmarks extracted at 30 fps; x, y rounded to 0.1 px, z to 1e-4; world landmarks dropped.'


def hand(h):
    return {'handedness': h['handedness'],
            'keypoints': [{'x': round(p['x'], 1), 'y': round(p['y'], 1), 'z': round(p.get('z', 0), 4)} for p in h['keypoints']]}


def clip(path):
    frames, width, height = [], None, None
    for line in gzip.open(path, 'rt'):
        r = json.loads(line)
        width, height = r['value']['width'], r['value']['height']
        frames.append({'t': round(r['t'], 4), 'hands': [hand(h) for h in r['value']['hands']]})
    return {'video': os.path.basename(path).replace('.hands.ndjson.gz', '.avi'), 'width': width, 'height': height, 'frames': frames}


clips = []
for path in sorted(glob.glob(os.path.join(SRC, '*_clip*.hands.ndjson.gz'))):
    m = re.match(r'([A-Z])_clip(\d+)', os.path.basename(path))
    if not m:
        continue
    letter, k = m.group(1), int(m.group(2))
    if letter in 'JZ' or k == 1:
        clips.append({'letter': letter, **clip(path)})
with gzip.GzipFile(OUT, 'wb', mtime=0) as f:
    f.write(json.dumps({'source': ATTRIBUTION, 'clips': clips}, separators=(',', ':')).encode())
print(len(clips), 'clips ->', OUT, os.path.getsize(OUT), 'bytes')
