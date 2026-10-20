"""Hand landmarks for every item of the chosen corpora (see ``sources.py``), in parallel,
with the browser's own model (MediaPipe ``hand_landmarker.task``, float16).

Output: ``<out>/<source>/<label>.jsonl``, one line per clip or image::

    {source, id, label, signer, split, kind, w, h, fps, frames: [[t, hands], ...], ...}

``hands`` is a list of ``[handedness, score, img(63), world(63)]``: image landmarks are
MediaPipe's normalised x, y (0..1) and z; world landmarks are metres. A still image is a
one-frame record (``t = 0``). Corpora that ship landmarks instead of pixels (ASLNow!,
om192006) are converted without a MediaPipe run: their hands are ``[None, None, img, None]``
(no handedness) and their frame size is the assumed ``--keypoint-size``.

Extractor settings are those of the first pipeline (the models shipped before 2026-10):
videos in VIDEO mode, 2 hands, detection/presence/tracking confidence 0.5, every
``round(fps / target_fps)``-th frame; stills in IMAGE mode, 2 hands, detection confidence
0.3, padded by 35% of their longer side with grey (tight crops defeat the palm detector).
One addition: frames whose longer side exceeds ``--max-side`` are downscaled first (PopSign
is 1944x2592), which keeps the aspect ratio and so the features.

Resumable: items whose id is already in the output file are skipped. Memory is checked
before every submission: no new work starts while MemAvailable is under ``--min-avail-mb``.

usage: extract.py --raw RAW --out OUT --model hand_landmarker.task --sources popsign,wlasl [--workers 12]
"""
import argparse
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import sources as S


def mem_available_mb():
    """MemAvailable from /proc/meminfo, in MB (None where there is no /proc)."""
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except OSError:
        return None


_CFG = {}
_CACHE = {}


def _init(cfg):
    import cv2
    cv2.setNumThreads(1)
    _CFG.update(cfg)


def _r4(v):
    return round(v, 4)


def _hands(res):
    return [[res.handedness[k][0].category_name, _r4(res.handedness[k][0].score),
             [_r4(c) for p in pts for c in (p.x, p.y, p.z)],
             [_r4(c) for p in res.hand_world_landmarks[k] for c in (p.x, p.y, p.z)]]
            for k, pts in enumerate(res.hand_landmarks)]


def _landmarker(mode):
    from mediapipe.tasks.python import BaseOptions, vision
    opts = dict(base_options=BaseOptions(model_asset_path=_CFG['model']), num_hands=2)
    if mode == 'video':
        opts.update(running_mode=vision.RunningMode.VIDEO, min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    else:
        opts.update(running_mode=vision.RunningMode.IMAGE, min_hand_detection_confidence=0.3)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**opts))


def _zip(path):
    if path not in _CACHE:
        _CACHE[path] = zipfile.ZipFile(path)
    return _CACHE[path]


def _read_bytes(loc):
    kind = loc[0]
    if kind == 'file':
        return open(loc[1], 'rb').read()
    if kind == 'zip':
        return _zip(loc[1]).read(loc[2])
    if kind == 'tar':
        with open(loc[1], 'rb') as f:
            f.seek(loc[2])
            return f.read(loc[3])
    if kind == 'bytes':
        return loc[1]
    raise ValueError(f'cannot read {kind}')


def extract_video(path, *, target_fps, max_side):
    """Per-frame hands of one video, as the shipped extractor samples it."""
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / target_fps))
    w, h = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, max_side / max(w, h)) if max(w, h) > 0 else 1.0
    lm = _landmarker('video')
    frames, i = [], 0
    try:
        while cap.grab():
            if i % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                if scale < 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
                t = i / fps
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
                frames.append([round(t, 3), _hands(res)])
            i += 1
    finally:
        cap.release()
        lm.close()
    return {'w': w, 'h': h, 'fps': fps, 'frames': frames}


def extract_image(data, *, pad):
    """Hands in one still (bytes of any PIL-readable image), padded as the shipped extractor does."""
    import mediapipe as mp
    import numpy as np
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data)).convert('RGB')
    p = int(pad * max(im.size))
    im = ImageOps.expand(im, border=p, fill=(128, 128, 128))
    if 'image' not in _CACHE:
        _CACHE['image'] = _landmarker('image')
    res = _CACHE['image'].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(im))))
    return {'w': im.size[0], 'h': im.size[1], 'fps': None, 'frames': [[0.0, _hands(res)]]}


def convert_keypoints(loc, *, size):
    """A corpus's own 21-point landmarks as a one-frame record without handedness."""
    if loc[0] == 'csv21':
        rows = _zip(loc[1]).read(loc[2]).decode().strip().splitlines()[1:]
        pts = [[float(v) for v in r.split(',')[1:4]] for r in rows]
    else:
        pts = [[p['x'], p['y'], p['z']] for p in json.load(open(loc[1]))]
    hands = [[None, None, [_r4(c) for p in pts for c in p], None]] if len(pts) == 21 else []
    return {'w': size[0], 'h': size[1], 'fps': None, 'frames': [[0.0, hands]], 'assumed_size': True}


def process(item):
    """One item -> its record (the item's metadata plus w, h, fps, frames), or an error record."""
    rec = {k: v for k, v in item.items() if k != 'loc'}
    try:
        if item['kind'] == 'keypoints':
            rec.update(convert_keypoints(item['loc'], size=_CFG['keypoint_size']))
        elif item['kind'] == 'image':
            rec.update(extract_image(_read_bytes(item['loc']), pad=_CFG['pad']))
        elif item['loc'][0] == 'file':
            rec.update(extract_video(item['loc'][1], target_fps=_CFG['fps'], max_side=_CFG['max_side']))
        else:
            suffix = os.path.splitext(item['id'])[1] or '.mp4'
            with tempfile.NamedTemporaryFile(suffix=suffix, dir=_CFG['tmp']) as f:
                f.write(_read_bytes(item['loc']))
                f.flush()
                rec.update(extract_video(f.name, target_fps=_CFG['fps'], max_side=_CFG['max_side']))
    except Exception as e:  # one unreadable file must not stop a corpus
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def _resolve_parquet(items):
    """Replace ('parquet', path, row) locators by the image bytes (read once, in this process)."""
    tables = {}
    for it in items:
        if it['loc'][0] == 'parquet':
            import pyarrow.parquet as pq
            p = it['loc'][1]
            if p not in tables:
                tables[p] = pq.read_table(p, columns=['image']).column('image')
            it = dict(it, loc=('bytes', tables[p][it['loc'][2]].as_py()['bytes']))
        yield it


def _repair(path):
    """Cut a last line left half-written by an interrupted run, so appends start clean."""
    with open(path, 'rb+') as f:
        data = f.read()
        if data and not data.endswith(b'\n'):
            f.truncate(data.rfind(b'\n') + 1)


def _done_ids(out_dir):
    done = set()
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.endswith('.jsonl'):
                _repair(os.path.join(out_dir, name))
                for line in open(os.path.join(out_dir, name)):
                    try:
                        done.add(json.loads(line)['id'])
                    except (ValueError, KeyError):
                        pass  # a line cut by an interrupted run is redone
    return done


def _every_kth(xs, n):
    return xs[::max(1, len(xs) // n)][:n]


def _select(items, *, per_label, labels, per_signer=None):
    """Optional subsampling (deterministic, evenly spaced): at most ``per_label`` items per label,
    and at most ``per_signer`` per (signer, label, corpus split), which keeps every signer."""
    items = [it for it in items if not labels or it['label'] in labels]
    if per_signer:
        by = defaultdict(list)
        for it in items:
            by[(it['signer'], it['label'], it['split'])].append(it)
        items = [it for xs in by.values() for it in _every_kth(xs, per_signer)]
    if not per_label:
        return items
    by = defaultdict(list)
    for it in items:
        by[it['label']].append(it)
    return [it for xs in by.values() for it in _every_kth(xs, per_label)]


def run(source, *, raw, out, workers, min_avail_mb, per_label=None, labels=None, cfg, max_tasks_per_child=None, per_signer=None):
    out_dir = os.path.join(out, source)
    os.makedirs(out_dir, exist_ok=True)
    done = _done_ids(out_dir)
    todo = [it for it in _select(S.items(source, raw), per_label=per_label, labels=labels, per_signer=per_signer) if it['id'] not in done]
    print(f'[{source}] {len(todo)} to do ({len(done)} already done)', flush=True)
    if not todo:
        return
    files, n, errors, t0 = {}, 0, 0, time.time()
    pending = set()
    it = iter(_resolve_parquet(todo))
    pool = dict(initializer=_init, initargs=(cfg,))
    if max_tasks_per_child:  # recycle workers (MediaPipe and the video decoder grow over many clips)
        import multiprocessing
        pool.update(mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=max_tasks_per_child)
    with ProcessPoolExecutor(workers, **pool) as ex:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                avail = mem_available_mb()
                if avail is not None and avail < min_avail_mb and pending:
                    print(f'  memory low ({avail} MB available): waiting', flush=True)
                    break
                nxt = next(it, None)
                if nxt is None:
                    exhausted = True
                    break
                pending.add(ex.submit(process, nxt))
            if not pending:
                if not exhausted:
                    time.sleep(1)
                continue
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                rec = fu.result()
                errors += 'error' in rec
                lab = rec['label']
                if lab not in files:
                    files[lab] = open(os.path.join(out_dir, f'{lab}.jsonl'), 'a')
                files[lab].write(json.dumps(rec, separators=(',', ':')) + '\n')
                n += 1
                if n % 200 == 0:
                    rate = n / (time.time() - t0)
                    print(f'  {n}/{len(todo)} ({rate:.1f}/s, eta {(len(todo) - n) / rate / 60:.1f} min, '
                          f'{errors} errors, {mem_available_mb()} MB available)', flush=True)
    for f in files.values():
        f.close()
    print(f'[{source}] done: {n} items, {errors} errors, {time.time() - t0:.0f}s', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='the raw-data dir (holds one folder per corpus)')
    ap.add_argument('--out', required=True, help='landmarks dir; <out>/<source>/<label>.jsonl')
    ap.add_argument('--model', required=True, help='MediaPipe hand_landmarker.task (float16)')
    ap.add_argument('--sources', required=True, help=f'comma list of: {",".join(S.CATALOG)}')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--fps', type=float, default=12.0, help='target sampling rate for videos')
    ap.add_argument('--max-side', type=int, default=1280, help='downscale video frames larger than this')
    ap.add_argument('--pad', type=float, default=0.35, help='still-image padding, fraction of the longer side')
    ap.add_argument('--keypoint-size', default='640x480', help='frame size assumed for landmark-only corpora')
    ap.add_argument('--min-avail-mb', type=int, default=4096, help='do not start work below this MemAvailable')
    ap.add_argument('--per-label', type=int, help='at most this many items per label (evenly spaced)')
    ap.add_argument('--per-signer', type=int, help='at most this many items per (signer, label, split)')
    ap.add_argument('--labels', help='comma list: only these labels')
    ap.add_argument('--tmp', default=tempfile.gettempdir(), help='where archived videos are unpacked one at a time')
    ap.add_argument('--max-tasks-per-child', type=int, help='restart each worker after this many items')
    a = ap.parse_args(argv)
    cfg = dict(model=os.path.abspath(a.model), fps=a.fps, max_side=a.max_side, pad=a.pad, tmp=a.tmp,
               keypoint_size=tuple(map(int, a.keypoint_size.split('x'))))
    labels = set(a.labels.split(',')) if a.labels else None
    for source in a.sources.split(','):
        run(source, raw=a.raw, out=a.out, workers=a.workers, min_avail_mb=a.min_avail_mb,
            per_label=a.per_label, labels=labels, cfg=cfg, max_tasks_per_child=a.max_tasks_per_child,
            per_signer=a.per_signer)


if __name__ == '__main__':
    sys.exit(main())
