"""The catalogue of raw ASL corpora: where each lives under the raw-data dir, its licence,
and how to enumerate its items (one clip or one image each) with label, signer and split.

An item is a plain dict::

    {'source', 'id', 'label', 'signer', 'split', 'kind', 'loc'}

``kind`` is ``'video'``, ``'image'`` or ``'keypoints'`` (landmarks shipped by the corpus,
no MediaPipe run needed). ``loc`` says how to read the bytes:

    ('file', path) | ('zip', zip_path, member) | ('tar', tar_path, offset, size)
    | ('parquet', path, row) | ('csv21', zip_path, member) | ('json21', path)

``signer`` is ``None`` when the corpus does not say; ``split`` is the corpus's own split
(``None`` when it has none). Our train/val/test assignment is made later, by
``protocol.py``; the corpus split is only an input to it.

Labels are lower case: ``'a'``..``'z'`` and ``'yes' 'no' 'hello' 'thankyou' 'please'
'sorry' 'bye'``; anything else keeps its own lower-case name (``'goodbye'``, ``'0'``).
"""
import csv
import json
import os
import re
import tarfile
import zipfile

LICENCES = {
    'aslense': ('MIT (HF card; provenance weak: the corpus is scraped)', 'train'),
    'aslnow': ('MIT', 'train'),
    'popsign': ('CC BY 4.0', 'train'),
    'popsign_negatives': ('CC BY 4.0', 'train'),
    'signalphaset_dynamic': ('CC BY 4.0', 'train'),
    'signalphaset_static': ('CC BY 4.0', 'train'),
    'asl_hg': ('CC BY 4.0', 'train'),
    'american_hand_signs': ('CC BY 4.0', 'train'),
    'asl_alphabet_gestures': ('CC BY 4.0', 'train'),
    'om192006': ('MIT', 'train'),
    'studiogalt': ('CC0 1.0', 'train'),
    'marxulia': ('none stated', 'test-only'),
    'asl_citizen': ('Microsoft Research licence, non-commercial research only', 'test-only'),
    'wlasl': ('C-UDA, academic/computational use only', 'test-only'),
}

WORD_ALIASES = {'thank you': 'thankyou', 'thank_you': 'thankyou'}


def norm_label(s):
    """Corpus gloss -> our label (lower case, 'thank you' -> 'thankyou')."""
    s = s.strip().lower()
    return WORD_ALIASES.get(s, s.replace(' ', ''))


def _zip_members(path, pred):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if pred(n)]


def aslense(raw):
    """Aslense videos fetched as <raw>/aslense/videos/<gloss>/<file>, originals only."""
    root = os.path.join(raw, 'aslense', 'videos')
    for gloss in sorted(os.listdir(root)):
        for name in sorted(os.listdir(os.path.join(root, gloss))):
            if '_aug_' in name:
                continue
            yield dict(source='aslense', id=f'{gloss}/{name}', label=gloss, signer=None, split=None,
                       kind='video', loc=('file', os.path.join(root, gloss, name)))


def popsign(raw):
    """PopSign v1.0 per-sign tars: <raw>/popsign_v1_0/{game,non-game}/{train,val,test}/<sign>.tar.
    Signer = the participant prefix of the file name (``4a.8036``)."""
    root = os.path.join(raw, 'popsign_v1_0')
    for kind in ('game', 'non-game'):
        for split in ('train', 'val', 'test'):
            d = os.path.join(root, kind, split)
            if not os.path.isdir(d):
                continue
            for tname in sorted(os.listdir(d)):
                if not tname.endswith('.tar'):
                    continue
                tp = os.path.join(d, tname)
                with tarfile.open(tp) as tf:
                    for m in tf.getmembers():
                        if not (m.isfile() and m.name.endswith('.mp4')):
                            continue
                        base = os.path.basename(m.name)
                        signer = re.sub(r'^gtsignstudy', '', base.split('-')[0])
                        yield dict(source='popsign', id=f'{kind}/{split}/{base}', label=norm_label(tname[:-4]),
                                   signer=signer, split=split, kind='video', loc=('tar', tp, m.offset_data, m.size),
                                   subset=kind)


def popsign_negatives(raw):
    """PopSign clips of signs we do not recognise (fetched by fetch_popsign_subset.py into
    <raw>/popsign_v1_0_negatives/<split>/<sign>/), used as ``none``. Label = the sign."""
    root = os.path.join(raw, 'popsign_v1_0_negatives')
    if not os.path.isdir(root):
        return
    for split in sorted(os.listdir(root)):
        for sign in sorted(os.listdir(os.path.join(root, split))):
            for name in sorted(os.listdir(os.path.join(root, split, sign))):
                if name.endswith('.mp4'):
                    signer = re.sub(r'^gtsignstudy', '', name.split('-')[0])
                    yield dict(source='popsign_negatives', id=f'{split}/{sign}/{name}', label=norm_label(sign),
                               signer=signer, split=split, kind='video', loc=('file', os.path.join(root, split, sign, name)))


def signalphaset_dynamic(raw):
    """SignAlphaSet ASL_dynamic.zip: <CLASS>/<CLASS>_clipN.avi (the jpg frame folders are skipped)."""
    zp = os.path.join(raw, 'mendeley_signalphaset', 'ASL_dynamic.zip')
    for n in sorted(_zip_members(zp, lambda n: n.lower().endswith('.avi'))):
        cls = n.split('/')[1]
        yield dict(source='signalphaset_dynamic', id=os.path.basename(n), label=norm_label(cls), signer=None,
                   split=None, kind='video', loc=('zip', zp, n))


def _image_zip(source, zp, pat, label_of, signer_of=lambda n: None):
    for n in sorted(_zip_members(zp, lambda n: re.search(pat, n) is not None)):
        lab = label_of(n)
        if lab is None:
            continue
        yield dict(source=source, id=n, label=lab, signer=signer_of(n), split=None, kind='image', loc=('zip', zp, n))


def _letter_folder(n):
    """The letter from a ``.../<L>/<file>.jpg`` member name, or None for any other class."""
    d = n.split('/')[-2]
    return d.lower() if len(d) == 1 and d.isalpha() else None


def signalphaset_static(raw):
    zp = os.path.join(raw, 'mendeley_signalphaset', 'SignAlphaSet.zip')
    return _image_zip('signalphaset_static', zp, r'\.jpg$', _letter_folder)


def asl_hg(raw):
    """ASL-HG raw images: asl_dataset/<L>/P<k>_<L>_<i>.jpg, 10 signers P1..P10."""
    zp = os.path.join(raw, 'mendeley_asl_hg', 'ASL_Raw_Images.zip')
    return _image_zip('asl_hg', zp, r'\.jpg$', _letter_folder, lambda n: os.path.basename(n).split('_')[0])


def american_hand_signs(raw):
    zp = os.path.join(raw, 'mendeley_american_hand_signs', 'DataZip.zip')
    return _image_zip('american_hand_signs', zp, r'\.jpg$', _letter_folder)


def asl_alphabet_gestures(raw):
    """Images of ASL Alphabet Gestures: only Type_01 (raw photos); the other types are
    skeleton renderings of the same images."""
    zp = os.path.join(raw, 'mendeley_asl_alphabet_gestures', 'Root.zip')
    return _image_zip('asl_alphabet_gestures', zp, r'Type_01_\(Raw_Gesture\)/.*\.jpg$', _letter_folder)


def om192006(raw):
    """om192006 MediaPipe keypoints: media_pipe_keypoints_dataset/<CLASS>/<CLASS>_i.csv (21 rows x,y,z)."""
    zp = os.path.join(raw, 'om192006_keypoints', 'media_pipe_keypoints_dataset.zip')
    for n in sorted(_zip_members(zp, lambda n: n.endswith('.csv') and not n.startswith('__MACOSX'))):
        lab = _letter_folder(n)
        if lab is not None:
            yield dict(source='om192006', id=n, label=lab, signer=None, split=None, kind='keypoints', loc=('csv21', zp, n))


def aslnow(raw):
    """ASLNow!: <raw>/aslnow/<L>/*.json, 21 {x, y, z} points, no handedness."""
    root = os.path.join(raw, 'aslnow')
    for L in sorted(os.listdir(root)):
        d = os.path.join(root, L)
        if not (os.path.isdir(d) and len(L) == 1):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith('.json'):
                yield dict(source='aslnow', id=f'{L}/{name}', label=L.lower(), signer=None, split=None,
                           kind='keypoints', loc=('json21', os.path.join(d, name)))


def studiogalt(raw):
    """StudioGalt mocap renders: Letters/<L>.mp4 and dictionary/*/SG ASL <Gloss> ... .mp4.
    'Check Please' is a different sign and is skipped."""
    root = os.path.join(raw, 'studiogalt_cc0')
    for name in sorted(os.listdir(os.path.join(root, 'Letters'))):
        if name.endswith('.mp4'):
            yield dict(source='studiogalt', id=f'Letters/{name}', label=name[:-4].lower(), signer='studiogalt-avatar',
                       split=None, kind='video', loc=('file', os.path.join(root, 'Letters', name)))
    for d in sorted(os.listdir(os.path.join(root, 'dictionary'))):
        for name in sorted(os.listdir(os.path.join(root, 'dictionary', d))):
            m = re.match(r'SG ASL (Bye|Hello|Please)\b', name)
            if m and 'Check' not in name:
                yield dict(source='studiogalt', id=f'dictionary/{name}', label=norm_label(m.group(1)),
                           signer='studiogalt-avatar', split=None, kind='video',
                           loc=('file', os.path.join(root, 'dictionary', d, name)))


def marxulia(raw):
    """Marxulia v03 parquet rows (image bytes, int label 0..25). ``sample60`` marks the rows the
    first pipeline's (since removed) image extractor sampled (60 per letter, seed 0), for comparability."""
    import random
    import pyarrow.parquet as pq
    p = os.path.join(raw, 'marxulia_v03', 'data', 'train-00000-of-00001.parquet')
    labels = pq.read_table(p, columns=['label']).column('label').to_pylist()
    random.seed(0)
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    pick = {i for l, ix in by.items() for i in random.sample(ix, min(60, len(ix)))}
    for i, l in enumerate(labels):
        yield dict(source='marxulia', id=str(i), label=chr(ord('a') + l), signer=None, split=None,
                   kind='image', loc=('parquet', p, i), sample60=i in pick)


def asl_citizen(raw):
    """ASL Citizen clips of our words, with participant and split from the official CSVs."""
    root = os.path.join(raw, 'asl_citizen', 'ASL_Citizen')
    meta = {}
    for split in ('train', 'val', 'test'):
        for r in csv.DictReader(open(os.path.join(root, 'splits', f'{split}.csv'))):
            meta[r['Video file']] = (r['Participant ID'], split, r['Gloss'])
    for name in sorted(os.listdir(os.path.join(root, 'videos'))):
        signer, split, gloss = meta.get(name, (None, None, name.rsplit('-', 1)[-1][:-4].replace('seed', '')))
        yield dict(source='asl_citizen', id=name, label=norm_label(gloss), signer=signer, split=split,
                   kind='video', loc=('file', os.path.join(root, 'videos', name)))


def wlasl(raw):
    """The Voxel51 WLASL mirror clips of our glosses (our_glosses.json lists them)."""
    root = os.path.join(raw, 'wlasl_voxel51')
    for r in json.load(open(os.path.join(root, 'our_glosses.json'))):
        yield dict(source='wlasl', id=r['filepath'], label=norm_label(r['gloss']), signer=None, split=None,
                   kind='video', loc=('file', os.path.join(root, r['filepath'])))


CATALOG = {f.__name__: f for f in (aslense, popsign, popsign_negatives, signalphaset_dynamic, signalphaset_static, asl_hg,
                                   american_hand_signs, asl_alphabet_gestures, om192006, aslnow, studiogalt,
                                   marxulia, asl_citizen, wlasl)}


def items(source, raw):
    """Every item of ``source`` (a CATALOG key) under the raw-data dir ``raw``."""
    return CATALOG[source](raw)
