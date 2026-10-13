"""Fetch a per-signer subset of PopSign v1.0 clips (CC BY 4.0) for signs we do not recognise,
as ``none`` (negative) examples for the motion model: other signs are what a player's hands
do between the signs we name.

PopSign ships one tar per sign and split (``.../popsign_v1_0/game/train/<sign>.tar``, about
3 GB each). Rather than download whole tars, this walks a tar's headers with HTTP range
requests and downloads only the chosen members (``--per-signer`` clips per signer, evenly
spread), into ``<out>/<split>/<sign>/<file>.mp4``. Resumable: files already present are skipped.

usage: asl.py fetch-popsign --out RAW/popsign_v1_0_negatives --signs mom,dad,... [--split train] [--per-signer 2]
"""
import argparse
import os
import re
import sys
import urllib.request
from collections import defaultdict

BASE = 'https://signdata.cc.gatech.edu/data/popsign_v1_0'


def _range(url, start, length):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + length - 1}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def tar_index(url):
    """``[(name, data_offset, size)]`` of a remote (uncompressed, ustar) tar, from its headers."""
    out, off = [], 0
    while True:
        h = _range(url, off, 512)
        if len(h) < 512 or not h.strip(b'\0'):
            return out
        name = h[:100].rstrip(b'\0').decode()
        prefix = h[345:500].rstrip(b'\0').decode()
        size = int(h[124:136].rstrip(b'\0 ') or b'0', 8)
        typ = h[156:157]
        if typ in (b'0', b'\0') and name.endswith('.mp4'):
            out.append((f'{prefix}/{name}' if prefix else name, off + 512, size))
        off += 512 + (size + 511) // 512 * 512


def signer_of(name):
    return re.sub(r'^gtsignstudy', '', os.path.basename(name).split('-')[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--signs', required=True, help='comma list of PopSign sign names')
    ap.add_argument('--split', default='train')
    ap.add_argument('--kind', default='game')
    ap.add_argument('--per-signer', type=int, default=2)
    a = ap.parse_args(argv)
    for sign in a.signs.split(','):
        url = f'{BASE}/{a.kind}/{a.split}/{sign}.tar'
        try:
            idx = tar_index(url)
        except Exception as e:  # a sign missing from this split is skipped, not fatal
            print(f'{sign}: {e}', flush=True)
            continue
        by = defaultdict(list)
        for m in idx:
            by[signer_of(m[0])].append(m)
        pick = [m for ms in by.values() for m in ms[::max(1, len(ms) // a.per_signer)][:a.per_signer]]
        d = os.path.join(a.out, a.split, sign)
        os.makedirs(d, exist_ok=True)
        n = 0
        for name, off, size in pick:
            dst = os.path.join(d, os.path.basename(name))
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                continue
            data = _range(url, off, size)
            with open(dst + '.part', 'wb') as f:
                f.write(data)
            os.replace(dst + '.part', dst)
            n += 1
        print(f'{sign}: {len(idx)} clips, {len(by)} signers, fetched {n} (kept {len(pick)})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
