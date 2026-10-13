"""One entry point for the ASL training pipeline: ``python -I asl.py <command> [args]``.

Run with ``python -I`` (isolated: no user site, no PYTHONPATH, no script-dir import) because
the pipeline reads downloaded, untrusted files; this launcher then puts its own directory,
and only that, on ``sys.path``. Commands: extract, protocol, train-static, train-motion,
evaluate, handedness, parity-fixture, replay-fixture, fetch-popsign (each is the module of that name; ``asl.py <command> -h`` for its options).
"""
import importlib
import os
import sys

COMMANDS = {'extract': 'extract', 'protocol': 'protocol', 'train-static': 'train_static',
            'train-motion': 'train_motion', 'evaluate': 'evaluate', 'parity-fixture': 'make_parity_fixture', 'replay-fixture': 'make_replay_fixture',
            'handedness': 'handedness', 'fetch-popsign': 'fetch_popsign_subset'}

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f'usage: asl.py {{{",".join(COMMANDS)}}} [args]')
    mod = importlib.import_module(COMMANDS[sys.argv[1]])
    sys.argv = [f'asl.py {sys.argv[1]}'] + sys.argv[2:]
    sys.exit(mod.main())
