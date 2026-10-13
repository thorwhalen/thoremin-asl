# training/: the ASL training pipeline

Trains the two models the recogniser ships (`src/lib/recognizer/models/static.json`: the 24 static letters; `motion.json`: J, Z and seven words), measures them on held-out cross-source test sets, and writes the test fixtures that pin the TS featurisation to this code. The models use the recogniser's featurisation (`geo` letter features, `asl-motion/1` windows) and the `thoremin-asl-mlp/1` format, so an exported model drops into `src/lib/recognizer/models/` unchanged. The numbers, and the reasoning behind every choice, are in [`docs/asl-recognition.md`](../docs/asl-recognition.md).

Data never lives in this repository. Raw corpora, landmarks, models and results live in the app-data dir (`~/.local/share/thoremin-asl/`: `raw/`, `landmarks/`, `models/`, `eval/`); `SOURCES.md` there has every corpus's licence and class (TRAIN or TEST-ONLY). Only permissively licensed data is trained on; the rest is test-only.

Run from this directory. `-I` (isolated) matters: the pipeline reads downloaded, untrusted files, and `asl.py` puts only its own directory on `sys.path`.

```bash
uv venv --python 3.11 venv && uv pip install --python venv/bin/python -r requirements.txt
D=~/.local/share/thoremin-asl
PY="venv/bin/python -I asl.py"
curl -L -o $D/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# 1. Landmarks for every corpus (parallel, resumable, pauses when memory is low)
$PY extract --raw $D/raw --out $D/landmarks --model $D/models/hand_landmarker.task \
  --sources aslense,studiogalt,signalphaset_dynamic,asl_citizen,wlasl,aslnow,om192006,marxulia,asl_hg,popsign
$PY extract ... --sources signalphaset_static --per-label 500
$PY extract ... --sources asl_alphabet_gestures --per-label 600
$PY fetch-popsign --out $D/raw/popsign_v1_0_negatives --per-signer 2 \
  --signs mom,dad,water,drink,milk,cat,dog,sleep,happy,sad,book,car,wait,where,who
$PY extract ... --sources popsign_negatives --workers 6 --max-tasks-per-child 40
# 2. The protocol, fixed before any training: who trains, who selects, who tests
$PY protocol --raw $D/raw --lm $D/landmarks --out $D/eval
# 3. Letters: leave-one-source-out selection, then the export (the shipped recipe)
$PY train-static --lm $D/landmarks --eval $D/eval --models logreg_c0.1,mlp64_a1 --loso
$PY train-static --lm $D/landmarks --eval $D/eval --models mlp64_a1 \
  --sources aslense,aslnow,asl_hg,asl_alphabet_gestures,om192006,studiogalt --export $D/models/candidates/static.json
# 4. Motion, on top of that letter model (PopSign validation signers + grouped CV for selection)
$PY train-motion --lm $D/landmarks --eval $D/eval --static $D/models/candidates/static.json \
  --aug 10 --hidden 128,32 --alpha 0.03 --threshold 0.95 --cv
$PY train-motion ... --export $D/models/candidates/motion.json        # the same flags, without --cv
# 5. Test: the shipped pair and the candidate pair on the same sets, then the comparison
$PY evaluate --lm $D/landmarks --eval $D/eval --static ../src/lib/recognizer/models/static.json \
  --motion ../src/lib/recognizer/models/motion.json --name baseline
$PY evaluate --lm $D/landmarks --eval $D/eval --static $D/models/candidates/static.json \
  --motion $D/models/candidates/motion.json --name candidate
$PY evaluate --compare $D/eval/results-baseline.json $D/eval/results-candidate.json --threshold 0.9
# 6. Adopting a candidate: copy both files into ../src/lib/recognizer/models/, then the fixtures
$PY parity-fixture --lm $D/landmarks --static ../src/lib/recognizer/models/static.json \
  --motion ../src/lib/recognizer/models/motion.json --out ../test/fixtures/parity.json
$PY replay-fixture --lm $D/landmarks --out ../test/fixtures/clips.json
```

The two models are a pair (the motion model reads the letter model's outputs): adopt both or neither. After adopting, run the plugin's tests inside thoremin (see the repository README): the parity test fails when the two featurisers drift apart, and the replay test when a fixture clip is no longer read as its sign (re-pick it in `make_replay_fixture.py`). The fixtures are committed, so they carry only permissively licensed clips (listed in `ATTRIBUTIONS.md`).

| File | Does |
|---|---|
| `asl.py` | The one entry point (`python -I asl.py <command>`). |
| `sources.py` | The corpus catalogue: where each lives under `raw/`, its licence and class, and how to list its clips or images with label, signer and split. |
| `extract.py` | MediaPipe hand landmarks for every item of a corpus, in parallel, into `landmarks/<source>/<label>.jsonl`. |
| `fetch_popsign_subset.py` | A per-signer subset of PopSign v1.0 clips of other signs, as "no sign" examples. |
| `protocol.py` | The evaluation protocol as data: `eval/manifest.csv` (train / val / test / excluded per item and task) and `eval/duplicates.csv` (Aslense clips that are copies of test-only clips). |
| `asl_common.py` | Vocabulary, record loading, the letter featurisers (mirror `src/lib/recognizer/features.ts`), the model export and its numpy forward pass. |
| `motion_common.py` | The motion-window featuriser (mirrors `src/lib/recognizer/motion.ts`) and a replay of the recogniser's buffer and window scoring. |
| `handedness.py` | Geometry to MediaPipe handedness, for corpora shipped as bare landmarks (ASLNow!, om192006). |
| `train_static.py` | The letter model: leave-one-source-out selection, export. |
| `train_motion.py` | The motion model: PopSign-val selection, grouped CV, augmentation options, export. |
| `evaluate.py` | A model pair on every test set, as the browser reads them; `--compare` prints the tables. |
| `make_parity_fixture.py` | `test/fixtures/parity.json`: real hands and a motion window with the features and probabilities this code computes. |
| `make_replay_fixture.py` | `test/fixtures/clips.json`: a few real clips for the recogniser's end-to-end replay test. |
