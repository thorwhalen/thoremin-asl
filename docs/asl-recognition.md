# ASL recognition in the browser: what was tried, what ships, how good it is

Date: 2026-10-12; models retrained on 2026-10-13 (moved to this repository the same day). Code: `src/lib/recognizer` (the recogniser), `src` (the thoremin extension), `training` (the pipeline). Raw data lives in `~/.local/share/thoremin-asl/` (on the training machine) and never in the repo.

## The question

Recognise the ASL fingerspelled alphabet and a few common signs (yes, no, hello, thank you, please, sorry, bye) live, in the browser, next to the MediaPipe hand model thoremin already runs, with a score per class every frame.

## Prior work found

The only earlier note is a general survey in the maintainer's notes (gesture taxonomies; Sign Language MNIST). Nothing ASL-specific existed in thoremin's issues, discussions or docs, and no fleet package does sign recognition.

## Candidates surveyed

| Candidate | Verdict |
|---|---|
| **MediaPipe hand landmarks + a small classifier** (the landmarks thoremin already computes) | **Chosen.** No second vision model, about 0.1 ms a frame for letters, and the classifier is a few kB of JSON. |
| Image classifiers on Hugging Face (ResNet-101, CLIP-vision and YOLO fingerspelling models) | Not adopted, and not benchmarked here. They are a second full-frame CNN per frame beside MediaPipe (tens to hundreds of MB), and trained on single-background photo sets. |
| Siruyy/asl-static-landmarks-v1 (HF) | Not reusable: "86 engineered features" with no code to reproduce them. |
| Sign Language MNIST | Not usable: 28×28 grayscale crops, too small for a hand detector. |
| Kaggle: Google ASL Fingerspelling (2023) and Isolated Sign Language Recognition (2023; 250 signs including yes, no, hello, thankyou, please, bye) | The best route to better motion signs: large MediaPipe-holistic landmark corpora, many signers. Not used yet: they need Kaggle credentials (none on the server) and tens of GB. |
| MediaPipe GestureRecognizer with a custom Model Maker head | The same idea as the chosen route, but it adds a second tasks-vision model. thoremin's HandLandmarker already yields the landmarks. |

## Data

Two rounds. The first (2026-10-12) trained on two MIT corpora and tested letters on one unlicensed photo set. The second (2026-10-13) surveyed every reachable corpus, recorded each one's licence, fixed the test sets **before** any training, and retrained both models. The licence and attribution of each source are in [`ATTRIBUTIONS.md`](../ATTRIBUTIONS.md). The rule: train only on permissively licensed data; data with no licence or a research-only one is test-only.

| Corpus | Licence | Role |
|---|---|---|
| PopSign ASL v1.0 (Georgia Tech / NTID / Google) | CC BY 4.0 | Words: train (32 signers), validation (8 signers), **test (8 other signers)**. 15 other PopSign signs as "no sign" examples (training signers only). |
| ASL-HG (10 signers), ASL Alphabet Gestures (3 signers) | CC BY 4.0 | Letters: train |
| om192006 keypoints, ASLNow! | MIT | Letters: train |
| StudioGalt mocap archive | CC0 | Letters, J, Z, some words: train (one signer, rendered avatar: low weight) |
| Aslense | MIT on its card; scraped (see below) | Letters, J, Z, words: train, minus 238 clips that are copies of test-only corpora |
| SignAlphaSet | CC BY 4.0 | **Test only, by choice**: the one permissive source no model sees, and the only held-out source for Z |
| Marxulia v03 | none stated | **Test only** (letters) |
| ASL Citizen | Microsoft research-only | **Test only** (words) |
| WLASL | C-UDA | **Test only** (words, J) |

**Aslense contains test-only data.** All 214 ASL Citizen clips of our seven words are in Aslense (208 byte-identical, 6 matched by file name), and 24 more Aslense clips are byte-identical to WLASL clips. The models shipped before 2026-10-13 were trained on them, so their ASL Citizen numbers were not held out. The current models never train on these 238 clips. Other Aslense word clips are named like WLASL downloads or dictionary-site videos, so its MIT label is doubtful; replacing it is the first item under "What would make it better".

YouTube refuses the training server's IP for every yt-dlp client, and the Kaggle corpora need credentials; neither was used.

## Letters

24 static letters (J and Z move). The `geo` features are 66 numbers: finger flexion (cosine at each joint), finger spreads, fingertip distances, thumb-to-knuckle distances, and the hand's orientation to the camera. Raw coordinates overfit the corpus they came from; geometric features carry across sources.

**Selection** used leave-one-source-out over the five training corpora (mean top-1 on the held-out corpus), never a test set:

| Recipe | Mean top-1 |
|---|---|
| First-round recipe (Aslense + ASLNow!, logistic C 0.1) | 0.729 |
| + ASL-HG + ASL Alphabet Gestures | 0.804 |
| + om192006 | 0.813 |
| + StudioGalt, C 0.03 | 0.819 |
| ±15° rotation augmentation | 0.817 |
| **MLP 64 hidden, L2 1.0 (shipped)** | **0.834** |

**Shipped:** a 66 → 64 → 24 MLP over the `geo` features, trained on Aslense letters, ASLNow!, ASL-HG, ASL Alphabet Gestures, om192006 and StudioGalt (at most 400 hands per source and letter, sources weighted equally; 33,717 hands). About 122 kB of JSON.

**Held-out results**, first-round model → shipped model, top-1 / top-3:

| Test set | First round | Shipped |
|---|---|---|
| Marxulia, the first round's sample (1,091 hands) | 0.625 / 0.826 | **0.738 / 0.918** |
| Marxulia, all (7,692 hands) | 0.609 / 0.811 | **0.712 / 0.903** |
| SignAlphaSet stills (9,361 hands) | 0.507 / 0.718 | **0.775 / 0.896** |
| SignAlphaSet letter videos (236 clips, vote), top-1 | 0.809 | **0.945** |

Per-letter recall on all of Marxulia, first round → shipped:

| Letter | Recall | Letter | Recall | Letter | Recall |
|---|---|---|---|---|---|
| A | .95 → .96 | I | .75 → .75 | R | .55 → .50 |
| **B** | **.68 → .47** | K | .40 → .40 | S | .50 → .76 |
| C | .57 → .64 | L | .68 → .84 | T | .29 → .46 |
| D | .91 → .89 | M | .61 → .76 | U | .16 → .44 |
| E | .75 → .94 | N | .60 → .57 | V | .55 → .64 |
| F | .70 → .90 | O | .76 → .84 | W | .37 → .67 |
| G | .75 → .76 | P | .63 → .67 | X | .46 → .74 |
| H | .81 → .89 | Q | .65 → .79 | Y | .49 → .82 |

On SignAlphaSet stills the big gains are C (0 → .58), D (.04 → .93), H (0 → .74), P (0 → .87), R (.09 → .76), S (.28 → 1.00), T (.01 → 1.00). Still failing there: G (0, read as Q), O (0), M (.44 → .01, read as S).

**Top confusions** (true > read, count):

- Marxulia, shipped: U>H 108, K>H 97, T>M 95, K>P 85, N>M 80, R>H 79, V>H 78, B>H 74. The model over-reads H on Marxulia, which is what costs B. (First round: W>M 134, K>P 91, T>M 89, U>R 89, B>M 85.)
- SignAlphaSet, shipped: G>Q 391, M>S 317, U>R 247, C>Q 208. (First round: C>Q 497, G>Q 434, H>I 409, T>M 360, S>E 359.)

The hard letters remain the closed fists (M, N, S, T: the thumb's place under the fingers, which landmarks see poorly) and the two-finger family (R, U, H, K, V). The chord letters A to G are among the most reliable, except G against Q (the same handshape, pointing sideways or down).

## Motion signs

J and Z are drawn in the air, and every one of the words is a movement, so a frame cannot read them. A second MLP (146 → 128 → 32 → 10) reads windows of the signing hand's last 0.6, 0.9, 1.3 and 1.8 s. Its inputs are the letter model's posterior at four times (the handshape over time), the wrist path and the index and pinky tip paths in palm lengths at eight times, the path length and the duration. Every window length is scored each frame, and the most confident sign wins if it reaches the threshold (0.95, carried in `motion.json`). Because it reads the letter model's outputs, **the two models ship as a pair**.

The "none" class is trained on windows from held-letter clips and on 848 PopSign clips of 15 other signs: without those negatives a third of held-letter clips fire a word as the hand moves into or out of the letter (R/U/N look like NO, S/T/M/A like YES, B/W/P like HELLO). Selection used PopSign validation signers and grouped 5-fold cross-validation (the only check for J, Z and sorry), scored as detection minus the held-letter false rate.

**Held-out results** at threshold 0.9 (a sign counts when the right sign is the most confident over the clip and reaches the threshold), first-round model → shipped:

| Test set | First round | Shipped |
|---|---|---|
| PopSign, 8 unseen signers (919 clips) | 0.838 | **0.948** |
| WLASL clips the first round never saw (36) | 0.944 | **0.972** |
| ASL Citizen test split (84) | 0.940 (trained on these clips) | 0.869 (held out) |
| SignAlphaSet signs (J, Z, 5 words; 60) | **0.500** | 0.433 |
| SignAlphaSet held letters with a false sign (240 clips) | **0.125** | 0.188 |

Per sign on PopSign's test signers: yes 148 → 156/158, no 128 → 148/153, hello 126 → 139/152, thank you 128 → 146/155, please 132 → 151/155, bye 108 → 131/146.

**Known weaknesses of the shipped pair**, all on held-out data:

- **Z is never recognised** (0/10 on SignAlphaSet; the first round was 0/10 too). Z does not transfer between the two small sources there are; it needs many signers.
- **J is weaker**: 5/14 (SignAlphaSet + WLASL) against the first round's 10/14.
- **HELLO signed as a wave reads as BYE** (SignAlphaSet 1/10, 9 read as BYE): that corpus's one signer greets with what PopSign's 32 signers sign as BYE.
- **False signs on held letters went up**, from 12.5% to 19% of clips (a clip counts if a false sign appears at any moment, including while the hand rises into view). A recognised motion sign stays on screen for 1 s (the latch).
- **SORRY rests on 6 Aslense clips** of doubtful provenance (ASL Citizen 8/12, SignAlphaSet 5/10). Trained without them, SORRY is never recognised.

## First round (2026-10-12), for the record

The first comparison chose the features: 5-fold cross-validation grouped by source clip on Aslense, with ASLNow! in every training fold, and Marxulia as the cross-source test. World or image landmarks reached 0.43 to 0.50 top-1 on Marxulia; `geo` features with ASLNow! and a logistic regression (C 0.1) reached 0.625 / 0.826, and shipped as a 66 × 24 linear softmax (34 kB). Its motion model (146 → 64 → 10, threshold 0.9) was trained on Aslense alone.

## Cost

Measured for the first-round models in Node on the (loaded) production server, per frame: 0.08 ms for the letters alone, 1.24 ms with the four motion windows, 4% of a 30 fps frame. The shipped models have a hidden layer each (64; 128 and 32), a few times the arithmetic, still well under a millisecond per window. They load lazily as separate chunks, with the extension's code, only when an ASL instrument or the `asl` dial is turned on (about 122 kB and 197 kB of JSON before compression).

## What would make it better, in order

1. Permissive data from many signers for Z, J and SORRY, to replace Aslense: the Kaggle corpora above (credentials needed), more PopSign signs, or recordings through the trainer.
2. Location: THANK YOU (chin) and HELLO (forehead) are told apart by movement alone today. The face landmarks thoremin already has would add where the hand is.
3. Personal calibration through the trainer (thoremin already enrols a player's own chord shapes): a few seconds per hard letter would fix a given player's M/N/S.
4. Fewer false signs on held letters: require a motion sign to win on two consecutive frames, or raise the threshold (0.98 trades a few points of detection for fewer false signs on validation).
5. A real live check with a webcam and a signer (thoremin#146).
