# The recogniser (`@thoremin/ext-asl/recognizer`)

American Sign Language from hand landmarks: the fingerspelled alphabet and a few common words, as a score per sign, every frame. Pure TypeScript (no DAG, no DOM, no ML runtime); give it MediaPipe hand landmarks, get a reading.

```ts
import { createSignRecognizer, parseMlpModel } from '@thoremin/ext-asl/recognizer';
import staticJson from '@thoremin/ext-asl/models/static.json';
import motionJson from '@thoremin/ext-asl/models/motion.json';

const reader = createSignRecognizer(parseMlpModel(staticJson), parseMlpModel(motionJson));
const r = reader.update({ width, height, hands }, tSeconds); // hands: MediaPipe HandLandmarker, keypoints in pixels
r.label;   // 'F', 'thankyou', or null
r.scores;  // one 0..1 score per sign in SIGN_IDS order (A..Z, then the words), summing to 1
```

## What it recognises

- **24 held letters** (all but J and Z), from one frame's handshape: a small classifier over 66 geometric features (finger flexion, spreads, fingertip distances, hand orientation; `features.ts`).
- **Movement signs**: J and Z (drawn in the air) and YES, NO, HELLO, THANK YOU, PLEASE, SORRY, BYE. Every one of these words is a movement, so a single frame cannot tell them apart from a letter. A second classifier reads a window of the last 0.6 to 1.8 s: the handshape over time (the letter model's posterior), the wrist path and two fingertip paths (`motion.ts`).

The two fuse into one distribution (`recognizer.ts`), with time-constant smoothing, a confidence floor, hysteresis between signs, and a short latch that keeps a word on screen after its movement ends.

Three options shape the reading:

- **`vocabulary`**: `'all'` (letters and words) or `'letters'` (A to Z only; a word is never named). What the motion model gave the words goes back to the letters as far as the hand clearly holds a letter's shape (the letter model's top probability from 0.6 to 0.8); the rest is an unnamed competitor, so a hand that moves like a word with no clear letter in it names nothing (a conductor's curved hand: weak C, strong THANK YOU). An instrument that maps letters to music reads letters only: a hand moving from one letter to the next makes exactly the movements the words are (B read as HELLO, C as THANK YOU). The `asl` dial's `vocabulary` is `auto` by default: letters while the letters play chords, letters and words in free play.
- **`orientation`** (on by default): letters that share a handshape and differ only in where it points are decided by the direction (`orientation.ts`). Today that is G (index level) and Q (index down): the model read a real player's level G as Q in half its frames.
- **`motionLetters`**: where J and Z come from. `'trajectory'` (the default) reads them from the path the hand draws (`trajectory.ts`): the pinky hand twisting into a hook is J, the index hand drawing across, back and across is Z, checked as geometry over the last 1.8 s. `'model'` uses the motion classifier's J and Z, `'both'` either.

Measured on two real sources the models never trained on:

| | before | after |
|---|---|---|
| A player's own ASL Chords take (17 held chord letters, A to G with each hand): frames naming the held letter, in ASL Chords | 43 % (45 % named a word) | 97 % (no word) |
| the same, in free play (letters and words) | 43 % | 52 % (G fixed) |
| SignAlphaSet's 10 J clips read as J (either vocabulary) | 1 / 10 | 10 / 10 |
| SignAlphaSet's 10 Z clips read as Z (either vocabulary) | 0 / 10 | 6 / 10 |
| 51 held-letter clips (two of each of 24 letters, three more of I) read as J or Z at some point | 1 / 51 | 1 / 51 |
| thoremin's stability harness, ASL Chords (`docs/testing/stability.md` there): a real signer, letters in flow / with gaps | missed 1/9, 3/8; latency p90 3.1 s, 3.4 s | missed 0/9, 1/8; latency p90 0.27 s, 1.7 s |
| the same, a choir conductor's hands (nobody signing) | 0 s of chord | 0 s of chord |

The player's take is private (tested by `test/player_recordings.test.ts` when it is checked out); the SignAlphaSet clips are committed (`test/fixtures/motion_letters.json.gz`, CC BY 4.0, `test/motion_letters.test.ts`). The J/Z thresholds were set on those same twenty clips, so treat the J/Z rates as an upper estimate until a second signer is measured.

## How good it is

See the model comparison in [`docs/asl-recognition.md`](asl-recognition.md). Both models were trained on public data (MIT-licensed corpora on Hugging Face) by `training/`, which also writes the parity fixture that pins this package's featurisation to the training code (`test/parity.test.ts`).

## Limits worth knowing

- Location is not used: the hand alone is read, not where it is relative to the face or body. THANK YOU (from the chin) and HELLO (from the forehead) are told apart by their movement, not by where they start.
- A relaxed hand that is not signing still looks like some letter; the confidence floor and smoothing keep that quiet, but do not remove it.
- The closed-fist letters (A, E, M, N, S, T) differ by where the thumb sits, which hand landmarks see poorly; they are the most confused.
