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
| SignAlphaSet's 10 J clips read as J (either vocabulary) | 0 / 10 | 10 / 10 |
| SignAlphaSet's 10 Z clips read as Z (either vocabulary) | 0 / 10 | 5 / 10 |
| SignAlphaSet's 24 held-letter clips (one per letter) read as J or Z at some point | 1 / 24 | 1 / 24 |
| thoremin's stability harness, ASL Chords (its `docs/testing/stability.md`), a real signer's letters in flow / with gaps: chords missed | 1/9, 3/8 | 0/9, 1/8 |
| the same: latency p90 | 3.1 s, 3.4 s | 0.27 s, 1.7 s |
| the same, with gaps: chord changes per second while a letter is held | 0.23 | 0.39 (worse: letters are read where words used to hide them) |
| the same, a choir conductor's hands (nobody signing): seconds of chord | 0 | 0 |

Two cautions. The edges of the player's 17 stretches were placed partly from the app's own live reading (the "before" reader), so the 97 % is in-sample on its boundaries; trimming 0.25, 0.5, 0.75 or 1 s off each edge gives 97 % every time (and "before" falls to 37 %), so the boundaries do not carry it. The J/Z thresholds were set on the same twenty SignAlphaSet clips they are scored on, so those two rates are an upper estimate until a second signer is measured.

The `asl` dial's `auto` follows its own "A..G play chords" toggle. Other instruments that steer music by signs (thoremin's Sign Map and Sign Wheel) set `vocabulary: 'letters'` in their seeds.

The player's take is private (tested by `test/player_recordings.test.ts` when it is checked out); the SignAlphaSet clips are committed (`test/fixtures/motion_letters.json.gz`, CC BY 4.0, `test/motion_letters.test.ts`). The J/Z thresholds were set on those same twenty clips, so treat the J/Z rates as an upper estimate until a second signer is measured.

## From a reading to a chord: how soon a letter plays

ASL Chords (`sign-chord`) and thoremin's Sign Map do not play a letter the moment it is named: a reading passes through letters on the way from one sign to the next, and a hand that is not signing is now and then read as a weak letter. Until 2026-11-05 a letter played once it had been named for 0.2 s, and the player heard the chord about 0.25 to 0.33 s after the label appeared (measured on the synth track of their ASL take: 0.21 s of dwell, then the pitch glide and the old chord's reverb).

Now a letter plays once the reader is sure enough of it for long enough (thoremin's `@thoremin/sdk/util/commit_gate`): its score above the reader's threshold (`floor` on the reading, the dial's `minConfidence`), summed over the seconds it is named, must reach 0.02 (0.1 s of a reading 0.2 over the threshold, 0.04 s of a sure one), or it must be named for 0.2 s, whichever comes first. A reading the reader keeps through a lost frame (`held`) times nothing: a letter being timed starts over, and the chord in force stays.

| Measured on | 0.2 s dwell | evidence |
|---|---|---|
| the player's 17 held chord letters: from the label to the chord, median (slowest) | 0.23 s (0.24 s) | 0.10 s (0.17 s) |
| the same: letters that play | 17 / 17 | 17 / 17 |
| a choir conductor's hands (nobody signing): seconds of chord | 0 | 0 |
| the same, with the hand lost for 2, 5 or 10 frames after one of its two weak C runs (0.47 to 0.56): cases that play a chord (for about 19 s) | 5 of 6 | 1 of 6 |
| thoremin's stability harness, the real signer in flow: latency median (p90) | 267 ms (333) | 167 ms (233) |
| the same with gaps: chord changes per second while a letter is held; chords missed | 0.39; 1/8 | 0; 0/8 |

The conductor's strongest weak C sums to 0.0145 over its six frames, against the 0.02 needed: a margin of 1.4, where the 0.2 s dwell it replaced cleared the same six frames (0.17 s) by 1.2. At `minConfidence` 0.40 (the dial allows it) both rules play that C.

## How good it is

See the model comparison in [`docs/asl-recognition.md`](asl-recognition.md). Both models were trained on public data (MIT-licensed corpora on Hugging Face) by `training/`, which also writes the parity fixture that pins this package's featurisation to the training code (`test/parity.test.ts`).

## Limits worth knowing

- Location is not used: the hand alone is read, not where it is relative to the face or body. THANK YOU (from the chin) and HELLO (from the forehead) are told apart by their movement, not by where they start.
- A relaxed hand that is not signing still looks like some letter; the confidence floor and smoothing keep that quiet, but do not remove it.
- The closed-fist letters (A, E, M, N, S, T) differ by where the thumb sits, which hand landmarks see poorly; they are the most confused.
