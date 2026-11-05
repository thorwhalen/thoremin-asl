/**
 * The ASL Chords take of the player's own recordings (the private `thoremin-recordings`
 * dataset): 17 annotated stretches where the player holds a chord letter, A to G with the right
 * hand, A to C again, then A to G with the left. The labels come from the video, the timing from
 * the app's own reading (the dataset's annotations say how).
 *
 * Measured with the shipped models (docs/recognizer.md): with letters AND words, 43 % of the
 * frames inside those stretches named the held letter and 45 % named a word (B read as HELLO,
 * C as THANK YOU, G as Q); with the letters-only vocabulary and the G/Q orientation rule, 97 %.
 * The stretches' edges were placed partly from the app's own live reading (the 'before' reader),
 * so the figure is in-sample on its boundaries; with up to 1 s trimmed off each edge
 * (`ASL_TRIM`) it holds at 97 % (docs/recognizer.md).
 *
 * The landmarks are the player's body and stay in that private repository, so this test reads
 * them at run time and is skipped without a checkout: `$THOREMIN_RECORDINGS_DIR`, or a
 * `thoremin-recordings/recordings` folder beside this checkout or any of its ancestors.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { createSignRecognizer, parseMlpModel, type RecognizerOptions } from '@thoremin/ext-asl/recognizer';
import staticJson from '@thoremin/ext-asl/models/static.json';
import motionJson from '@thoremin/ext-asl/models/motion.json';
import { letterTriad, signChordNode } from '@thoremin/ext-asl/nodes/sign_chord';
import { DEFAULT_ASL_DIAL, effectiveVocabulary } from '@thoremin/ext-asl/schema';
import type { NoteLetter, ScaleSpec } from '@thoremin/sdk/music/theory';

function recordingsDir(): string | null {
  if (process.env.THOREMIN_RECORDINGS_DIR) return existsSync(process.env.THOREMIN_RECORDINGS_DIR) ? process.env.THOREMIN_RECORDINGS_DIR : null;
  for (let dir = resolve(__dirname); ; dir = dirname(dir)) {
    const c = join(dirname(dir), 'thoremin-recordings', 'recordings');
    if (existsSync(c)) return c;
    if (dirname(dir) === dir) return null;
  }
}
const DIR = recordingsDir();
const TAKE = '2026-11-05T12-51-15';
/** Edges of each stretch left out: the hand is still settling. */
const TRIM = Number(process.env.ASL_TRIM ?? 0.25);

interface Letter { label: string; start: number; end: number }

function measure(options: RecognizerOptions) {
  const rows = gunzipSync(readFileSync(join(DIR!, TAKE, 'hands.ndjson.gz'))).toString('utf8').trim().split('\n').map((l) => JSON.parse(l));
  const letters: Letter[] = JSON.parse(readFileSync(join(DIR!, TAKE, 'annotations.json'), 'utf8')).events.filter((e: { kind: string }) => e.kind === 'asl-letter');
  const r = createSignRecognizer(parseMlpModel(staticJson), parseMlpModel(motionJson), options);
  let ok = 0;
  let words = 0;
  let n = 0;
  const longest = new Map<Letter, number>();
  const perLetter = new Map<string, [number, number]>();
  let run = { label: null as string | null, since: 0 };
  for (const row of rows) {
    const label = r.update(row.value, row.t).label;
    if (label !== run.label) run = { label, since: row.t };
    const iv = letters.find((e) => row.t >= e.start + TRIM && row.t < e.end - TRIM);
    if (!iv) continue;
    n++;
    if (label === iv.label) {
      ok++;
      longest.set(iv, Math.max(longest.get(iv) ?? 0, row.t - Math.max(run.since, iv.start)));
    }
    if (label && label.length > 1) words++;
    const pl = perLetter.get(iv.label) ?? [0, 0];
    perLetter.set(iv.label, [pl[0] + (label === iv.label ? 1 : 0), pl[1] + 1]);
  }
  return { accuracy: ok / n, words: words / n, readFor: letters.map((e) => longest.get(e) ?? 0), gFrames: perLetter.get('G') ?? [0, 0] };
}

/** The take replayed through the reader and sign-chord as ASL Chords wires them: from the first
 *  frame each annotated letter is named to the first frame its chord sounds. */
function chordLag() {
  const rows = gunzipSync(readFileSync(join(DIR!, TAKE, 'hands.ndjson.gz'))).toString('utf8').trim().split('\n').map((l) => JSON.parse(l));
  const letters: Letter[] = JSON.parse(readFileSync(join(DIR!, TAKE, 'annotations.json'), 'utf8')).events.filter((e: { kind: string }) => e.kind === 'asl-letter');
  const config = { enabled: true, chords: true };
  const dial = { ...DEFAULT_ASL_DIAL, ...config };
  const r = createSignRecognizer(parseMlpModel(staticJson), parseMlpModel(motionJson));
  r.setOptions({ minConfidence: dial.minConfidence, smoothing: dial.smoothing, vocabulary: effectiveVocabulary(dial) });
  const chord = signChordNode.make(signChordNode.params!.parse({}));
  const spec: ScaleSpec = { root: 2, type: 'major', octaves: 2, baseOctave: 3 };
  const triads = new Map(['A', 'B', 'C', 'D', 'E', 'F', 'G'].map((l) => [l, letterTriad(spec, l as NoteLetter).join()]));
  const frames = rows.map((row, i) => {
    const reading = r.update(row.value, row.t);
    const out = chord.process({ reading, spec, config, chordConfig: {}, octaveShift: 0 }, { tick: i, time: row.t, dt: i ? row.t - rows[i - 1].t : 0, resources: {} }) as { triad: number[] };
    return { t: row.t as number, label: reading.label, triad: out.triad.join() };
  });
  const lags: number[] = [];
  letters.forEach((iv, k) => {
    const from = k ? letters[k - 1]!.end : 0;
    const inside = frames.filter((f) => f.t >= from && f.t < iv.end);
    const named = inside.find((f) => f.label === iv.label);
    const sounds = inside.find((f) => f.triad === triads.get(iv.label));
    if (named && sounds) lags.push(Math.max(0, sounds.t - named.t));
  });
  lags.sort((a, b) => a - b);
  return { played: lags.length, median: lags[Math.floor(lags.length / 2)] ?? NaN, max: lags[lags.length - 1] ?? NaN };
}

describe.runIf(DIR !== null)("the player's own ASL Chords take", () => {
  it('letters only: the held letter is named in 9 frames of 10, never a word, and every letter plays', () => {
    const m = measure({ vocabulary: 'letters' });
    expect(m.accuracy).toBeGreaterThan(0.92);
    expect(m.words).toBe(0);
    // Each held letter is read without a break for longer than sign-chord's longest wait (0.2 s).
    for (const s of m.readFor) expect(s).toBeGreaterThanOrEqual(0.2);
  });

  it('ASL Chords: every held letter plays its chord, a median 0.1 s after the label appears', () => {
    const lag = chordLag();
    // With the fixed 0.2 s dwell this replaced: median 0.23 s, slowest 0.24 s. Now the reader's
    // confidence commits a clear letter in about three frames (sign-chord's letterEvidence).
    expect(lag.played).toBe(17);
    expect(lag.median).toBeLessThanOrEqual(0.12);
    expect(lag.max).toBeLessThanOrEqual(0.2);
  });

  it('letters and words (free play) still reads the words, and the orientation rule fixes G there too', () => {
    const m = measure({ vocabulary: 'all' });
    expect(m.words).toBeGreaterThan(0.3);
    expect(m.accuracy).toBeGreaterThan(0.45);
    // G: read as Q in more than half its frames without the rule (37 % G), 93 % with it.
    expect(m.gFrames[0] / m.gFrames[1]).toBeGreaterThan(0.85);
  });
});
