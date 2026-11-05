/**
 * The ASL Chords take of the player's own recordings (the private `thoremin-recordings`
 * dataset): 17 annotated stretches where the player holds a chord letter, A to G with the right
 * hand, A to C again, then A to G with the left. The labels come from the video, the timing from
 * the app's own reading (the dataset's annotations say how).
 *
 * Measured with the shipped models (docs/recognizer.md): with letters AND words, 43 % of the
 * frames inside those stretches named the held letter and 45 % named a word (B read as HELLO,
 * C as THANK YOU, G as Q); with the letters-only vocabulary and the G/Q orientation rule, 96 %.
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
const TRIM = 0.25;

interface Letter { label: string; start: number; end: number }

function measure(options: RecognizerOptions) {
  const rows = gunzipSync(readFileSync(join(DIR!, TAKE, 'hands.ndjson.gz'))).toString('utf8').trim().split('\n').map((l) => JSON.parse(l));
  const letters: Letter[] = JSON.parse(readFileSync(join(DIR!, TAKE, 'annotations.json'), 'utf8')).events.filter((e: { kind: string }) => e.kind === 'asl-letter');
  const r = createSignRecognizer(parseMlpModel(staticJson), parseMlpModel(motionJson), options);
  let ok = 0;
  let words = 0;
  let n = 0;
  const longest = new Map<Letter, number>();
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
  }
  return { accuracy: ok / n, words: words / n, readFor: letters.map((e) => longest.get(e) ?? 0) };
}

describe.runIf(DIR !== null)("the player's own ASL Chords take", () => {
  it('letters only: the held letter is named in 9 frames of 10, never a word, and every letter plays', () => {
    const m = measure({ vocabulary: 'letters' });
    expect(m.accuracy).toBeGreaterThan(0.92);
    expect(m.words).toBe(0);
    // ASL Chords plays a letter read for 0.2 s (sign-chord's letterDwellS).
    for (const s of m.readFor) expect(s).toBeGreaterThanOrEqual(0.2);
  });

  it('letters and words (free play) still reads the words, and the orientation rule fixes G there too', () => {
    const m = measure({ vocabulary: 'all' });
    expect(m.words).toBeGreaterThan(0.3);
    expect(m.accuracy).toBeGreaterThan(0.5);
  });
});
