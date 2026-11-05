/**
 * The sign-chord node: A..G signed → the key's chord on stable voice ids; held between
 * letters by default, released when the hand leaves; silent unless the dial says chords.
 */
import { describe, expect, it } from 'vitest';
import { signChordNode, SIGN_CHORD_VOICE_ID_BASE, MAX_SIGN_CHORD_VOICES } from '@thoremin/ext-asl/nodes/sign_chord';
import { emptyScores, SIGN_IDS, type SignId, type SignScores } from '@thoremin/ext-asl/recognizer';
import { chordName, type ScaleSpec } from '@thoremin/sdk/music/theory';
import type { SynthParams } from '@thoremin/sdk/nodes/domain';

const D_MAJOR: ScaleSpec = { root: 2, type: 'major', octaves: 2, baseOctave: 3 };
const reading = (label: SignId | null, present = true): SignScores => ({ ...emptyScores(), present, label, kind: label ? 'static' : null });
const ctx = (i: number) => ({ tick: i, time: i / 30, dt: 1 / 30, resources: {} });

/** One reading per camera frame (30 fps). The dwell is off unless a test turns it on. */
function run(seq: (SignScores | undefined)[], config: Record<string, unknown> = { enabled: true, chords: true }, chordConfig: Record<string, unknown> = {}, letterDwellS = 0) {
  const node = signChordNode.make(signChordNode.params!.parse({ letterDwellS }));
  return seq.map((r, i) => node.process({ reading: r, spec: D_MAJOR, config, chordConfig, octaveShift: 0 }, ctx(i)) as { params: SynthParams; triad: number[]; captions: string[] });
}

describe('sign-chord', () => {
  it('F in D major plays F#m on four stable voice ids', () => {
    const [out] = run([reading('F')]);
    expect(chordName(out!.triad)).toBe('F#m');
    expect(out!.params.voices.map((v) => v.id)).toEqual(Array.from({ length: MAX_SIGN_CHORD_VOICES }, (_, i) => SIGN_CHORD_VOICE_ID_BASE + i));
    expect(out!.params.voices.some((v) => v.present)).toBe(true);
  });

  it('holds the chord through other letters, releases when the hand leaves', () => {
    const outs = run([reading('A'), reading('H'), reading(null), reading(null, false)]);
    expect(outs.map((o) => chordName(o.triad) || '-')).toEqual(['A', 'A', 'A', '-']);
    expect(outs[3]!.params.voices.every((v) => !v.present)).toBe(true);
  });

  it('with the dwell, a letter read for a moment plays nothing; one read for 0.2 s plays', () => {
    const on = { enabled: true, chords: true };
    // Four frames of C (a misread, or a letter passed on the way), then another sign: no chord.
    const passing = run([...Array(4).fill(reading('C')), ...Array(20).fill(reading('H'))], on, {}, 0.2);
    expect(passing.every((o) => o.triad.length === 0)).toBe(true);
    // F held: nothing for the first 0.2 s, then F#m, and it stays.
    const held = run(Array(12).fill(reading('F')), on, {}, 0.2);
    expect(held.slice(0, 6).every((o) => o.triad.length === 0)).toBe(true);
    expect(held.slice(6).every((o) => chordName(o.triad) === 'F#m')).toBe(true);
    // A chord in force is not interrupted by a passing letter.
    const through = run([...Array(10).fill(reading('A')), reading('E'), reading('E'), ...Array(5).fill(reading('A'))], on, {}, 0.2);
    expect(through.slice(6).map((o) => chordName(o.triad))).toEqual(Array(11).fill('A'));
  });

  it('a letter the reader is sure of plays sooner; a marginal one waits for the dwell', () => {
    const on = { enabled: true, chords: true };
    // A reading as the recogniser gives it: the letter's smoothed score and the reader's threshold.
    const scored = (label: SignId, score: number, extra: Partial<SignScores> = { floor: 0.45 }): SignScores => ({
      ...reading(label),
      scores: SIGN_IDS.map((id) => (id === label ? score : 0)),
      ...extra,
    });
    const firstChord = (seq: SignScores[], evidence = 0.02) => {
      const n = signChordNode.make(signChordNode.params!.parse({ letterDwellS: 0.2, letterEvidence: evidence }));
      return seq.findIndex((r, i) => (n.process({ reading: r, spec: D_MAJOR, config: on, chordConfig: {}, octaveShift: 0 }, ctx(i)) as { triad: number[] }).triad.length > 0);
    };
    const hold = (r: SignScores, n = 30) => Array<SignScores>(n).fill(r);
    // 0.95: 0.5 over the threshold, 0.0167 a frame: the second frame after the first reaches 0.02.
    expect(firstChord(hold(scored('F', 0.95)))).toBe(2);
    // 0.65: 0.2 over, 0.0067 a frame: three frames (0.1 s).
    expect(firstChord(hold(scored('F', 0.65)))).toBe(3);
    // Just over the threshold (a conductor's hand read as a weak C): the full 0.2 s dwell.
    expect(firstChord(hold(scored('F', 0.48)))).toBe(6);
    // Evidence off, or a reading that gives no threshold: the dwell alone, whatever the score.
    expect(firstChord(hold(scored('F', 0.95)), 0)).toBe(6);
    expect(firstChord(hold(scored('F', 0.95, {})))).toBe(6);
    // A reading kept through lost frames (`held`) times nothing: one sure frame, then the hand is
    // lost and the reader repeats it, and nothing plays (a conductor's hand read as a weak C, then
    // lost, played the C on the kept reading alone). When the hand is back, the timing starts over.
    const kept = scored('F', 0.95, { floor: 0.45, held: true });
    expect(firstChord([scored('F', 0.95), ...hold(kept, 12)])).toBe(-1);
    expect(firstChord([scored('F', 0.95), ...hold(kept, 3), ...hold(scored('F', 0.95))])).toBe(6);
    // The chord in force is not dropped by kept readings.
    const n = signChordNode.make(signChordNode.params!.parse({}));
    const outs = [...hold(scored('F', 0.95), 6), ...hold(kept, 5)].map((r, i) => n.process({ reading: r, spec: D_MAJOR, config: on, chordConfig: {}, octaveShift: 0 }, ctx(i)) as { triad: number[] });
    expect(outs.slice(3).every((o) => chordName(o.triad) === 'F#m')).toBe(true);
  });

  it('without holdChord, only a held A..G sounds', () => {
    const outs = run([reading('A'), reading('H')], { enabled: true, chords: true, holdChord: false });
    expect(outs.map((o) => o.triad.length)).toEqual([3, 0]);
  });

  it('is silent while the dial says no chords, and captions only when it does', () => {
    const [off] = run([reading('A')], { enabled: true, chords: false });
    expect(off!.triad).toEqual([]);
    expect(off!.captions).toEqual([]);
    const [on] = run([reading('A')]);
    expect(on!.captions[SIGN_IDS.indexOf('F')]).toBe('F#m');
  });

  it('plays through the shared realizer: voiceLed leads each chord from the last', () => {
    const midiOf = (o: { params: SynthParams }) => o.params.voices.filter((v) => v.present).map((v) => Math.round(69 + 12 * Math.log2(v.freq / 440))).sort((a, b) => a - b);
    const led = run([reading('D'), reading('G')], undefined, { voicing: 'voiceLed' }).map(midiOf);
    const spread = run([reading('D'), reading('G')], undefined, { voicing: 'spread' }).map(midiOf);
    const motion = ([a, b]: number[][]) => a!.reduce((sum, n, i) => sum + Math.abs((b![i] ?? n) - n), 0);
    expect(led[1]!.length).toBeGreaterThan(0);
    expect(motion(led)).toBeLessThan(motion(spread));
  });
});
