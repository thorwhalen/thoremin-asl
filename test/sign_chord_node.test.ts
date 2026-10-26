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
