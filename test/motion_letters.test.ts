/**
 * The letters-only vocabulary, the G/Q orientation rule and the drawn J and Z (#3, #5 and the
 * letters-only issue): unit behaviour on self-made inputs, then detection rates on real clips.
 *
 * The real clips are `fixtures/motion_letters.json.gz`: SignAlphaSet's ten J and ten Z clips and
 * one clip of each held letter, a signer no model here was trained on (CC BY 4.0, see
 * ATTRIBUTIONS.md; built by `training/make_motion_letters_fixture.py`). Before this change the
 * shipped pair read J in 1 of those 10 clips and Z in 0.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import {
  arbitrateOrientation,
  createSignRecognizer,
  inVocabulary,
  parseMlpModel,
  strokes,
  trajectoryLetter,
  type HandInput,
  type RecognizerOptions,
  type TrajectoryFrame,
} from '@thoremin/ext-asl/recognizer';
import staticJson from '@thoremin/ext-asl/models/static.json';
import motionJson from '@thoremin/ext-asl/models/motion.json';

const staticModel = parseMlpModel(staticJson);
const motionModel = parseMlpModel(motionJson);

interface Clip {
  letter: string;
  video: string;
  width: number;
  height: number;
  frames: { t: number; hands: HandInput[] }[];
}
const { clips } = JSON.parse(gunzipSync(readFileSync(join(__dirname, 'fixtures', 'motion_letters.json.gz'))).toString('utf8')) as { clips: Clip[] };

function named(clip: Clip, options: RecognizerOptions): Set<string> {
  const r = createSignRecognizer(staticModel, motionModel, options);
  const out = new Set<string>();
  for (const f of clip.frames) {
    const s = r.update({ width: clip.width, height: clip.height, hands: f.hands }, f.t);
    if (s.label) out.add(s.label);
  }
  return out;
}

describe('the vocabulary', () => {
  it('letters only names letters, all names words too', () => {
    expect(inVocabulary('J', 'letters')).toBe(true);
    expect(inVocabulary('hello', 'letters')).toBe(false);
    expect(inVocabulary('hello', 'all')).toBe(true);
  });


  it('letters only never names a word on any real clip', () => {
    for (const c of clips) for (const id of named(c, { vocabulary: 'letters' })) expect(id.length, `${c.video} named ${id}`).toBe(1);
  });
});

describe('G and Q, the same hand pointed two ways', () => {
  const hand = (dx: number, dy: number): HandInput => ({
    handedness: 'Right',
    keypoints: Array.from({ length: 21 }, (_, i) => (i === 8 ? { x: 100 + dx, y: 100 + dy } : { x: 100, y: 100 })),
  });
  const labels = ['G', 'Q', 'A'];

  it('a level index gives the pair to G, a downward one to Q, in between the model decides', () => {
    expect(arbitrateOrientation([0.2, 0.7, 0.1], labels, hand(50, 5))).toEqual([0.8999999999999999, 0, 0.1]);
    expect(arbitrateOrientation([0.7, 0.2, 0.1], labels, hand(5, 50))).toEqual([0, 0.8999999999999999, 0.1]);
    expect(arbitrateOrientation([0.2, 0.7, 0.1], labels, hand(30, 30))).toEqual([0.2, 0.7, 0.1]);
  });
});

describe('J and Z drawn in the air', () => {
  const frame = (t: number, shape: TrajectoryFrame['shape'], ix: number, knuckles = 0, py = 0): TrajectoryFrame => ({
    t,
    shape,
    palm: 1,
    wrist: [0, 0],
    index: [ix, 0],
    pinky: [0, py],
    knuckles,
  });

  it('cuts a path into strokes', () => {
    expect(strokes([0, 1, 2, 1.5, 0.5, 1.5, 2.5], 0.6)).toEqual([2, 1.5, 2]);
    expect(strokes([0, 0.3, 0.1, 0.4], 0.6)).toEqual([0]);
  });

  it('an index drawing across, back and across is a Z; one stroke is not', () => {
    const z = [0, 1, 2, 1.5, 1, 0.5, 1, 1.5, 2, 2, 2].map((x, i) => frame(i / 10, 'index', x));
    expect(trajectoryLetter(z)).toBe('Z');
    const line = [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2].map((x, i) => frame(i / 10, 'index', x));
    expect(trajectoryLetter(line)).toBeNull();
  });

  it('a pinky hand that twists while its tip travels is a J; a still one is not', () => {
    const j = Array.from({ length: 12 }, (_, i) => frame(i / 10, 'pinky', 0, (i / 11) * 1.2, i * 0.2));
    expect(trajectoryLetter(j)).toBe('J');
    const still = Array.from({ length: 12 }, (_, i) => frame(i / 10, 'pinky', 0, 0, 0));
    expect(trajectoryLetter(still)).toBeNull();
  });

  it('two pointing hands that swap height are not a Z (the path starts again with each hand)', () => {
    // An index hand: index straight up, the other fingers folded back toward the wrist.
    const indexHand = (x: number, y: number): HandInput => {
      const k = Array.from({ length: 21 }, () => ({ x: x + 10, y: y - 50, z: 0 }));
      k[0] = { x, y, z: 0 };
      k[6] = { x, y: y - 60, z: 0 };
      k[8] = { x, y: y - 120, z: 0 };
      for (const [pip, tip] of [[10, 12], [14, 16], [18, 20]] as const) {
        k[pip] = { x: x + 15, y: y - 55, z: 0 };
        k[tip] = { x: x + 12, y: y - 40, z: 0 };
      }
      return { handedness: 'Right', keypoints: k };
    };
    const run = (pathJump: number) => {
      const r = createSignRecognizer(staticModel, motionModel, { vocabulary: 'letters', pathJump });
      const named = new Set<string>();
      for (let i = 0; i < 90; i++) {
        const leftHigher = Math.floor(i / 4) % 2 === 0;
        const hands = [indexHand(100, leftHigher ? 300 : 320), indexHand(400, leftHigher ? 320 : 300)];
        const s = r.update({ width: 640, height: 480, hands }, i / 30);
        if (s.label) named.add(s.label);
      }
      return named;
    };
    expect(run(Infinity).has('Z'), 'the scenario draws a Z when the hands are not told apart').toBe(true);
    expect(run(1.5).has('Z')).toBe(false);
  });

  // Measured on these clips (docs/recognizer.md): J 10/10 and Z 5/10 in both vocabularies with
  // the drawn path, 1 of the 24 held clips read as J or Z; the shipped motion model alone read J
  // 0/10 and Z 0/10 beside the words.
  for (const vocabulary of ['letters', 'all'] as const) {
    it(`reads a held-out signer's J and Z, and rarely a held letter as one (${vocabulary})`, () => {
      const rate = (letter: string) => {
        const cs = clips.filter((c) => c.letter === letter);
        return cs.filter((c) => named(c, { vocabulary }).has(letter)).length / cs.length;
      };
      expect(rate('J')).toBeGreaterThanOrEqual(0.9);
      expect(rate('Z')).toBeGreaterThanOrEqual(0.5);
      const held = clips.filter((c) => c.letter !== 'J' && c.letter !== 'Z');
      const falseFires = held.filter((c) => {
        const n = named(c, { vocabulary });
        return n.has('J') || n.has('Z');
      });
      expect(falseFires.length / held.length).toBeLessThanOrEqual(0.1);
    });
  }
});
