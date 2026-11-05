/**
 * Letters that share a handshape and differ only in where the hand points.
 *
 * G and Q are the same hand (index and thumb out, the other fingers closed); G points the index
 * level, Q points it at the floor. The static model reads pointing direction among its
 * features, but on a real player it still read a level G as Q for most of a held G (measured on
 * the player's own recordings: 54 % of the right hand's G frames, its index 6 degrees below
 * level). So the direction is applied as the rule it is: within {@link OrientationPair.levelMaxDeg}
 * of level the pair's probability goes to the level letter, beyond
 * {@link OrientationPair.downMinDeg} below level to the downward one, and in between the model's
 * split stands. The pair's total is unchanged, so the posterior still sums to one.
 *
 * Only G/Q is listed because only G/Q has been measured. H/U and K/P are the same kind of pair
 * (H is a level U, P a downward K) and slot in here once a recording shows the model needs it.
 */

import type { HandInput } from './features';

export interface OrientationPair {
  level: string;
  down: string;
  /** The finger whose direction decides: [knuckle, tip] landmark indices. */
  finger: readonly [number, number];
  levelMaxDeg: number;
  downMinDeg: number;
}

export const ORIENTATION_PAIRS: readonly OrientationPair[] = [
  { level: 'G', down: 'Q', finger: [5, 8], levelMaxDeg: 35, downMinDeg: 55 },
];

/** Degrees below level of a finger in the image (+ down, - up; left and right alike). */
export function fingerDropDeg(hand: HandInput, [a, b]: readonly [number, number]): number {
  const p = hand.keypoints[a]!;
  const q = hand.keypoints[b]!;
  return (Math.atan2(q.y - p.y, Math.abs(q.x - p.x)) * 180) / Math.PI;
}

/** The static posterior with each orientation pair's mass moved to the letter the hand's
 *  direction allows. `labels` are the posterior's classes, in order. */
export function arbitrateOrientation(
  post: readonly number[],
  labels: readonly string[],
  hand: HandInput,
  pairs: readonly OrientationPair[] = ORIENTATION_PAIRS,
): number[] {
  const out = post.slice();
  for (const pr of pairs) {
    const i = labels.indexOf(pr.level);
    const j = labels.indexOf(pr.down);
    if (i < 0 || j < 0) continue;
    const drop = fingerDropDeg(hand, pr.finger);
    const mass = out[i]! + out[j]!;
    if (Math.abs(drop) <= pr.levelMaxDeg) {
      out[i] = mass;
      out[j] = 0;
    } else if (drop >= pr.downMinDeg) {
      out[i] = 0;
      out[j] = mass;
    }
  }
  return out;
}
