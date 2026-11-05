/**
 * J and Z read from the path the hand draws, not from a learnt window.
 *
 * J and Z are the two letters that are movements: J is the I handshape (the pinky up) twisted
 * into a hook, Z the index drawing a zigzag. The motion model learns them from few clips (13 and
 * 7) and, with the words beside them, reads a held-out signer's Z 0 times in 10. What makes each
 * letter is simple geometry, so this path checks it directly over the last {@link
 * TrajectoryOptions.window} seconds:
 *
 * - the handshape: for J the pinky extended and the index and middle not; for Z the index extended
 *   and the ring and pinky not, in at least {@link TrajectoryOptions.shapeFraction} of the window's
 *   frames (J: {@link TrajectoryOptions.jShapeFraction}). A finger is extended when its tip is {@link TrajectoryOptions.extendRatio} times farther
 *   from the wrist than its middle joint (3D: z scaled like x);
 * - Z: the index tip's horizontal path, in palm lengths, has at least three strokes of
 *   {@link TrajectoryOptions.zStroke} or more (a stroke ends when the path comes back by
 *   {@link TrajectoryOptions.zHysteresis}): across, back, across. Direction is not checked, so a
 *   left or right hand, mirrored or not, draws the same Z;
 * - J: the knuckle line (index to pinky knuckle) turns by {@link TrajectoryOptions.jTurnDeg} or
 *   more, which is the twist of the hook, while the pinky tip travels {@link
 *   TrajectoryOptions.jTravel} palm lengths or more.
 *
 * Measured on SignAlphaSet's J and Z clips (a signer no model was trained on; the numbers are in
 * `docs/recognizer.md`).
 */
import type { HandInput, Point3 } from './features';

export interface TrajectoryOptions {
  window: number;
  minFrames: number;
  /** Z: the share of the window's frames that must show the index hand. */
  shapeFraction: number;
  /** J: the share that must show the pinky hand (stricter: a hand coming up into another letter
   *  passes through an I for a few frames). */
  jShapeFraction: number;
  extendRatio: number;
  zStroke: number;
  zHysteresis: number;
  zStrokes: number;
  jTurnDeg: number;
  jTravel: number;
}

export const DEFAULT_TRAJECTORY_OPTIONS: TrajectoryOptions = {
  window: 1.8,
  minFrames: 8,
  shapeFraction: 0.4,
  jShapeFraction: 0.6,
  extendRatio: 1.15,
  zStroke: 0.8,
  zHysteresis: 0.6,
  zStrokes: 3,
  jTurnDeg: 50,
  jTravel: 1.5,
};

/** What the path keeps of one frame of the signing hand. */
export interface TrajectoryFrame {
  t: number;
  shape: 'index' | 'pinky' | null;
  palm: number;
  wrist: [number, number];
  index: [number, number];
  pinky: [number, number];
  /** The knuckle line's angle in the image, radians. */
  knuckles: number;
}

const TIP = { index: 8, middle: 12, ring: 16, pinky: 20 } as const;
const PIP = { index: 6, middle: 10, ring: 14, pinky: 18 } as const;

function extended(k: readonly Point3[], f: keyof typeof TIP, width: number, ratio: number): boolean {
  const w = k[0]!;
  const d = (i: number) => Math.hypot(k[i]!.x - w.x, k[i]!.y - w.y, ((k[i]!.z ?? 0) - (w.z ?? 0)) * width);
  return d(TIP[f]) > ratio * d(PIP[f]);
}

/** The handshape a motion letter starts from, or null. */
export function motionLetterShape(hand: HandInput, width: number, ratio = DEFAULT_TRAJECTORY_OPTIONS.extendRatio): 'index' | 'pinky' | null {
  const k = hand.keypoints;
  if (k.length < 21) return null;
  const e = (f: keyof typeof TIP) => extended(k, f, width, ratio);
  if (e('index') && !e('ring') && !e('pinky')) return 'index';
  if (e('pinky') && !e('index') && !e('middle')) return 'pinky';
  return null;
}

export function toTrajectoryFrame(hand: HandInput, width: number, t: number, ratio?: number): TrajectoryFrame {
  const k = hand.keypoints;
  return {
    t,
    shape: motionLetterShape(hand, width, ratio),
    palm: Math.hypot(k[9]!.x - k[0]!.x, k[9]!.y - k[0]!.y) || 1,
    wrist: [k[0]!.x, k[0]!.y],
    index: [k[8]!.x, k[8]!.y],
    pinky: [k[20]!.x, k[20]!.y],
    knuckles: Math.atan2(k[17]!.y - k[5]!.y, k[17]!.x - k[5]!.x),
  };
}

/** The lengths of a 1-D path's monotone strokes; a stroke ends when the path comes back by
 *  `hysteresis` from its furthest point. */
export function strokes(xs: readonly number[], hysteresis: number): number[] {
  if (!xs.length) return [];
  const pivots = [xs[0]!];
  let dir = 0;
  let far = xs[0]!;
  for (const x of xs.slice(1)) {
    if (dir === 0) {
      if (Math.abs(x - pivots[0]!) > hysteresis) {
        dir = x > pivots[0]! ? 1 : -1;
        far = x;
      }
      continue;
    }
    if ((x - far) * dir > 0) far = x;
    else if (Math.abs(x - far) > hysteresis) {
      pivots.push(far);
      dir = -dir;
      far = x;
    }
  }
  pivots.push(far);
  const out: number[] = [];
  for (let i = 0; i + 1 < pivots.length; i++) out.push(Math.abs(pivots[i + 1]! - pivots[i]!));
  return out;
}

/** Which motion letter the last window of frames draws, or null. Frames time-ascending. */
export function trajectoryLetter(frames: readonly TrajectoryFrame[], o: TrajectoryOptions = DEFAULT_TRAJECTORY_OPTIONS): 'J' | 'Z' | null {
  if (!frames.length) return null;
  const now = frames[frames.length - 1]!.t;
  const win = frames.filter((f) => f.t > now - o.window - 1e-9);
  if (win.length < o.minFrames) return null;
  const palm = win.reduce((s, f) => s + f.palm, 0) / win.length || 1;
  const frac = (s: TrajectoryFrame['shape']) => win.filter((f) => f.shape === s).length / win.length;
  if (frac('index') >= o.shapeFraction) {
    const big = strokes(win.map((f) => f.index[0] / palm), o.zHysteresis).filter((s) => s >= o.zStroke);
    if (big.length >= o.zStrokes) return 'Z';
  }
  if (frac('pinky') >= o.jShapeFraction) {
    const a0 = win[0]!.knuckles;
    let turn = 0;
    for (const f of win) {
      const d = Math.atan2(Math.sin(f.knuckles - a0), Math.cos(f.knuckles - a0));
      turn = Math.max(turn, Math.abs(d));
    }
    let travel = 0;
    for (let i = 1; i < win.length; i++) travel += Math.hypot(win[i]!.pinky[0] - win[i - 1]!.pinky[0], win[i]!.pinky[1] - win[i - 1]!.pinky[1]) / palm;
    if ((turn * 180) / Math.PI >= o.jTurnDeg && travel >= o.jTravel) return 'J';
  }
  return null;
}
