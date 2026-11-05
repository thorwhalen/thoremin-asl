/**
 * The recogniser: hand frames in, one score per sign out, every frame.
 *
 * Per frame it picks the signing hand (the one whose wrist is highest in the image, as in
 * training), scores its shape with the static model, and keeps a short history of that hand
 * for the motion model. The two models fuse into ONE distribution over {@link SIGN_IDS}: a
 * static letter's score is its static probability times the motion model's "no motion
 * sign" probability, a motion sign's score is the motion model's own. So the bars always
 * sum to one, and a held letter and a sign in flight compete on one scale.
 *
 * On top of the raw scores: exponential smoothing (a time constant, so the result does not
 * depend on the frame rate), a confidence floor below which no sign is named, hysteresis (a
 * new sign must beat the current one by a margin to take over), and a latch that keeps a
 * motion sign on screen for a moment after its movement ends (a word is over by the time it
 * is recognised).
 */
import { handFeatures, type FeatureKind, type HandInput } from './features';
import { mlpPredict, type MlpModel } from './mlp';
import { arbitrateOrientation } from './orientation';
import { toTrajectoryFrame, trajectoryLetter, type TrajectoryFrame } from './trajectory';
import { motionMeta, scoreMotion, toMotionFrame, type MotionFrame } from './motion';
import { inVocabulary, SIGN_IDS, STATIC_SIGNS, type SignId, type Vocabulary } from './labels';

export interface SignFrame {
  width: number;
  height: number;
  hands: readonly HandInput[];
}

export interface SignScores {
  /** A hand is being read. */
  present: boolean;
  /** The sign currently named, or null (no hand, or nothing confident enough). */
  label: SignId | null;
  /** Every class, in {@link SIGN_IDS} order. */
  labels: readonly SignId[];
  /** Smoothed 0..1 score per class, aligned with `labels`; sums to 1 while a hand is read. */
  scores: number[];
  /** Whether the named sign is a held shape or a movement. */
  kind: 'static' | 'motion' | null;
  /** Which hand signs. */
  hand: 'Left' | 'Right' | null;
  /** The score a sign needs to be named (the reader's `minConfidence`): what a consumer counts a
   *  reading's confidence from (sign-chord's and Sign Map's commit gate). */
  floor?: number;
  /** The reading is the last one, kept through a short loss of the hand (`dropout`): nothing new
   *  was read, so it is no new evidence for the sign it names. */
  held?: boolean;
}

export interface RecognizerOptions {
  /** Smoothing time constant in seconds (0 = none). */
  smoothing?: number;
  /** A sign is named only when its smoothed score reaches this. */
  minConfidence?: number;
  /** A new sign takes over only when it beats the current one by this much. */
  switchMargin?: number;
  /** How long a recognised motion sign stays named after it fires (s). */
  motionHold?: number;
  /** A motion sign fires when its raw window probability reaches this (default: the model's). */
  motionThreshold?: number;
  /** History kept for the motion model (s): the longest window plus slack. */
  history?: number;
  /** Frames closer than this are not added to the motion history (s): the training rate. */
  minFrameGap?: number;
  /** Frames without a hand longer than this clear the history (s). */
  dropout?: number;
  /** The signs that may be named: `all` (letters and words) or `letters` (an instrument that
   *  maps letters to music). A sign outside it is never named. What the motion model gave it
   *  goes back to the letters as far as the hand clearly holds a letter's shape
   *  ({@link returnBelow}..{@link returnAbove}): a held B the model calls HELLO is a B; a hand
   *  that holds no clear shape while it moves like a word (a conductor) is no letter. */
  vocabulary?: Vocabulary;
  /** The letter model's top probability at or below which none of a left-out sign's share
   *  goes back to the letters (measured: a conductor's hand 0.55 at the 75th percentile). */
  returnBelow?: number;
  /** ... at or above which all of it does (a player's held letters 0.77 at the 10th). */
  returnAbove?: number;
  /** Apply the orientation rule to letters that differ only in direction (G level, Q down:
   *  `./orientation`). */
  orientation?: boolean;
  /** Where J and Z come from: the motion `model`, the drawn `trajectory` (`./trajectory`), or
   *  `both` (either one names it). */
  motionLetters?: 'model' | 'trajectory' | 'both';
  /** A wrist that moves more than this many palm lengths between two frames is another hand
   *  (the signing hand is re-picked each frame): the drawn path starts again. */
  pathJump?: number;
}

export const DEFAULT_RECOGNIZER_OPTIONS: Required<Omit<RecognizerOptions, 'motionThreshold'>> = {
  vocabulary: 'all',
  returnBelow: 0.6,
  returnAbove: 0.8,
  orientation: true,
  motionLetters: 'trajectory',
  pathJump: 1.5,
  smoothing: 0.12,
  minConfidence: 0.45,
  switchMargin: 0.1,
  motionHold: 1.0,
  history: 2.0,
  minFrameGap: 1 / 15,
  dropout: 0.35,
};

export interface SignRecognizer {
  /** Read one frame taken at `t` seconds (the camera's capture time where there is one). A
   *  frame whose `t` has not advanced is the same camera frame ticked again: the last reading
   *  is returned unchanged. */
  update(frame: SignFrame | null | undefined, t: number): SignScores;
  /** Retune the smoothing, confidence floor, margin or hold WITHOUT losing the history (a
   *  slider drag calls this every frame). */
  setOptions(options: Omit<RecognizerOptions, 'history' | 'minFrameGap'>): void;
  reset(): void;
}

/** The hand a signer signs with: the highest wrist in the image. */
export function signingHand(hands: readonly HandInput[]): HandInput | null {
  let best: HandInput | null = null;
  for (const h of hands) if (h.keypoints.length >= 21 && (!best || h.keypoints[0]!.y < best.keypoints[0]!.y)) best = h;
  return best;
}

export function emptyScores(): SignScores {
  return { present: false, label: null, labels: SIGN_IDS, scores: SIGN_IDS.map(() => 0), kind: null, hand: null };
}

export function createSignRecognizer(staticModel: MlpModel, motionModel: MlpModel | null = null, options: RecognizerOptions = {}): SignRecognizer {
  const o = { ...DEFAULT_RECOGNIZER_OPTIONS, ...options };
  const meta = motionModel ? motionMeta(motionModel) : null;
  let motionThreshold = options.motionThreshold ?? meta?.threshold ?? 0.6;
  const staticIdx = staticModel.labels.map((l) => SIGN_IDS.indexOf(l as SignId));
  const motionIdx = motionModel?.labels.map((l) => SIGN_IDS.indexOf(l as SignId)) ?? [];
  const noneIdx = motionModel?.labels.indexOf('none') ?? -1;
  /** Per motion label, whether it may be named; undefined when every one may. */
  const motionAllowed = (v: Vocabulary, letters: RecognizerOptions['motionLetters']): boolean[] | undefined => {
    if (!motionModel || (v === 'all' && letters !== 'trajectory')) return undefined;
    return motionModel.labels.map((l, i) => i === noneIdx || (inVocabulary(l, v) && !(letters === 'trajectory' && (l === 'J' || l === 'Z'))));
  };
  let allowed = motionAllowed(o.vocabulary, o.motionLetters);
  let path: TrajectoryFrame[] = [];
  let lastDrawn: 'J' | 'Z' | null = null;
  let pathHand: HandInput['handedness'] | null = null;

  let history: MotionFrame[] = [];
  let smoothed = SIGN_IDS.map(() => 0);
  let smoothedUnnamed = 0;
  let lastT: number | null = null;
  let lastHandT = -Infinity;
  let current: SignId | null = null;
  let latched: { id: SignId; until: number } | null = null;
  /** The best motion sign on the previous frame was above threshold: a latch arms on the
   *  rising edge only, so a movement that stays inside the longest window does not keep
   *  re-arming it (and hold the word on screen for seconds). */
  let motionWasHigh = false;
  let last: SignScores = emptyScores();

  function reset(): void {
    history = [];
    path = [];
    lastDrawn = null;
    pathHand = null;
    smoothed = SIGN_IDS.map(() => 0);
    smoothedUnnamed = 0;
    lastT = null;
    lastHandT = -Infinity;
    current = null;
    latched = null;
    motionWasHigh = false;
    last = emptyScores();
  }

  function setOptions(next: Omit<RecognizerOptions, 'history' | 'minFrameGap'>): void {
    for (const [k, v] of Object.entries(next)) if (v !== undefined && k !== 'motionThreshold') (o as Record<string, unknown>)[k] = v;
    if (next.motionThreshold !== undefined) motionThreshold = next.motionThreshold;
    if (next.vocabulary !== undefined || next.motionLetters !== undefined) {
      allowed = motionAllowed(o.vocabulary, o.motionLetters);
      // A sign the new vocabulary leaves out does not stay on screen for the rest of its latch.
      if (latched && !inVocabulary(latched.id, o.vocabulary)) latched = null;
    }
  }

  function update(frame: SignFrame | null | undefined, t: number): SignScores {
    if (lastT !== null && t <= lastT) return last;
    const dt = lastT === null ? 0 : t - lastT;
    lastT = t;
    const hand = frame ? signingHand(frame.hands) : null;
    const x = hand && frame ? handFeatures(hand, frame.width, staticModel.features as FeatureKind) : null;
    if (!hand || !x) {
      // A frame or two without the hand (tracking blinked) keeps the last reading, so a
      // held chord does not dip; a longer absence is the hand gone.
      if (last.present && t - lastHandT <= o.dropout) {
        if (!last.held) last = { ...last, held: true };
        return last;
      }
      reset();
      lastT = t;
      return last;
    }
    lastHandT = t;

    const post = mlpPredict(staticModel, x);
    const raw = SIGN_IDS.map(() => 0);

    let motion: number[] | null = null;
    if (motionModel && meta) {
      const rec = toMotionFrame(hand, post, t);
      const last = history[history.length - 1];
      if (!last || t - last.t >= o.minFrameGap - 1e-6) history.push(rec);
      while (history.length && history[0]!.t < t - o.history) history.shift();
      motion = scoreMotion(motionModel, history, meta);
    }

    // The share the model gave to signs left out (the words in a letters-only reading, the model's
    // J and Z when the drawn path reads them) goes back to "no movement sign", and so to the
    // letters, as far as the hand clearly holds a letter's shape. What does not go back stays an
    // unnamed competitor: a hand moving like a word with no clear letter in it names nothing, as
    // when the word could be named (a conductor's curved hand is a weak C to the shape model and
    // THANK YOU to the motion model, frame after frame).
    // The motion model keeps the raw posterior it was trained on; the letters' own scores (and the
    // shape's confidence below) take the orientation rule.
    const shape = o.orientation ? arbitrateOrientation(post, staticModel.labels, hand) : post;
    let excluded = 0;
    if (motion && allowed) motion.forEach((p, i) => (excluded += allowed![i] ? 0 : p));
    const top = Math.max(...shape);
    const back = excluded > 0 ? Math.min(1, Math.max(0, (top - o.returnBelow) / (o.returnAbove - o.returnBelow))) : 1;
    const pNone = motion && noneIdx >= 0 ? motion[noneIdx]! + back * excluded : 1;
    const unnamed = (1 - back) * excluded;
    shape.forEach((p, i) => {
      const j = staticIdx[i]!;
      if (j >= 0) raw[j]! += p * pNone;
    });
    if (motion) {
      motion.forEach((p, i) => {
        const j = motionIdx[i]!;
        if (i !== noneIdx && j >= 0 && (!allowed || allowed[i])) raw[j]! += p;
      });
    }

    // The first frame of a hand starts from its own reading, not from zeros.
    const alpha = last.present && o.smoothing > 0 && dt > 0 ? 1 - Math.exp(-dt / o.smoothing) : 1;
    smoothed = smoothed.map((s, i) => s + (raw[i]! - s) * alpha);
    smoothedUnnamed += (unnamed - smoothedUnnamed) * alpha;

    // A motion sign fires on its RAW window probability (smoothing would blunt a
    // half-second movement) and is latched so it can be read.
    if (motion) {
      let bi = -1;
      let bp = 0;
      motion.forEach((p, i) => {
        if (i !== noneIdx && (!allowed || allowed[i]) && p > bp) {
          bp = p;
          bi = i;
        }
      });
      const high = bi >= 0 && bp >= motionThreshold && motionIdx[bi]! >= 0;
      if (high && (!motionWasHigh || latched?.id !== SIGN_IDS[motionIdx[bi]!])) latched = { id: SIGN_IDS[motionIdx[bi]!]!, until: t + o.motionHold };
      motionWasHigh = high;
    }
    // J and Z drawn in the air: fired on the rising edge of the path's reading, latched like a
    // motion sign, whatever the motion model says.
    if (motionModel && o.motionLetters !== 'model' && frame) {
      const tf = toTrajectoryFrame(hand, frame.width, t);
      // The signing hand is re-picked every frame (the higher wrist); when it changes, or the wrist
      // jumps further than a hand moves in one frame, the path so far belongs to another hand.
      const prev = path[path.length - 1];
      if (prev && (hand.handedness !== pathHand || Math.hypot(tf.wrist[0] - prev.wrist[0], tf.wrist[1] - prev.wrist[1]) > o.pathJump * tf.palm)) {
        path = [];
        lastDrawn = null;
      }
      pathHand = hand.handedness;
      path.push(tf);
      while (path.length && path[0]!.t < t - o.history) path.shift();
      const drawn = trajectoryLetter(path);
      if (drawn && drawn !== lastDrawn) latched = { id: drawn, until: t + o.motionHold };
      lastDrawn = drawn;
    }
    if (latched && t > latched.until) latched = null;

    let label: SignId | null;
    if (latched) label = latched.id;
    else {
      let top = 0;
      for (let i = 1; i < smoothed.length; i++) if (smoothed[i]! > smoothed[top]!) top = i;
      const topId = SIGN_IDS[top]!;
      const cur = current ? smoothed[SIGN_IDS.indexOf(current)]! : 0;
      if (smoothed[top]! < o.minConfidence || smoothedUnnamed > smoothed[top]!) label = null;
      else if (current && topId !== current && cur >= o.minConfidence && smoothed[top]! < cur + o.switchMargin) label = current;
      else label = topId;
    }
    current = label;
    const kind = label === null ? null : STATIC_SIGNS.includes(label) ? 'static' : 'motion';
    last = { present: true, label, labels: SIGN_IDS, scores: smoothed.slice(), kind, hand: hand.handedness, floor: o.minConfidence };
    return last;
  }

  return { update, setOptions, reset };
}
