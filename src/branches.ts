/**
 * The ASL branches: the sign reader (hands to a reading, drawn as a big label and a bar per
 * sign) and, on top of it, the sign chords (A..G to chords of the key, into the synth).
 */
import { defineBranch } from '@thoremin/sdk/instruments/branch';
import { TRUNK } from '@thoremin/sdk/instruments/trunk';

const ui = (port: string) => ({ node: TRUNK.ui, port });
const to = (from: { node: string; port: string }, toRef: { node: string; port: string }) => ({ from, to: toRef });

/** Above any other classifier on the overlay's category pool. */
const ASL_PRIORITY = 100;

export const ASL_SIGNS_NODE_ID = 'aslSigns';
export const SIGN_CHORD_NODE_ID = 'signChord';

export const aslSigns = defineBranch({
  id: 'asl-signs',
  description: 'The ASL sign reader: the hands to a score per sign, drawn as the named sign and a bar per sign.',
  nodes: [{ id: ASL_SIGNS_NODE_ID, type: 'asl-signs', params: {} }],
  edges: [
    to({ node: TRUNK.cam, port: 'hands' }, { node: ASL_SIGNS_NODE_ID, port: 'hands' }),
    to(ui('asl'), { node: ASL_SIGNS_NODE_ID, port: 'config' }),
  ],
  // The reading goes to the overlay's category pool (thoremin#600 S10b), not a port named here. A
  // classifier, drawn ahead of the player's own categorizers while it reads; the priority is above
  // any other classifier, as the legacy slot always was.
  categories: [{ from: { node: ASL_SIGNS_NODE_ID, port: 'signs' }, kind: 'classifier', priority: ASL_PRIORITY }],
  overlay: ['categoryLabel', 'categoryBars'],
});

export const aslChord = defineBranch({
  id: 'asl-chord',
  description: 'ASL chords: the letters A..G play the chord of the key whose root has that letter, voiced like the face chords.',
  requires: ['asl-signs'],
  nodes: [
    { id: SIGN_CHORD_NODE_ID, type: 'sign-chord', params: {} },
  ],
  edges: [
    to({ node: ASL_SIGNS_NODE_ID, port: 'reading' }, { node: SIGN_CHORD_NODE_ID, port: 'reading' }),
    to(ui('chordSpec'), { node: SIGN_CHORD_NODE_ID, port: 'spec' }),
    to(ui('chordConfig'), { node: SIGN_CHORD_NODE_ID, port: 'chordConfig' }),
    to(ui('octaveShift'), { node: SIGN_CHORD_NODE_ID, port: 'octaveShift' }),
    to(ui('asl'), { node: SIGN_CHORD_NODE_ID, port: 'config' }),
  ],
  // The chord each letter plays, drawn as the bars' second label on the sign reader's bars.
  categoryCaptions: [{ from: { node: SIGN_CHORD_NODE_ID, port: 'captions' }, for: { node: ASL_SIGNS_NODE_ID, port: 'signs' } }],
  voices: [{ from: { node: SIGN_CHORD_NODE_ID, port: 'params' }, role: 'instrument' }],
  // The sign chord on the shared chord display (thoremin#600 S4): the host composes the selector
  // and its edge to the overlay. Below the face's chords (30, 20), as it always was.
  chords: [{ from: { node: SIGN_CHORD_NODE_ID, port: 'triad' }, priority: 10 }],
  overlay: ['chordName'],
});

export const ASL_BRANCHES = [aslSigns, aslChord] as const;
