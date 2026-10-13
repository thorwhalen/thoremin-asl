# thoremin-asl

American Sign Language for [thoremin](https://github.com/thorwhalen/thoremin): the fingerspelled alphabet and common words (yes, no, hello, thank you, please, sorry, bye) read from the hands, and two instruments built on it.

- **ASL Viewer** (video only): the sign being made, written large, and a bar per sign (the alphabet, then the words), in the face-expression bar idiom.
- **ASL Chords**: the face-expression chords instrument with letters for faces. A..G play the chord of the key whose root has that letter name, with the key's accidentals applied: in D major, F plays F#m (iii) and C plays C#dim (vii°). The key, sound and voicing are the face chords' settings (`faceChord.*`), also editable in the Sign language section.

Both are in thoremin's Instruments view under **Sign instruments** (🤟). Try them at https://apps.thorwhalen.com/thoremin/.

It is a **lazy plugin**: thoremin ships only its stubs (about 5 kB: the dial, the branches as data, the two instruments) and fetches the rest (nodes, recogniser, weights, panel; about 150 kB) the first time a player picks an ASL instrument or switches the `asl` dial on. The pattern, and its guards, are thoremin's [`docs/design/lazy-extensions.md`](https://github.com/thorwhalen/thoremin/blob/main/docs/design/lazy-extensions.md).

## How thoremin uses it

thoremin depends on this repository as a git dependency pinned by commit, and names it in its `extensions.json` as a lazy extension:

```jsonc
// thoremin/package.json
"@thoremin/ext-asl": "github:thorwhalen/thoremin-asl#<commit sha>"
// thoremin/extensions.json
{ "id": "asl", "module": "@thoremin/ext-asl", "ui": "@thoremin/ext-asl/ui",
  "lazy": { "module": "@thoremin/ext-asl/lazy", "ui": "@thoremin/ext-asl/lazy_ui" } }
```

The repository is public, so `npm ci` installs it with no token, on the deploy host and in CI alike. The package is TypeScript source with no build step: thoremin's Vite build compiles it. The SDK packages it imports (`@thoremin/dag`, `lazy`, `sdk`, `sdk-ui`) and `react`, `zod`, `zustand` are peer dependencies that thoremin provides.

### Releasing a new version into thoremin

A change here reaches players only when a thoremin commit pins it:

1. Land the change here (PR, review, green CI, squash-merge). Note the merge commit's sha.
2. In a thoremin branch: `npm install github:thorwhalen/thoremin-asl#<sha>`. This updates the pin in `package.json` and `package-lock.json`.
3. Run thoremin's gates: `npm run typecheck && npm test && npm run build && npm run check:lazy`, plus the browser smoke (`npm run smoke`). Then open the PR. thoremin's owner merges it and deploys it.

Never pin a branch name: the lockfile must name an exact commit, so a deploy builds exactly what was tested.

### Developing against a thoremin checkout

Clone both side by side. In thoremin, copy your working copy over the installed package (re-run after each edit; `npm ci` in thoremin restores the pinned version):

```bash
cd thoremin
rsync -a --delete ../thoremin-asl/package.json ../thoremin-asl/src node_modules/@thoremin/ext-asl/
cp -r ../thoremin-asl/test/. test/ext-asl/   # this plugin's tests, in thoremin's suite (do not commit them there)
npm run dev                                   # the app, with your plugin
npm run typecheck && npx vitest run           # thoremin's gates against your plugin
```

A copy, not a symlink: TypeScript resolves a symlinked package's imports from its real path, outside thoremin's `node_modules`, and the shared SDK types go missing. (Vite alone would cope: thoremin's config resolves the shared packages from its own `node_modules`, `resolve.dedupe`.) This is exactly what this repository's CI does.

## What is where

| | |
|---|---|
| `src/index.ts`, `src/ui.tsx` | The EAGER stubs: the manifest without nodes (`provides` names them), the editor section as data. |
| `src/lazy.ts`, `src/lazy_ui.tsx` | The DEFERRED halves (`extensions.json`'s `lazy` entry): the node definitions, and the React half. |
| `src/schema.ts` | The `asl` dial's Zod schema, in its own module so the stub carries the dial without the node. |
| `src/dials.ts` | The `asl` dial: on/off, movement signs, confidence, smoothing, chords, hold. |
| `src/branches.ts` | `asl-signs` (the reader, plus thoremin's generic `categoryLabel` / `categoryBars` overlay cues) and `asl-chord` (requires it; a voice, the chord name). |
| `src/nodes/asl_signs.ts` | `asl-signs`: the hands to a reading (`reading`, by sign id; `signs`, the overlay's generic `CategoryScores`). Loads the weights lazily (`@thoremin/lazy`). |
| `src/nodes/sign_chord.ts` | `sign-chord`: A..G to the key's chord, voiced and rendered like the face chords. |
| `src/panels/asl.tsx`, `src/app/aslStatus.ts` | The Sign language section and its live readout. |
| `src/lib/recognizer/` | The recogniser (the extension's library): pure TypeScript (no DAG, no DOM, no ML runtime), usable without thoremin as `@thoremin/ext-asl/recognizer`. See [`docs/recognizer.md`](docs/recognizer.md). |
| `src/lib/recognizer/models/` | The trained weights (`thoremin-asl-mlp/1` JSON). |
| `training/` | The Python training pipeline: landmark extraction, the models, the fixtures. See [`training/README.md`](training/README.md). |
| `test/` | The recogniser's parity and replay tests (against fixtures written by `training/`) and the chord node's. |
| `docs/asl-recognition.md` | The research write-up: the data, the model choices, the numbers. |

## Tests and CI

This plugin is useless without its host, so its CI runs inside a real thoremin: it checks out thoremin's `main`, puts this commit in place of the pinned plugin, copies `test/` into thoremin's suite, and runs thoremin's typecheck, its whole test suite (thoremin's static guards scan the plugin's source too), the build and the lazy-bundle check (`.github/workflows/ci.yml`).

## Data and training

The models are trained only on permissively licensed data. Data with no licence, or a non-commercial or research-only one, is used for held-out testing and never for training. The sources, their licences and the attribution they require are in [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md). The raw data stays outside this repository, on the training machine (`~/.local/share/thoremin-asl/`). Only the small weight files are committed.

## Licence

None yet: thoremin's own licence is an open decision (thoremin#262), and this repository will follow it. The training data's licences, and the attribution they require, are in [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).
