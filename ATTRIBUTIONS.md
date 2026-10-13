# Data attributions

The weights in `src/lib/recognizer/models/` are trained only on data whose licence allows it. This file lists every source the **shipped** models were trained on, with its licence and the attribution it asks for, and every source the test fixtures carry landmarks from. When a model is retrained on more sources, the training PR updates this list in the same commit.

Changes made to every source below: MediaPipe hand landmarks were extracted from the images or videos, and the models were trained on those landmarks. The models contain no copy of any image or video, only weights learnt from the landmarks.

## Trained on (the shipped `static.json` and `motion.json`)

| Source | Licence | Used for |
|---|---|---|
| PopSign ASL v1.0: T. Starner et al., "PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones", NeurIPS 2023 Datasets and Benchmarks ([data](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | movement signs: yes, no, hello, thank you, please, bye (training signers only), and 15 other signs as "no sign" examples |
| ASL-HG: American Sign Language Hand Gesture Image Dataset, M. F. I. Pranto, M. R. Islam, M. A. Akbor, N. Ghosh, M. R. Alam, S. Chaki, M. M. Islam, Mendeley Data, 2025, [doi:10.17632/j4y5w2c8w9.1](https://doi.org/10.17632/j4y5w2c8w9.1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | letters |
| Images of American Sign Language (ASL) Alphabet Gestures, A. R. Ahmed Sowrow, M. I. A. S. Iftekhar, S. I. Prova, M. R. Huq, Mendeley Data, 2024, [doi:10.17632/48dg9vhmyk.2](https://doi.org/10.17632/48dg9vhmyk.2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | letters |
| [om192006/sign_language_keypoints](https://huggingface.co/datasets/om192006/sign_language_keypoints) (Hugging Face) | MIT | letters |
| ASLNow!: [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) (Hugging Face) | MIT | letters |
| [StudioGalt Sign-Language-Mocap-Archive](https://github.com/StudioGalt/Sign-Language-Mocap-Archive) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | letters, J, Z, hello, please, bye |
| Aslense: [ZahidYasinMittha/American-Sign-Language-Dataset](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset) (Hugging Face) | MIT (see the note below) | letters (original clips only, not the augmented copies), J, Z and the words |

The MIT licence asks that its copyright and permission notice be kept with copies of the data. The models contain no copy of the data; the MIT sources are credited here regardless.

### Aslense: what was excluded, and what is still in doubt

Aslense's card says MIT, and also that its videos were "scraped, collected from multiple sources". Checking it against the test-only corpora found **238 of its clips to be copies of them**: every ASL Citizen clip of the seven words (208 byte-identical, 6 by file name) and 24 byte-identical WLASL clips. Those 238 clips are excluded from training (the models shipped before 2026-10 were trained on them).

The rest of Aslense is still used, and its MIT grant is weak: several clips are named like WLASL downloads or dictionary-site videos. SORRY in particular is learnt from 6 Aslense clips only, and J and Z mostly from Aslense. Replacing Aslense with permissive data from many signers is tracked in this repository's issues.

## In the test fixtures

`test/fixtures/clips.json` and `test/fixtures/parity.json` carry MediaPipe hand landmarks (no pixels) of a few clips from:

| Source | Licence |
|---|---|
| SignAlphaSet, B. Garg, M. Kasar, A. Kashyap, A. Vats, G. Sharma, A. Hange, Mendeley Data, 2025, [doi:10.17632/8fmvr9m98w.3](https://doi.org/10.17632/8fmvr9m98w.3) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| PopSign ASL v1.0 (above) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| StudioGalt Sign-Language-Mocap-Archive (above) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

## Tested on, never trained on

These are used only for held-out evaluation on the training machine. Apart from the fixtures above, nothing derived from them is in this repository.

| Source | Why test-only |
|---|---|
| [Marxulia/asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) | no licence stated |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft) | research-only, no redistribution |
| [WLASL](https://github.com/dxli94/WLASL) | C-UDA (computational use only) |
| SignAlphaSet (above) | CC BY 4.0, so trainable; held out by choice, as the one permissive source no model sees, and the only held-out source for Z |
