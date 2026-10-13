# Data attributions

The weights in `models/` are trained only on data whose licence allows it. This file lists every source the **shipped** models were trained on, with its licence and the attribution it asks for. When a model is retrained on more sources, the training PR updates this list in the same commit.

## Trained on (the shipped `models/static.json` and `models/motion.json`)

| Source | Licence | Used for |
|---|---|---|
| Aslense: [ZahidYasinMittha/American-Sign-Language-Dataset](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset) (Hugging Face) | MIT | letters (original clips only, not the augmented copies) and the movement signs |
| ASLNow!: [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) (Hugging Face) | MIT | letters |

The MIT licence asks that its copyright and permission notice be kept with copies of the data. These models contain no copy of the data, only weights learnt from MediaPipe hand landmarks extracted from it. Both datasets are credited here regardless.

## Tested on, never trained on

These are used only for held-out evaluation on the training machine. Nothing derived from them is in this repository.

| Source | Why test-only |
|---|---|
| [Marxulia/asl_sign_languages_alphabets_v03](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) | no licence stated |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft) | research-only, no redistribution |
| [WLASL](https://github.com/dxli94/WLASL) | C-UDA (computational use only) |
