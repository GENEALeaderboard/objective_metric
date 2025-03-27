## Objective evaluation for GENEA Leaderboard

This repository contains the code for the objective evaluation of the GENEA Leaderboard. The GENEA Leaderboard is available at [https://genea-workshop.github.io/leaderboard/](https://genea-workshop.github.io/leaderboard/). 

A large portion of the code is from https://huggingface.co/H-Liu1997/emage_evaltools, https://github.com/PantoMatrix/PantoMatrix, and https://github.com/facebookresearch/audio2photoreal/blob/main/utils/eval.py.

### Preparation

* Download `emage_evaltools` from [this link](https://huggingface.co/H-Liu1997/emage_evaltools) and place it in the `./emage_evaltools` directory.
* Install required Python packages using the PantoMatrix setup script: [PantoMatrix setup.sh](https://github.com/PantoMatrix/PantoMatrix/blob/main/setup.sh)

### How to run

* Place human motion files and generated motion files in `./examples/motion_human` and `./examples/motion_generated`, respectively. They must be in `.npz` format with matching filenames.
* Place associated audio files in `./BEAT2/beat_english_v2.0.0/wave16k/`, or specify a different path via the `make_list` function. Audio files must be in `.wav` format and have the same filenames as the motion files.
* Run the script to evaluate:

```bash
python exp_eval_metrics.py
```
