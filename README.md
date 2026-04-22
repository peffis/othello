# othello

This repository contains a python notebook for a learning experiment around Othello
and temporal difference learning (TD-lambda). See https://blog.hellkvist.org for more info.

## Requirements

- Python 3.10 or newer (`int.bit_count()` is used in the bitboard code)
- macOS or Linux (TensorFlow wheels on recent Apple Silicon and x86_64 Just Work since TF 2.16)

## Setup

Create a virtual environment and install the project in editable mode together with
the dev and notebook extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebook]"
```

This pulls in `numpy`, `tensorflow`, and `keras` for the learner, `pytest` for the
test suite, and `jupyter` for running the notebook.

## Run the notebook

```bash
jupyter notebook othello.ipynb
```

(Or `jupyter lab` if you prefer JupyterLab.) The cells are meant to be run in order:

1. Imports from the `othello` package.
2. Self-training loop — long-running; trains a shared network by self-play.
3. Interactive play — human (black) vs. the trained model. Enter moves like `d3`,
   `q` to quit. Expects a saved model at `tdPlayer.weights.h5` / `tdPlayer.meta.json`,
   which the training cell writes at the end of each epoch.

## Run the tests

```bash
pytest -q
```

The suite covers `Board`, the move rules, and the TD(λ) eligibility-trace math.
The trace-math test is pure numpy — no TensorFlow required for that one.

## Layout

```
othello/
  board.py    Bitboard + color constants + display
  rules.py    make_move, get_possible_boards, game_over
  players.py  Player ABC + Random / Greedy / Interactive players
  nn.py       Keras model and NN wrapper
  td.py       TDPlayer with online TD(λ) eligibility-trace updates
  game.py     play_game driver
tests/        pytest suite
othello.ipynb Thin notebook driver (imports, training loop, interactive play)
```
