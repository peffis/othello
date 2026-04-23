import json
import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

from .board import Board, BLACK
from .rules import get_possible_boards
from .players import Player
from .nn import NN


ALPHA = 0.01
LAMBDA = 0.7


class TDPlayer(Player):
    def __init__(self, c, model=None):
        super().__init__(c)
        params = {'n_inputs': 128, 'n_hidden': 60, 'n_outputs': 1}
        self.model = model if model is not None else NN(params)

        self.predictions = []
        self.eligibility = None
        self.gamesPlayed = 0
        self.epsilon = 0.1

    def _paths(self, fname):
        base, _ = os.path.splitext(fname)
        return base + '.weights.h5', base + '.meta.json'

    def save(self, fname):
        weights_path, meta_path = self._paths(fname)
        self.model.model.save_weights(weights_path)
        with open(meta_path, 'w') as f:
            json.dump({'gamesPlayed': self.gamesPlayed}, f)

    def load(self, fname):
        weights_path, meta_path = self._paths(fname)
        self.model.model.load_weights(weights_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.gamesPlayed = meta['gamesPlayed']

    def reset(self):
        self.predictions = []
        # Zero the eligibility Variables in place if they've already been
        # allocated; otherwise leave as None for lazy allocation on the first
        # contemplate call. Avoids creating fresh tf.Variables per game.
        if self.eligibility is not None:
            for ev in self.eligibility:
                ev.assign(tf.zeros_like(ev))
        self.gamesPlayed += 1

    def make_move(self, b) -> Tuple[Board, str]:
        boards = get_possible_boards(b, self.mycolor)
        if len(boards) == 0:
            return (None, '')

        if self.epsilon > 0 and random.random() < self.epsilon:
            candidate, _, coord = random.choice(boards)
            return (candidate, coord)

        X = np.concatenate([cand.toInputVector() for cand, _, _ in boards], axis=0)
        preds = self.model._forward(X).numpy().reshape(-1)
        if self.mycolor == BLACK:
            preds = -preds
        best_idx = int(np.argmax(preds))
        candidate, _, coord = boards[best_idx]
        return (candidate, coord)

    def _apply_td_update(self, target, current):
        delta = ALPHA * (target - current)
        for var, e in zip(self.model.trainable_variables(), self.eligibility):
            var.assign_add(delta * e)

    def contemplate(self, b, game_over=False):
        # Online TD(λ) with accumulating eligibility traces:
        #   δ = V(s_{t+1}) - V(s_t)
        #   w ← w + α δ e_t
        #   e_{t+1} ← λ e_t + ∇V(s_{t+1})
        # At the terminal call V(s_{t+1}) is replaced by the actual outcome.
        if game_over:
            if not self.predictions:
                return
            whites, blacks = b.count()
            if whites > blacks:
                V_final = 1.0
            elif whites < blacks:
                V_final = -1.0
            else:
                V_final = 0.0
            self._apply_td_update(V_final, self.predictions[-1])
            return

        X_values = b.toInputVector()
        p, grads = self.model._forward_and_grad(X_values)
        p_scalar = float(p[0][0])

        if self.eligibility is None:
            # First contemplate ever on this TDPlayer — allocate the trace
            # Variables matching the model's trainable-var shapes. Zero-init;
            # the assign below will set them to g on this step.
            self.eligibility = [tf.Variable(tf.zeros_like(g)) for g in grads]

        if self.predictions:
            self._apply_td_update(p_scalar, self.predictions[-1])

        # Running eligibility trace update:  e ← λ·e + ∇V(s_{t+1})
        for ev, g in zip(self.eligibility, grads):
            ev.assign(LAMBDA * ev + g)

        self.predictions.append(p_scalar)
