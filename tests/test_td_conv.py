"""Smoke test for TDPlayerConv — exercises the conv forward path
end-to-end (board → (1, 8, 8, 2) input → tf.function trace → argmax)
without any training."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from othello.board import Board, BLACK
from othello.td import TDPlayerConv


def test_td_player_conv_returns_legal_opening_move():
    player = TDPlayerConv(BLACK)
    player.epsilon = 0.0   # force the deterministic-greedy branch
    b = Board()
    move, coord = player.make_move(b)
    assert move is not None, "conv player returned no move on opening board"
    assert coord in {'c4', 'd3', 'e6', 'f5'}, f"unexpected coord {coord!r}"
