from othello.board import Board, BLACK
from othello.players import NoisyGreedyPlayer


def test_noisy_greedy_with_epsilon_zero_plays_greedy():
    """epsilon=0 forces the deterministic-greedy branch; on the opening
    board every legal move flips exactly one disk, so any of the four
    openers is acceptable. The coord must be one of them and the flip
    count must be the max available."""
    b = Board()
    player = NoisyGreedyPlayer(BLACK, epsilon=0.0)
    move, coord = player.make_move(b)
    assert move is not None
    assert coord in {'c4', 'd3', 'e6', 'f5'}
