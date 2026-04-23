from othello.board import Board, WHITE, BLACK
from othello.players import (
    NoisyGreedyPlayer,
    POSITION_WEIGHTS,
    _weighted_position_score,
)


def test_weighted_position_score_starting_board():
    """Starting board has each colour on two centre squares. The centre
    weights in POSITION_WEIGHTS are all 4, so each side scores 4+4=8."""
    b = Board()
    assert _weighted_position_score(b, WHITE) == 8
    assert _weighted_position_score(b, BLACK) == 8


def test_position_weights_layout_invariants():
    """Lock in the key properties of the weight table so a future edit
    cannot silently break the heuristic."""
    rows = [POSITION_WEIGHTS[y * 8 : (y + 1) * 8] for y in range(8)]
    # Horizontal symmetry (each row is a palindrome).
    for y, r in enumerate(rows):
        assert r == r[::-1], f'row {y} not palindromic: {r}'
    # Vertical symmetry (top half mirrors bottom half).
    for y in range(4):
        assert rows[y] == rows[7 - y]
    # Corners are the max (9).
    for y, x in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        assert rows[y][x] == 9
    # C-squares adjacent to the top-left corner are 1.
    assert rows[0][1] == 1
    assert rows[1][0] == 1


def test_noisy_greedy_with_epsilon_zero_plays_weighted_greedy():
    """epsilon=0 forces the deterministic-greedy branch. All four
    opening moves tie at weighted-score 16 (the player ends up with four
    pieces at centre squares, each weight 4), so any one of them is
    an acceptable pick."""
    b = Board()
    player = NoisyGreedyPlayer(BLACK, epsilon=0.0)
    move, coord = player.make_move(b)
    assert move is not None
    assert coord in {'c4', 'd3', 'e6', 'f5'}
    # The chosen move should produce a board with weighted-score 16
    # for BLACK (the tie all four opening moves share under this matrix).
    assert _weighted_position_score(move, BLACK) == 16
