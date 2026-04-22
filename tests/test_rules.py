from othello.board import Board, WHITE, BLACK
from othello.rules import make_move, get_possible_boards, game_over, to_coord


def test_to_coord():
    assert to_coord(0, 0) == 'a1'
    assert to_coord(7, 7) == 'h8'
    assert to_coord(3, 2) == 'd3'


def test_opening_black_moves():
    b = Board()
    cands = get_possible_boards(b, BLACK)
    coords = sorted(c for _, _, c in cands)
    assert coords == ['c4', 'd3', 'e6', 'f5']


def test_opening_white_moves():
    b = Board()
    cands = get_possible_boards(b, WHITE)
    coords = sorted(c for _, _, c in cands)
    assert coords == ['c5', 'd6', 'e3', 'f4']


def test_make_move_flips_one_on_open():
    b = Board()
    flipped = make_move(3, 2, BLACK, b)  # d3
    assert flipped == 1
    assert b.get(3, 3) == BLACK  # the originally-white stone is flipped
    assert b.get(3, 2) == BLACK  # and the new stone is placed


def test_make_move_rejects_occupied_square():
    b = Board()
    assert make_move(3, 3, BLACK, b) == 0  # d4 is already white


def test_make_move_rejects_no_capture():
    b = Board()
    # a1 is empty but placing black there flips nothing → illegal
    assert make_move(0, 0, BLACK, b) == 0


def test_make_move_rejects_out_of_bounds():
    b = Board()
    assert make_move(-1, 0, BLACK, b) == 0
    assert make_move(8, 0, BLACK, b) == 0
    assert make_move(0, -1, BLACK, b) == 0
    assert make_move(0, 8, BLACK, b) == 0


def test_make_move_flips_multiple_directions():
    # Build a setup where one move flips across two directions.
    b = Board()
    # Board starts with white d4/e5 and black d5/e4.
    # BLACK at f4 flips WHITE at e4? No — e4 is already BLACK.
    # Easier: BLACK at c4 flips WHITE at d4 only (one direction, confirmed in other test).
    # For multi-flip, construct a contrived line.
    # Clear and rebuild: BLACK at (0,0), WHITE at (1,0)(2,0)(3,0), then BLACK at (4,0) flips three.
    b.clear(3, 3); b.clear(4, 4); b.clear(3, 4); b.clear(4, 3)
    b.set(0, 0, BLACK)
    b.set(1, 0, WHITE)
    b.set(2, 0, WHITE)
    b.set(3, 0, WHITE)
    flipped = make_move(4, 0, BLACK, b)
    assert flipped == 3
    assert b.get(1, 0) == BLACK
    assert b.get(2, 0) == BLACK
    assert b.get(3, 0) == BLACK


def test_game_over_initial():
    assert not game_over(Board())
