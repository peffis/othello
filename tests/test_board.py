from othello.board import Board, WHITE, BLACK, UNOCCUPIED


def test_starting_state():
    b = Board()
    assert b.get(3, 3) == WHITE
    assert b.get(4, 4) == WHITE
    assert b.get(3, 4) == BLACK
    assert b.get(4, 3) == BLACK
    assert b.get(0, 0) == UNOCCUPIED


def test_count_initial():
    assert Board().count() == (2, 2)


def test_set_get_clear():
    b = Board()
    b.set(0, 0, WHITE)
    assert b.get(0, 0) == WHITE
    b.clear(0, 0)
    assert b.get(0, 0) == UNOCCUPIED


def test_toInputVector_alignment():
    """Each color occupies a fixed 64-slot block. The starting board
    puts WHITE at squares 27, 36 (offsets 0..63) and BLACK at squares
    28, 35 (offsets 64..127)."""
    b = Board()
    v = b.toInputVector()
    assert v.shape == (1, 128)
    ones = sorted(int(i) for i, x in enumerate(v[0]) if x == 1)
    assert ones == [27, 36, 92, 99]
    assert v[0].sum() == 4


def test_copy_is_independent():
    b = Board()
    c = b.copy()
    b.set(0, 0, WHITE)
    assert c.get(0, 0) == UNOCCUPIED
    c.set(7, 7, BLACK)
    assert b.get(7, 7) == UNOCCUPIED


def test_toInputTensor_shape_and_positions():
    """Shape (1, 8, 8, 2) with channel 0 = WHITE, channel 1 = BLACK.
    Starting board: WHITE at (3,3) and (4,4); BLACK at (3,4) and (4,3).
    Note that in a (y, x, c) tensor, a piece at (x, y) is at t[0, y, x, c]."""
    b = Board()
    t = b.toInputTensor()
    assert t.shape == (1, 8, 8, 2)
    assert t.sum() == 4
    # WHITE (channel 0) at (x=3, y=3) and (x=4, y=4).
    assert t[0, 3, 3, 0] == 1
    assert t[0, 4, 4, 0] == 1
    # BLACK (channel 1) at (x=3, y=4) and (x=4, y=3).
    assert t[0, 4, 3, 1] == 1
    assert t[0, 3, 4, 1] == 1
