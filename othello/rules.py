from .board import UNOCCUPIED, WHITE, BLACK


DX = (-1, 0, 1, -1, 1, -1, 0, 1)
DY = (-1, -1, -1, 0, 0, 1, 1, 1)


def other_color(color):
    if color == WHITE:
        return BLACK
    if color == BLACK:
        return WHITE


def legal_flips(x, y, color, b):
    """Squares that would be flipped if `color` plays at (x, y); empty list
    if the move is illegal. Does not mutate `b`."""
    if x < 0 or x > 7 or y < 0 or y > 7 or b.get(x, y) != UNOCCUPIED:
        return []

    flips = []
    otherColor = other_color(color)

    for d in range(8):
        cx = x + DX[d]
        cy = y + DY[d]

        run_start = len(flips)
        while 0 <= cx < 8 and 0 <= cy < 8 and b.get(cx, cy) == otherColor:
            flips.append((cx, cy))
            cx += DX[d]
            cy += DY[d]

        # Keep the run only if bounded by our own color; otherwise rewind.
        if (
            len(flips) == run_start
            or not (0 <= cx < 8 and 0 <= cy < 8)
            or b.get(cx, cy) != color
        ):
            del flips[run_start:]

    return flips


def apply_flips(x, y, color, b, flips):
    """Mutate `b`: place `color` at (x, y) and flip every square in `flips`.
    Assumes legality has already been established."""
    for cx, cy in flips:
        b.clear(cx, cy)
        b.set(cx, cy, color)
    b.set(x, y, color)


def make_move(x, y, color, b):
    """Play `color` at (x, y) on `b`, mutating it in place. Returns the
    number of flipped pieces, or 0 if the move is illegal (in which case
    `b` is untouched)."""
    flips = legal_flips(x, y, color, b)
    if not flips:
        return 0
    apply_flips(x, y, color, b, flips)
    return len(flips)


def to_coord(x, y):
    return '{}{}'.format("abcdefgh"[x], y + 1)


def get_possible_boards(b, color):
    boards = []
    for y in range(8):
        for x in range(8):
            flips = legal_flips(x, y, color, b)
            if flips:
                clone = b.copy()
                apply_flips(x, y, color, clone, flips)
                boards.append((clone, len(flips), to_coord(x, y)))
    return boards


def game_over(b):
    return len(get_possible_boards(b, WHITE)) == 0 and len(get_possible_boards(b, BLACK)) == 0
