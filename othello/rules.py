from .board import UNOCCUPIED, WHITE, BLACK


def other_color(color):
    if color == WHITE:
        return BLACK
    if color == BLACK:
        return WHITE


def make_move(x, y, color, b):
    if x < 0 or x > 7 or y < 0 or y > 7 or b.get(x, y) != UNOCCUPIED:
        return 0

    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    flipped = 0
    otherColor = other_color(color)

    for d in range(8):
        cx = x + dx[d]
        cy = y + dy[d]

        steppedOverOpponent = False
        while 0 <= cx < 8 and 0 <= cy < 8 and b.get(cx, cy) == otherColor:
            cx += dx[d]
            cy += dy[d]
            steppedOverOpponent = True

        if 0 <= cx < 8 and 0 <= cy < 8 and b.get(cx, cy) == color and steppedOverOpponent:
            cx -= dx[d]
            cy -= dy[d]
            while cx != x or cy != y:
                b.clear(cx, cy)
                b.set(cx, cy, color)
                cx -= dx[d]
                cy -= dy[d]
                flipped += 1

    if flipped > 0:
        b.set(x, y, color)

    return flipped


def to_coord(x, y):
    return '{}{}'.format("abcdefgh"[x], y + 1)


def get_possible_boards(b, color):
    boards = []
    for y in range(8):
        for x in range(8):
            clone = b.copy()
            flipped = make_move(x, y, color, clone)
            if flipped > 0:
                boards.append((clone, flipped, to_coord(x, y)))
    return boards


def game_over(b):
    return len(get_possible_boards(b, WHITE)) == 0 and len(get_possible_boards(b, BLACK)) == 0
