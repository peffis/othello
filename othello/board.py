import numpy as np

UNOCCUPIED = 0
WHITE = 1
BLACK = 2


def color_to_string(color):
    c = ' · '
    if color == WHITE:
        c = ' x '
    if color == BLACK:
        c = ' o '
    return c


class Board:
    def __init__(self):
        self.board = [np.uint64(0), np.uint64(0)]
        self.set(3, 3, WHITE)
        self.set(4, 4, WHITE)
        self.set(3, 4, BLACK)
        self.set(4, 3, BLACK)

    def set(self, x, y, color):
        if color == WHITE:
            index = 0
        elif color == BLACK:
            index = 1
        else:
            return
        mask = np.uint64(1) << np.uint64(y * 8 + x)
        self.board[index] = np.bitwise_or(self.board[index], mask)

    def clear(self, x, y):
        mask = np.uint64(1) << np.uint64(y * 8 + x)
        self.board[0] = np.bitwise_and(self.board[0], ~mask)
        self.board[1] = np.bitwise_and(self.board[1], ~mask)

    def get(self, x, y):
        mask = np.uint64(1) << np.uint64(y * 8 + x)
        if np.bitwise_and(self.board[0], mask) > 0:
            return WHITE
        if np.bitwise_and(self.board[1], mask) > 0:
            return BLACK
        return UNOCCUPIED

    def copy(self):
        clone = Board.__new__(Board)
        clone.board = [self.board[0], self.board[1]]
        return clone

    def toInputVector(self):
        words = np.array([self.board[0], self.board[1]], dtype=np.uint64)
        bits = np.unpackbits(words.view(np.uint8), bitorder='little')
        return bits.astype(np.float32, copy=False)[np.newaxis]

    def count(self):
        whites = self.board[0]
        blacks = self.board[1]
        return (int(whites).bit_count(), int(blacks).bit_count())

    def display(self):
        print('   a  b  c  d  e  f  g  h')
        for y in range(8):
            print('{} '.format(y + 1), end='')
            for x in range(8):
                color = self.get(x, y)
                print(color_to_string(color), end='')
            print('')
