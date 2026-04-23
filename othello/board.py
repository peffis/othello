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
        self.board = [0, 0]
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
        self.board[index] |= 1 << (y * 8 + x)

    def clear(self, x, y):
        mask = ~(1 << (y * 8 + x))
        self.board[0] &= mask
        self.board[1] &= mask

    def get(self, x, y):
        mask = 1 << (y * 8 + x)
        if self.board[0] & mask:
            return WHITE
        if self.board[1] & mask:
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

    def toInputTensor(self):
        """Board as a (1, 8, 8, 2) channels-last tensor, for Conv2D inputs.
        Channel 0 = WHITE plane, channel 1 = BLACK plane. Bit i = (y, x)
        where y = i // 8, x = i % 8 — the same layout toInputVector uses."""
        words = np.array([self.board[0], self.board[1]], dtype=np.uint64)
        bits = np.unpackbits(words.view(np.uint8), bitorder='little')
        return (bits.astype(np.float32, copy=False)
                    .reshape(2, 8, 8)
                    .transpose(1, 2, 0)[np.newaxis])

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
