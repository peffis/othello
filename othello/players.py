import random
from abc import ABC, abstractmethod
from typing import Tuple

from .board import Board, BLACK, WHITE
from .rules import make_move, get_possible_boards


# Classical positional weights for Othello evaluation. 8x8 row-major, flat
# tuple indexed by bit position (i = y*8 + x) to match Board.board layout.
# Symmetric under both axes and both diagonals: corners carry the highest
# weight (9), the C-squares adjacent to corners the lowest (1).
POSITION_WEIGHTS = (
    9, 1, 6, 4, 4, 6, 1, 9,
    1, 1, 3, 2, 2, 3, 1, 1,
    6, 3, 4, 4, 4, 4, 3, 6,
    4, 3, 4, 4, 4, 4, 3, 4,
    4, 3, 4, 4, 4, 4, 3, 4,
    6, 3, 4, 4, 4, 4, 3, 6,
    1, 1, 3, 2, 2, 3, 1, 1,
    9, 1, 6, 4, 4, 6, 1, 9,
)


def _weighted_position_score(board, color):
    """Sum of POSITION_WEIGHTS at every square the given colour owns."""
    bitboard = board.board[0] if color == WHITE else board.board[1]
    total = 0
    for i in range(64):
        if bitboard & (1 << i):
            total += POSITION_WEIGHTS[i]
    return total


class Player(ABC):
    def __init__(self, c):
        self.mycolor = c
        self.verbose = True

    @abstractmethod
    def make_move(self, b) -> Tuple[Board, str]:
        pass

    def reset(self):
        pass

    def contemplate(self, b, game_over=False):
        pass

    def setVerbosity(self, verbose):
        self.verbose = verbose


class RandomPlayer(Player):
    def make_move(self, b) -> Tuple[Board, str]:
        boards = get_possible_boards(b, self.mycolor)
        if len(boards) > 0:
            candidate, _, coord = random.choice(boards)
            return (candidate, coord)
        return (None, '')


class GreedyPlayer(Player):
    def make_move(self, b) -> Tuple[Board, str]:
        boards = get_possible_boards(b, self.mycolor)
        if len(boards) > 0:
            candidate, _, coord = max(boards, key=lambda bs: bs[1])
            return (candidate, coord)
        return (None, '')


class NoisyGreedyPlayer(Player):
    """Epsilon-noisy positional greedy: with probability `epsilon` picks a
    random legal move, otherwise plays the move that maximises the
    position-weighted sum of the player's pieces on the resulting board
    (see POSITION_WEIGHTS). Corners are prized, C-squares are avoided.
    The epsilon branch exists so evaluations against this otherwise
    deterministic policy can still average over variance."""

    def __init__(self, c, epsilon=0.1):
        super().__init__(c)
        self.epsilon = epsilon

    def make_move(self, b) -> Tuple[Board, str]:
        boards = get_possible_boards(b, self.mycolor)
        if not boards:
            return (None, '')
        if self.epsilon > 0 and random.random() < self.epsilon:
            candidate, _, coord = random.choice(boards)
            return (candidate, coord)
        candidate, _, coord = max(
            boards,
            key=lambda bs: _weighted_position_score(bs[0], self.mycolor),
        )
        return (candidate, coord)


class InteractivePlayer(Player):
    def make_move(self, b) -> Tuple[Board, str]:
        boards = get_possible_boards(b, self.mycolor)
        if len(boards) > 0:
            while True:
                b.display()
                move = input("enter your move: ")
                if move == 'q':
                    break
                if len(move) == 2:
                    col, row = move[0], move[1]
                    x = ord(col) - ord('a')
                    y = ord(row) - ord('0') - 1
                    if make_move(x, y, self.mycolor, b) > 0:
                        return (b, move)
                print("not a valid move")
        return (None, '')
