import random
from abc import ABC, abstractmethod
from typing import Tuple

from .board import Board, BLACK, WHITE
from .rules import make_move, get_possible_boards


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
    """Epsilon-noisy greedy: with probability `epsilon` picks a random legal
    move, otherwise plays the max-flip move. Exists so evaluations against
    an otherwise deterministic opponent can still average over variance."""

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
        candidate, _, coord = max(boards, key=lambda bs: bs[1])
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
