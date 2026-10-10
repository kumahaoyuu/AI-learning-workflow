#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""General (N, N, K) game rules, shared by the UI, CLI and match tools."""

from dataclasses import dataclass

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def validate_move(move, board):
    """Protocol v1 is deliberately strict: tuple of two native Python ints."""
    if not isinstance(move, tuple) or len(move) != 2:
        return "move must be an (int, int) tuple"
    r, c = move
    if type(r) is not int or type(c) is not int:
        return "coordinates must be native Python ints"
    n = len(board)
    if not (0 <= r < n and 0 <= c < n):
        return "coordinates outside the board"
    if board[r][c] != EMPTY:
        return "cell already occupied"
    return None


def winning_line(board, row, col, player, win_length):
    """Return the complete contiguous winning line, including an overline."""
    n = len(board)
    if board[row][col] != player:
        return ()
    for dr, dc in DIRECTIONS:
        line = [(row, col)]
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            arm = []
            while 0 <= r < n and 0 <= c < n and board[r][c] == player:
                arm.append((r, c))
                r, c = r + sign * dr, c + sign * dc
            if sign == -1:
                line = list(reversed(arm)) + line
            else:
                line.extend(arm)
        if len(line) >= win_length:
            return tuple(line)
    return ()


def check_win(board, row, col, player, win_length):
    return bool(winning_line(board, row, col, player, win_length))


@dataclass(frozen=True)
class Move:
    player: int
    row: int
    col: int
    elapsed: float = 0.0


class GomokuGame:
    def __init__(self, board_size=15, win_length=5):
        if type(board_size) is not int or board_size < 3:
            raise ValueError("N must be an integer >= 3")
        if type(win_length) is not int or not 3 <= win_length <= board_size:
            raise ValueError("K must satisfy 3 <= K <= N")
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = [[EMPTY] * self.board_size for _ in range(self.board_size)]
        self.to_move = BLACK
        self.winner = EMPTY
        self.draw = False
        self.history = []
        self.win_line = ()
        self.reason = ""

    @property
    def finished(self):
        return bool(self.winner or self.draw)

    def last_move(self, player=None):
        for move in reversed(self.history):
            if player is None or move.player == player:
                return move.row, move.col
        return None

    def play(self, move, elapsed=0.0):
        if self.finished:
            raise ValueError("game already finished")
        error = validate_move(move, self.board)
        if error:
            raise ValueError(error)
        r, c = move
        side = self.to_move
        self.board[r][c] = side
        self.history.append(Move(side, r, c, elapsed))
        self.win_line = winning_line(self.board, r, c, side, self.win_length)
        if self.win_line:
            self.winner, self.reason = side, "{}-in-a-row".format(self.win_length)
        elif len(self.history) == self.board_size ** 2:
            self.draw, self.reason = True, "board full"
        self.to_move = 3 - side

    def forfeit(self, player, reason):
        if self.finished:
            return
        self.winner, self.reason = 3 - player, reason

    def undo(self, count=1):
        if count < 1:
            return
        for _ in range(min(count, len(self.history))):
            move = self.history.pop()
            self.board[move.row][move.col] = EMPTY
            self.to_move = move.player
        self.winner, self.draw, self.win_line, self.reason = EMPTY, False, (), ""

    def format_board(self):
        n = self.board_size
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = ["    " + " ".join("{:2d}".format(c) for c in range(n))]
        for r in range(n):
            lines.append("{:3d} ".format(r) + " ".join("{:>2}".format(symbols[v]) for v in self.board[r]))
        return "\n".join(lines)
