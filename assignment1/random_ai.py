#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random legal baseline, equivalent to the course's supplied reference AI.

Set an instance's rng = random.Random(seed) for reproducible experiments.
"""

import random

NAME = "Random baseline"


class GomokuAI:
    def __init__(self, player_id, board_size, win_length):
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        self.rng = random.Random()

    def get_move(self, board, last_opponent_move, time_limit):
        empties = [(r, c) for r in range(self.board_size)
                   for c in range(self.board_size) if board[r][c] == 0]
        if not empties:
            raise ValueError("no legal move")
        return self.rng.choice(empties)
