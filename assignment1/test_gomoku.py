#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavioral regressions for rules, tactical search, rollback and protocol.

Run: python test_gomoku.py
These tests do not establish tournament strength; experiments.py measures that.
"""

import copy
import random
import time
import unittest

from gomoku_ai import GomokuAI, _Position, _SearchTimeout
from gomoku_game import GomokuGame, check_win, validate_move


def winning_moves(board, player, k):
    wins = set()
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == 0:
                board[r][c] = player
                if check_win(board, r, c, player, k):
                    wins.add((r, c))
                board[r][c] = 0
    return wins


class RulesTests(unittest.TestCase):
    def test_all_directions_overline_and_edges(self):
        for n, k in ((5, 3), (9, 4), (15, 5), (15, 7), (15, 15), (17, 6)):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                board = [[0] * n for _ in range(n)]
                r0, c0 = (0, n - 1) if dc < 0 else (0, 0)
                length = min(n, k + 1)
                for j in range(length):
                    board[r0 + j * dr][c0 + j * dc] = 2
                self.assertTrue(check_win(board, r0, c0, 2, k), (n, k, dr, dc))
                board[r0][c0] = 1
                self.assertFalse(check_win(board, r0, c0, 1, k))

    def test_draw_and_undo(self):
        game = GomokuGame(3, 3)
        sequence = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 0),
                    (1, 2), (2, 1), (2, 0), (2, 2))
        for move in sequence:
            game.play(move)
        self.assertTrue(game.draw)
        self.assertEqual(game.winner, 0)
        with self.assertRaises(ValueError):
            game.play((0, 0))
        game.undo(1)
        self.assertFalse(game.finished)
        self.assertEqual(game.to_move, 1)
        game.play((2, 2))
        self.assertTrue(game.draw)

    def test_illegal_moves_do_not_change_game(self):
        game = GomokuGame(5, 3)
        for move in ([0, 0], (True, 0), (0.0, 1), (-1, 0), (5, 0), None):
            with self.assertRaises(ValueError):
                game.play(move)
            self.assertFalse(game.history)
            self.assertEqual(game.to_move, 1)
        game.play((0, 0))
        with self.assertRaises(ValueError):
            game.play((0, 0))


class ProtocolAndSearchTests(unittest.TestCase):
    def test_opponent_continuous_threat_screen_and_rollback(self):
        board = [[0] * 9 for _ in range(9)]
        for r, c in ((4, 1), (4, 2), (4, 3), (2, 4), (3, 4)):
            board[r][c] = 2
        for r, c in ((4, 0), (0, 0), (0, 2), (7, 7), (8, 0)):
            board[r][c] = 1
        ai = GomokuAI(1, 9, 5)
        ai.pos = _Position(ai, board)
        ai.vcf_nodes, ai._vcf_fail = 0, {}
        ai._deadline = time.perf_counter() + 0.5
        snapshot = copy.deepcopy((ai.pos.cells, ai.pos.hash, ai.pos.links, ai.pos.threats))
        ai._screen_root(1, [(8 * 9 + 8, 0, 0, 0), (4 * 9 + 4, 0, 0, 0)])
        self.assertIn(8 * 9 + 8, ai._root_excluded)
        self.assertNotIn(4 * 9 + 4, ai._root_excluded)
        self.assertEqual((ai.pos.cells, ai.pos.hash, ai.pos.links, ai.pos.threats), snapshot)

    def test_interrupted_opponent_screen_restores_position(self):
        board = [[0] * 9 for _ in range(9)]
        for c in (2, 3, 4):
            board[4][c] = 2
        ai = GomokuAI(1, 9, 5)
        ai.pos = _Position(ai, board)
        ai._deadline = time.perf_counter() + 0.5
        snapshot = copy.deepcopy((ai.pos.cells, ai.pos.hash, ai.pos.links, ai.pos.threats))
        def interrupted(*args):
            raise _SearchTimeout
        ai._vcf = interrupted
        with self.assertRaises(_SearchTimeout):
            ai._screen_root(1, [(8 * 9 + 8, 0, 0, 0)])
        self.assertEqual((ai.pos.cells, ai.pos.hash, ai.pos.links, ai.pos.threats), snapshot)
        self.assertFalse(ai._root_excluded)

    def test_root_does_not_accept_narrow_interior_cache_as_proof(self):
        board = [[0] * 9 for _ in range(9)]
        board[4][4] = 1
        ai = GomokuAI(2, 9, 5)
        pos = _Position(ai, board)
        key = pos.hash ^ ai.side_hash
        ai.tt[key] = (30, -ai.mate + 12, 0, 3 * 9 + 3)
        ai.max_depth = 2
        move = ai.get_move(board, (4, 4), 0.4)
        self.assertIsNone(validate_move(move, board))
        self.assertGreater(ai.last_stats["nodes"], 1)
        self.assertEqual(ai.last_stats["depth"], 2)
        self.assertLess(abs(ai.last_stats["score"]), ai.mate // 2)

    def test_setup_forks_against_independent_rules(self):
        for n, k in ((7, 4), (9, 5), (11, 7)):
            board = [[0] * n for _ in range(n)]
            row, start = n // 2, 2
            for c in range(start, start + k - 3):
                board[row][c] = 1
            ai = GomokuAI(1, n, k)
            ai.pos = _Position(ai, board)
            setups = ai._setup_forks(1)
            for move in ((row, start - 1), (row, start + k - 3), (row - 1, start), (n - 1, n - 1)):
                r, c = move
                board[r][c] = 1
                forks = 0
                for rr in range(n):
                    for cc in range(n):
                        if board[rr][cc]:
                            continue
                        board[rr][cc] = 1
                        if len(winning_moves(board, 1, k)) >= 2:
                            forks |= 1 << (rr * n + cc)
                        board[rr][cc] = 0
                board[r][c] = 0
                self.assertEqual(setups.get(r * n + c, 0), forks, (n, k, move))

    def test_win_and_block_every_assignment_configuration(self):
        for n in range(5, 16):
            for k in range(3, n + 1):
                for attacker in (1, 2):
                    board = [[0] * n for _ in range(n)]
                    for c in range(k - 1):
                        board[0][c] = attacker
                    original = copy.deepcopy(board)
                    for player in (attacker, 3 - attacker):
                        ai = GomokuAI(player, n, k)
                        move = ai.get_move(board, None, 0.05)
                        self.assertEqual(move, (0, k - 1), (n, k, player))
                        self.assertEqual(board, original)
                        self.assertIsNone(validate_move(move, board))

    def test_win_has_priority_over_block(self):
        board = [[0] * 9 for _ in range(9)]
        for c in range(3):
            board[0][c] = 1
            board[1][c] = 2
        ai = GomokuAI(1, 9, 4)
        self.assertEqual(ai.get_move(board, None, 0.1), (0, 3))

    def test_broken_diagonal_immediate_win(self):
        for n, k in ((5, 3), (9, 4), (15, 7), (17, 9)):
            board = [[0] * n for _ in range(n)]
            gap = k // 2
            for i in range(k):
                if i != gap:
                    board[i][n - 1 - i] = 1
            self.assertEqual(GomokuAI(1, n, k).get_move(board, None, 0.1), (gap, n - 1 - gap))

    def test_double_threat_is_real(self):
        for n, k in ((7, 3), (9, 4), (15, 5), (15, 7)):
            board = [[0] * n for _ in range(n)]
            row, start = n // 2, 2
            for c in range(start, start + k - 2):
                board[row][c] = 1
            ai = GomokuAI(1, n, k)
            move = ai.get_move(board, None, 0.2)
            board[move[0]][move[1]] = 1
            self.assertGreaterEqual(len(winning_moves(board, 1, k)), 2, (n, k, move))
            self.assertEqual(ai.last_stats["reason"], "proved double threat")

    def test_opponent_open_three_is_defended(self):
        board = [[0] * 15 for _ in range(15)]
        for c in (6, 7, 8):
            board[7][c] = 2
        board[2][2] = board[3][10] = board[10][3] = 1
        ai = GomokuAI(1, 15, 5)
        move = ai.get_move(board, None, 0.3)
        board[move[0]][move[1]] = 1
        self.assertIn(move, ((7, 5), (7, 9)))
        for r, row in enumerate(board):
            for c, cell in enumerate(row):
                if cell == 0:
                    board[r][c] = 2
                    self.assertLess(len(winning_moves(board, 2, 5)), 2, (move, r, c))
                    board[r][c] = 0

    def test_continuous_threat_proof(self):
        board = [[0] * 9 for _ in range(9)]
        for r, c in ((4, 1), (4, 2), (4, 3), (2, 4), (3, 4)):
            board[r][c] = 1
        for r, c in ((4, 0), (0, 0), (0, 2), (7, 7), (8, 0)):
            board[r][c] = 2
        ai = GomokuAI(1, 9, 5)
        self.assertEqual(ai.get_move(board, None, 0.5), (4, 4))
        self.assertEqual(ai.last_stats["reason"], "proved continuous threat win")
        # Independent rules check of the forcing continuation.
        board[4][4] = 1
        self.assertEqual(winning_moves(board, 1, 5), {(4, 5)})
        board[4][5] = 2
        self.assertFalse(winning_moves(board, 2, 5))
        board[5][4] = 1
        self.assertEqual(winning_moves(board, 1, 5), {(1, 4), (6, 4)})

    def test_incremental_state_against_fresh_rebuild(self):
        rng = random.Random(821)
        for n, k in ((5, 3), (9, 4), (15, 5), (15, 15)):
            ai = GomokuAI(1, n, k)
            board = [[0] * n for _ in range(n)]
            pos = _Position(ai, board)
            stack = []
            for turn in range(min(n * n, 50)):
                p = rng.choice([p for p, value in enumerate(pos.cells) if value == 0])
                side = 1 + turn % 2
                stack.append(pos.make(p, side))
                rebuilt = _Position(ai, [pos.cells[r * n:(r + 1) * n] for r in range(n)])
                self.assertEqual(pos.value, rebuilt.value)
                self.assertEqual(pos.counts, rebuilt.counts)
                self.assertEqual(pos.threats, rebuilt.threats)
                self.assertEqual(pos.hash, rebuilt.hash)
                self.assertEqual(pos.links, rebuilt.links)
                self.assertEqual(pos.link_squares, rebuilt.link_squares)
                self.assertEqual(pos.pair_counts, rebuilt.pair_counts)
            while stack:
                pos.unmake(stack.pop())
                rebuilt = _Position(ai, [pos.cells[r * n:(r + 1) * n] for r in range(n)])
                self.assertEqual(pos.value, rebuilt.value)
                self.assertEqual(pos.threats, rebuilt.threats)
                self.assertEqual(pos.links, rebuilt.links)
                self.assertEqual(pos.link_squares, rebuilt.link_squares)
            self.assertEqual(pos.occupied, 0)
            self.assertEqual(pos.hash, 0)

    def test_tactical_index_against_independent_rules(self):
        # Quiet positions with broken, crossing and board-edge threats.
        for n, k in ((5, 3), (9, 4), (15, 5), (15, 7), (17, 9)):
            board = [[0] * n for _ in range(n)]
            for j in range(k - 2):
                board[0][j] = 1
                board[j + 1][n - 2] = 2
            self.assertFalse(winning_moves(board, 1, k))
            self.assertFalse(winning_moves(board, 2, k))
            ai = GomokuAI(1, n, k)
            pos = _Position(ai, board)
            for side in (1, 2):
                for r in range(n):
                    for c in range(n):
                        if board[r][c]:
                            continue
                        board[r][c] = side
                        actual = winning_moves(board, side, k)
                        board[r][c] = 0
                        expected = sum(1 << (rr * n + cc) for rr, cc in actual)
                        self.assertEqual(pos.links[side].get(r * n + c, 0), expected,
                                         (n, k, side, r, c))

    def test_real_search_deadline_and_input_preservation(self):
        for n, k in ((5, 3), (9, 4), (15, 5), (15, 12)):
            board = [[0] * n for _ in range(n)]
            board[n // 2][n // 2] = 1
            original = copy.deepcopy(board)
            ai = GomokuAI(2, n, k)
            started = time.perf_counter()
            move = ai.get_move(board, (n // 2, n // 2), 0.1)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.1, (n, k, elapsed))
            self.assertIsNone(validate_move(move, board))
            self.assertEqual(board, original)
            self.assertGreater(ai.last_stats["nodes"], 0)

    def test_tiny_budget_and_single_empty(self):
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1
        ai = GomokuAI(2, 5, 3)
        for seconds in (0, 0.0001, 0.002):
            self.assertIsNone(validate_move(ai.get_move(board, None, seconds), board))
        almost_full = [[1] * 5 for _ in range(5)]
        almost_full[4][4] = 0
        self.assertEqual(ai.get_move(almost_full, None, 0.1), (4, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
