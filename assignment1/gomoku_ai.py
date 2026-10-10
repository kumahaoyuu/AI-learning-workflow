#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Codex Aegis 3: a standalone Gomoku AI Protocol v1 implementation.

Only the Python standard library is used. N and K are constructor parameters;
overlines win, and there are no forbidden moves. The input board is never edited.

Algorithm
---------
Iterative-deepening negamax (the zero-sum form of minimax), alpha-beta / principal
variation search, a Zobrist transposition table, killer/history move ordering,
verified late-move reductions, and threat quiescence. A bounded continuous-four
search looks for *proved* forcing wins before positional search. All K-cell
windows, threats, and evaluation weights are constructed at runtime for (N, K).
Immediate wins and mandatory blocks are exact and bypass candidate pruning.
Incremental two-empty-window links avoid scanning quiet squares in tactical
search. Quiescence transpositions reuse completed searches with valid bounds.
Three-empty windows order threat-building moves. The wider root never accepts
a selective interior-node cache cutoff; proved opponent VCFs screen root moves.
General positional search is selective, so its scores are heuristic, not proofs.

Configuration for local experiments: set max_depth, candidate_limit or
enable_vcf on an instance. The required constructor and get_move stay unchanged.
last_stats contains diagnostics; nothing is printed by the protocol entry point.
"""

import math
import random
import time

NAME = "Codex Aegis 3"
_EXACT, _LOWER, _UPPER = 0, 1, 2


class _SearchTimeout(Exception):
    pass


def _indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class _Position:
    """Bitboards plus incrementally maintained K-window scores and win squares."""

    def __init__(self, ai, board):
        self.ai = ai
        self.cells = [int(x) for row in board for x in row]
        self.bits = [0, 0, 0]
        self.occupied = 0
        self.frontier = 0
        self.hash = 0
        for p, side in enumerate(self.cells):
            if side:
                bit = 1 << p
                self.bits[side] |= bit
                self.occupied |= bit
                self.frontier |= ai.neighbours[p]
                self.hash ^= ai.zobrist[side][p]
        self.counts = [None, [], []]
        self.threats = [None, {}, {}]
        # Each unblocked K-window with K-2 stones has two empty endpoints.
        # links[side][p] is exactly the set of wins created by playing p.
        self.links = [None, {}, {}]
        self.pair_counts = [None, {}, {}]
        self.link_squares = [0, 0, 0]
        self.value = 0
        self.winner = 0
        for mask in ai.windows:
            a = (mask & self.bits[1]).bit_count()
            b = (mask & self.bits[2]).bit_count()
            self.counts[1].append(a)
            self.counts[2].append(b)
            self.value += ai.weights[a] if not b else 0
            self.value -= ai.weights[b] if not a else 0
            if a == ai.win_length:
                self.winner = 1
            if b == ai.win_length:
                self.winner = 2
            for side, mine, other in ((1, a, b), (2, b, a)):
                if mine == ai.win_length - 1 and not other:
                    p = (mask & ~self.occupied).bit_length() - 1
                    self.threats[side][p] = self.threats[side].get(p, 0) + 1
                elif mine == ai.win_length - 2 and not other:
                    self._change_pair(side, mask & ~self.occupied, 1)

    def _change_pair(self, side, empty, delta):
        first = empty & -empty
        second = empty ^ first
        p, q = first.bit_length() - 1, second.bit_length() - 1
        pairs, links = self.pair_counts[side], self.links[side]
        key = (p, q)
        old = pairs.get(key, 0)
        count = old + delta
        if count:
            pairs[key] = count
            if not old:
                links[p] = links.get(p, 0) | second
                links[q] = links.get(q, 0) | first
                self.link_squares[side] |= empty
        else:
            del pairs[key]
            for square, bit in ((p, second), (q, first)):
                targets = links[square] ^ bit
                if targets:
                    links[square] = targets
                else:
                    del links[square]
                    self.link_squares[side] &= ~(1 << square)

    def _remove_threats(self, incident):
        ai = self.ai
        for w in incident:
            a, b = self.counts[1][w], self.counts[2][w]
            for side, mine, other in ((1, a, b), (2, b, a)):
                if mine == ai.win_length - 1 and not other:
                    p = (ai.windows[w] & ~self.occupied).bit_length() - 1
                    d = self.threats[side]
                    count = d[p] - 1
                    if count:
                        d[p] = count
                    else:
                        del d[p]
                elif mine == ai.win_length - 2 and not other:
                    self._change_pair(side, ai.windows[w] & ~self.occupied, -1)

    def _add_threats(self, incident):
        ai = self.ai
        for w in incident:
            a, b = self.counts[1][w], self.counts[2][w]
            for side, mine, other in ((1, a, b), (2, b, a)):
                if mine == ai.win_length - 1 and not other:
                    p = (ai.windows[w] & ~self.occupied).bit_length() - 1
                    d = self.threats[side]
                    d[p] = d.get(p, 0) + 1
                elif mine == ai.win_length - 2 and not other:
                    self._change_pair(side, ai.windows[w] & ~self.occupied, 1)

    def make(self, p, side):
        ai = self.ai
        incident = ai.incident[p]
        undo = (p, side, self.frontier, self.winner, self.value)
        self._remove_threats(incident)
        counts, others, weights = self.counts[side], self.counts[3 - side], ai.weights
        delta = 0
        for w in incident:
            mine, other = counts[w], others[w]
            # Own gain; an opponent-only window disappears when we block it.
            if not other:
                delta += weights[mine + 1] - weights[mine]
            elif not mine:
                delta += weights[other]
            counts[w] += 1
            if counts[w] == ai.win_length:
                self.winner = side
        self.value += delta if side == 1 else -delta
        bit = 1 << p
        self.bits[side] |= bit
        self.occupied |= bit
        self.frontier |= ai.neighbours[p]
        self.cells[p] = side
        self.hash ^= ai.zobrist[side][p]
        self._add_threats(incident)
        return undo

    def unmake(self, undo):
        p, side, self.frontier, self.winner, self.value = undo
        incident = self.ai.incident[p]
        self._remove_threats(incident)
        for w in incident:
            self.counts[side][w] -= 1
        bit = 1 << p
        self.bits[side] ^= bit
        self.occupied ^= bit
        self.cells[p] = 0
        self.hash ^= self.ai.zobrist[side][p]
        self._add_threats(incident)

    def candidates(self):
        mask = self.frontier & ~self.occupied & self.ai.all_cells
        return mask or (self.ai.all_cells & ~self.occupied)


class GomokuAI:
    """Protocol v1. Import this single file to play against any compatible AI."""

    def __init__(self, player_id: int, board_size: int, win_length: int):
        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")
        if not isinstance(board_size, int) or board_size < 3:
            raise ValueError("board_size must be an integer >= 3")
        if not isinstance(win_length, int) or not 3 <= win_length <= board_size:
            raise ValueError("win_length must satisfy 3 <= K <= N")
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        n, k = board_size, win_length
        self.all_cells = (1 << (n * n)) - 1
        self.windows = []
        self.incident = [[] for _ in range(n * n)]
        for r in range(n):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if not (0 <= r + (k - 1) * dr < n and
                            0 <= c + (k - 1) * dc < n):
                        continue
                    cells = [(r + j * dr) * n + c + j * dc for j in range(k)]
                    w = len(self.windows)
                    self.windows.append(sum(1 << p for p in cells))
                    for p in cells:
                        self.incident[p].append(w)
        # Integer arithmetic, including for K > 15; terminal values dominate.
        self.weights = [0] + [7 ** i for i in range(1, k + 1)]
        self.mate = max(10 ** 12, self.weights[k] * n * n * 1000)
        self.infinity = 4 * self.mate
        self.neighbours = []
        self.centrality = []
        for r in range(n):
            for c in range(n):
                self.neighbours.append(sum(
                    1 << (rr * n + cc)
                    for rr in range(max(0, r - 2), min(n, r + 3))
                    for cc in range(max(0, c - 2), min(n, c + 3))
                    if (rr, cc) != (r, c)))
                self.centrality.append(2 * n - abs(2 * r - n + 1) - abs(2 * c - n + 1))
        rng = random.Random(0xAE615 + n * 131 + k)
        self.zobrist = [None] + [[rng.getrandbits(64) for _ in range(n * n)]
                                for _ in (1, 2)]
        self.side_hash = rng.getrandbits(64)
        self.tt = {}
        self.qtt = {}
        self.history = [[0] * (n * n) for _ in range(3)]
        self.killers = {}
        self.max_depth = 64
        self.candidate_limit = 32
        self.enable_vcf = True
        self.last_stats = {}
        self._deadline = 0.0
        self._root_excluded = set()

    def _check_time(self):
        if time.perf_counter() >= self._deadline:
            raise _SearchTimeout

    def _setup_forks(self, side):
        """New next-turn forks created by a quiet move, derived from K-windows.

        A K-3 window has three empties p,q,t. Playing p adds the q->t link.
        Merge all those links with current links to count distinct fork squares.
        This is move ordering only, not a claim of a forced win.
        """
        if self.win_length < 4:
            return {}
        pos, pending = self.pos, {}
        mine, enemy = pos.counts[side], pos.counts[3 - side]
        for w, mask in enumerate(self.windows):
            if enemy[w] or mine[w] != self.win_length - 3:
                continue
            empty = mask & ~pos.occupied
            squares = list(_indices(empty))
            for p in squares:
                if pos.links[side].get(p):
                    continue
                future = pending.setdefault(p, {})
                rest = empty ^ (1 << p)
                for q in squares:
                    if q != p:
                        future[q] = future.get(q, 0) | (rest ^ (1 << q))
        result = {}
        for p, futures in pending.items():
            forks = 0
            for q, new in futures.items():
                old = pos.links[side].get(q, 0)
                targets = old | new
                if not old & (old - 1) and targets & (targets - 1):
                    forks |= 1 << q
            if forks:
                result[p] = forks
        return result

    def _rank(self, side, ply=0, preferred=None, tactical=False):
        """(square, ordering score, own next-win mask, opponent next-win mask)."""
        self._check_time()
        pos, ai = self.pos, self
        counts, others = pos.counts[side], pos.counts[3 - side]
        weights = ai.weights
        rank = []
        killers = self.killers.get(ply, ())
        tactical_unit = weights[ai.win_length] * 100
        relevant = pos.link_squares[1] | pos.link_squares[2]
        candidates = relevant if tactical else pos.candidates() | relevant
        own_links, enemy_links = pos.links[side], pos.links[3 - side]
        own_setups = {} if tactical else self._setup_forks(side)
        enemy_setups = {} if tactical else self._setup_forks(3 - side)
        for p in _indices(candidates):
            attack, defence = 0, 0
            own_targets, enemy_targets = own_links.get(p, 0), enemy_links.get(p, 0)
            for w in ai.incident[p]:
                a, b = counts[w], others[w]
                if not b:
                    attack += weights[a + 1] - weights[a]
                if not a:
                    defence += weights[b + 1] - weights[b]
            own_fork = bool(own_targets & (own_targets - 1))
            enemy_fork = bool(enemy_targets & (enemy_targets - 1))
            category = 60 if own_fork else 50 if enemy_fork else 40 if own_targets else 30 if enemy_targets else 0
            if not category:
                attack_setups = own_setups.get(p, 0).bit_count()
                defence_setups = enemy_setups.get(p, 0).bit_count()
                category = (20 if attack_setups >= 2 else 0) + (19 if defence_setups >= 2 else 0)
                if not category:
                    category = (10 if attack_setups else 0) + (9 if defence_setups else 0)
            score = category * tactical_unit + attack * 10 + defence * 11
            score += ai.centrality[p] + min(self.history[side][p], tactical_unit // 100)
            if p in killers:
                score += tactical_unit // 10
            if p == preferred:
                score += 100 * tactical_unit
            rank.append((p, score, own_targets, enemy_targets))
        rank.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return rank

    def _tactical_moves(self, rank):
        """Return a proved fork, and exact defences against opponent forks.

        A hypothetical fork at p leaves >=2 distinct next-turn win squares.
        With two squares, only p or one of those squares can stop that fork.
        With >=3 squares, only p can stop it in one move. Intersect those sets
        across all forks; forcing counterattacks are also legal defences.
        """
        fork, defence_mask = None, self.all_cells
        threatened = False
        for p, _, own, enemy in rank:
            if own & (own - 1) and fork is None:
                fork = p
            if enemy & (enemy - 1):
                threatened = True
                defence_mask &= (1 << p) | (enemy if enemy.bit_count() == 2 else 0)
        if threatened:
            return fork, [m for m in rank if m[2] or (defence_mask >> m[0]) & 1]
        return fork, None

    def _evaluate(self, side):
        value = self.pos.value if side == 1 else -self.pos.value
        # Keep nonterminal estimates separated from distance-to-mate scores.
        limit = self.mate // 4
        return max(-limit, min(limit, value))

    def _pack_score(self, score, ply):
        if score > self.mate // 2:
            return score + ply
        if score < -self.mate // 2:
            return score - ply
        return score

    def _unpack_score(self, score, ply):
        if score > self.mate // 2:
            return score - ply
        if score < -self.mate // 2:
            return score + ply
        return score

    def _terminal(self, side, ply):
        pos = self.pos
        if pos.winner:
            return (self.mate - ply) if pos.winner == side else (-self.mate + ply)
        if pos.threats[side]:
            return self.mate - ply - 1
        if len(pos.threats[3 - side]) > 1:
            return -self.mate + ply + 2
        if pos.occupied == self.all_cells:
            return 0
        return None

    def _quiescence(self, alpha, beta, side, ply, remaining):
        self.nodes += 1
        self.qnodes += 1
        self._check_time()
        terminal = self._terminal(side, ply)
        if terminal is not None:
            return terminal
        original_alpha, original_beta = alpha, beta
        key = self.pos.hash ^ (self.side_hash if side == 2 else 0)
        entry = self.qtt.get(key)
        if entry and entry[0] == remaining:
            score = self._unpack_score(entry[1], ply)
            if entry[2] == _EXACT or (entry[2] == _LOWER and score >= beta) or (entry[2] == _UPPER and score <= alpha):
                self.qtt_hits += 1
                return score
        enemy_wins = self.pos.threats[3 - side]
        if enemy_wins:
            moves = [(next(iter(enemy_wins)), 0, 0, 0)]
            forced = True
        else:
            rank = self._rank(side, ply, tactical=True)
            fork, defences = self._tactical_moves(rank)
            if fork is not None:
                return self.mate - ply - 3
            forced = defences is not None
            if forced and not defences:
                return -self.mate + ply + 4
            moves = defences if forced else [m for m in rank if m[2]]
        static = self._evaluate(side)
        if remaining <= 0:
            return static
        best = -self.infinity if forced else static
        if not forced:
            if static >= beta:
                return static
            alpha = max(alpha, static)
        for p, _, _, _ in moves:
            undo = self.pos.make(p, side)
            try:
                score = -self._quiescence(-beta, -alpha, 3 - side, ply + 1, remaining - 1)
            finally:
                self.pos.unmake(undo)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        flag = _UPPER if best <= original_alpha else _LOWER if best >= original_beta else _EXACT
        if len(self.qtt) < 100000:
            self.qtt[key] = (remaining, self._pack_score(best, ply), flag)
        return best

    def _search(self, depth, alpha, beta, side, ply):
        self.nodes += 1
        self._check_time()
        terminal = self._terminal(side, ply)
        if terminal is not None:
            if ply == 0 and self.pos.threats[side]:
                self._root_best = min(self.pos.threats[side])
            return terminal
        if depth <= 0:
            return self._quiescence(alpha, beta, side, ply, min(12, self.win_length * 2 + 2))
        original_alpha, original_beta = alpha, beta
        key = self.pos.hash ^ (self.side_hash if side == 2 else 0)
        entry = self.tt.get(key)
        preferred = entry[3] if entry else None
        if ply != 0 and entry and entry[0] >= depth:
            self.tt_hits += 1
            score = self._unpack_score(entry[1], ply)
            flag = entry[2]
            if flag == _EXACT or (flag == _LOWER and score >= beta) or (flag == _UPPER and score <= alpha):
                if ply == 0:
                    self._root_best = entry[3]
                return score
        enemy_wins = self.pos.threats[3 - side]
        if enemy_wins:
            moves = [(next(iter(enemy_wins)), 0, 0, 0)]
            forcing = True
        else:
            rank = self._rank(side, ply, preferred)
            fork, defences = self._tactical_moves(rank)
            if fork is not None:
                if ply == 0:
                    self._root_best = fork
                return self.mate - ply - 3
            if defences is not None:
                if not defences:
                    return -self.mate + ply + 4
                moves, forcing = defences, True
            else:
                if ply == 0 and self._root_excluded:
                    safe = [m for m in rank if m[0] not in self._root_excluded]
                    rank = safe or rank
                width = self.candidate_limit if ply == 0 else min(self.candidate_limit, 14 if depth <= 2 else 12)
                # Keep every forcing move even outside the positional beam.
                moves = rank[:width] + [m for m in rank[width:] if m[2] or m[3]]
                forcing = False
        if ply == 0 and self._root_excluded:
            safe = [m for m in moves if m[0] not in self._root_excluded]
            moves = safe or moves
        best, best_move = -self.infinity, moves[0][0]
        for index, (p, _, own, enemy) in enumerate(moves):
            undo = self.pos.make(p, side)
            try:
                if index == 0:
                    score = -self._search(depth - 1, -beta, -alpha, 3 - side, ply + 1)
                else:
                    reduction = 0
                    if depth >= 3 and index >= 4 and not forcing and not own and not enemy:
                        reduction = 1 + int(depth >= 6 and index >= 10)
                    score = -self._search(depth - 1 - reduction, -alpha - 1, -alpha, 3 - side, ply + 1)
                    if reduction and score > alpha:
                        score = -self._search(depth - 1, -alpha - 1, -alpha, 3 - side, ply + 1)
                    if alpha < score < beta:
                        score = -self._search(depth - 1, -beta, -alpha, 3 - side, ply + 1)
            finally:
                self.pos.unmake(undo)
            if score > best:
                best, best_move = score, p
                if ply == 0:
                    self._root_best = p
            alpha = max(alpha, score)
            if alpha >= beta:
                if not own and not enemy:
                    killers = self.killers.get(ply, ())
                    self.killers[ply] = (p,) + tuple(m for m in killers if m != p)[:1]
                    self.history[side][p] += depth * depth
                break
        flag = _UPPER if best <= original_alpha else _LOWER if best >= original_beta else _EXACT
        if len(self.tt) < 150000:
            self.tt[key] = (depth, self._pack_score(best, ply), flag, best_move)
        return best

    def _vcf(self, side, remaining):
        """AND/OR proof search over continuous, immediately forcing threats.

        Opponent's only legal escape from one win square is blocking it; two
        distinct win squares prove a win. Failed search means 'not proved'.
        """
        self.vcf_nodes += 1
        self._check_time()
        pos = self.pos
        if pos.threats[side]:
            return min(pos.threats[side])
        if remaining <= 0 or len(pos.threats[3 - side]) > 1:
            return None
        key = pos.hash ^ (self.side_hash if side == 2 else 0)
        if self._vcf_fail.get(key, -1) >= remaining:
            return None
        rank = self._rank(side, tactical=True)
        required = next(iter(pos.threats[3 - side]), None)
        for p, _, targets, _ in rank:
            if required is not None and p != required:
                continue
            if not targets:
                continue
            undo = pos.make(p, side)
            try:
                if pos.threats[3 - side]:
                    continue
                if len(pos.threats[side]) >= 2:
                    return p
                if not pos.threats[side]:
                    continue
                block = next(iter(pos.threats[side]))
                reply = pos.make(block, 3 - side)
                try:
                    if not pos.winner and self._vcf(side, remaining - 2) is not None:
                        return p
                finally:
                    pos.unmake(reply)
            finally:
                pos.unmake(undo)
        self._vcf_fail[key] = remaining
        return None

    def _screen_root(self, side, rank):
        """Reject only moves followed by a proved opponent continuous-four win.

        A timeout or a failed proof is inconclusive, so those moves stay legal.
        Every make is rolled back even if the proof deadline interrupts it.
        """
        if not self.pos.link_squares[3 - side]:
            return
        self._vcf_fail = {}
        for p, _, _, _ in rank:
            self._check_time()
            undo = self.pos.make(p, side)
            try:
                if self._vcf(3 - side, 18) is not None:
                    self._root_excluded.add(p)
            finally:
                self.pos.unmake(undo)

    def get_move(self, board, last_opponent_move, time_limit):
        """Return a tuple of two native ints under a conservative wall deadline."""
        started = time.perf_counter()
        n = self.board_size
        if len(board) != n or any(len(row) != n for row in board):
            raise ValueError("board shape differs from board_size")
        if any(v not in (0, 1, 2) for row in board for v in row):
            raise ValueError("board must contain only 0, 1, 2")
        empties = [r * n + c for r in range(n) for c in range(n) if board[r][c] == 0]
        if not empties:
            raise ValueError("no legal move on a full board (game already finished)")
        best = max(empties, key=lambda p: (self.centrality[p], -p))
        self.nodes = self.qnodes = self.tt_hits = self.vcf_nodes = self.qtt_hits = 0
        self._root_excluded = set()
        self.last_stats = {"depth": 0, "nodes": 0, "score": 0, "reason": "fallback"}

        def finish(p, reason, depth=0, score=0):
            self.last_stats = {"depth": depth, "nodes": self.nodes, "qnodes": self.qnodes,
                               "tt_hits": self.tt_hits, "vcf_nodes": self.vcf_nodes,
                               "qtt_hits": self.qtt_hits,
                               "screened_losing_moves": len(self._root_excluded),
                               "score": score, "reason": reason,
                               "elapsed": time.perf_counter() - started}
            return divmod(int(p), n)

        try:
            seconds = float(time_limit)
        except (TypeError, ValueError):
            seconds = 0.0
        if not math.isfinite(seconds) or seconds <= 0.003:
            return finish(best, "short-budget fallback")
        if len(empties) == n * n:
            return finish(best, "central opening")
        if len(empties) == 1:
            return finish(empties[0], "only legal move")
        # Leave 12% plus a small fixed reserve for the judge and Python scheduling.
        self._deadline = started + max(0.0, seconds * 0.88 - min(0.012, seconds * 0.08))
        self.pos = _Position(self, board)
        side = self.player_id
        if self.pos.threats[side]:
            return finish(min(self.pos.threats[side]), "immediate win", score=self.mate - 1)
        if self.pos.threats[3 - side]:
            # Multiple threats cannot all be blocked; pick the strongest block.
            best = max(self.pos.threats[3 - side],
                       key=lambda p: (self.pos.threats[3 - side][p], self.centrality[p], -p))
            return finish(best, "mandatory block")
        depth_done, score = 0, 0
        try:
            rank = self._rank(side)
            best = rank[0][0]
            fork, defences = self._tactical_moves(rank)
            if fork is not None:
                return finish(fork, "proved double threat", score=self.mate - 3)
            if defences:
                best = defences[0][0]
            if len(self.tt) >= 120000:
                self.tt.clear()
            if len(self.qtt) >= 80000:
                self.qtt.clear()
            self.killers.clear()
            self.history = [[value // 2 for value in row] for row in self.history]
            if self.enable_vcf and seconds >= 0.08:
                total_deadline = self._deadline
                self._deadline = min(total_deadline, time.perf_counter() + min(0.65, seconds * 0.16))
                self._vcf_fail = {}
                try:
                    for proof_depth in (4, 8, 12, 18, 24):
                        winning = self._vcf(side, proof_depth)
                        if winning is not None:
                            return finish(winning, "proved continuous threat win", score=self.mate - proof_depth)
                except _SearchTimeout:
                    pass
                finally:
                    self._deadline = total_deadline
                self._deadline = min(total_deadline, time.perf_counter() + min(0.45, seconds * 0.10))
                try:
                    self._screen_root(side, rank)
                except _SearchTimeout:
                    pass
                finally:
                    self._deadline = total_deadline
            for depth in range(1, min(self.max_depth, len(empties)) + 1):
                self._check_time()
                self._root_best = best
                if depth >= 3 and abs(score) < self.mate // 2:
                    margin = self.weights[max(1, self.win_length - 2)] * 10
                    low, high = score - margin, score + margin
                    value = self._search(depth, low, high, side, 0)
                    if value <= low or value >= high:
                        value = self._search(depth, -self.infinity, self.infinity, side, 0)
                else:
                    value = self._search(depth, -self.infinity, self.infinity, side, 0)
                if self._root_best is not None:
                    best = self._root_best
                depth_done, score = depth, value
                # A selective interior search may miss a quiet root defence.
                if value > self.mate // 2:
                    break
        except _SearchTimeout:
            pass
        return finish(best, "iterative alpha-beta", depth_done, score)


if __name__ == "__main__":
    # A protocol smoke check; use main.py for the complete game.
    ai = GomokuAI(1, 15, 5)
    print(NAME, ai.get_move([[0] * 15 for _ in range(15)], None, 5.0))
