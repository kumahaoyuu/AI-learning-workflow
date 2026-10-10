#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol v1 arena with persistent subprocesses and enforceable wall timeouts.

Examples (run in assignment1):
    python arena.py gomoku_ai.py random_ai.py --board 15 --win 5 --games 2
    python arena.py gomoku_ai.py gemini_ai.py --time 5 --games 10 --quiet

Unlike an unkillable daemon thread, a timed-out AI process is terminated. Each
game gets fresh AI instances. The result includes actual per-move wall times.
Optional --json and --record write results only when explicitly requested.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

from gomoku_game import GomokuGame, validate_move


def load_ai(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("AI file does not exist: {}".format(path))
    name = "_gomoku_plugin_{}_{}".format(abs(hash(str(path))), time.time_ns())
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Also support classmates' modules importing siblings of their AI file.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    cls = getattr(module, "GomokuAI", None)
    if cls is None or not callable(cls):
        raise TypeError("AI file must define class GomokuAI")
    return cls, str(getattr(module, "NAME", None) or path.stem)


def _ai_worker(conn, path, player, n, k, seed, options):
    try:
        random.seed(seed)
        cls, name = load_ai(path)
        ai = cls(player, n, k)
        if hasattr(ai, "rng") and isinstance(ai.rng, random.Random):
            ai.rng.seed(seed)
        for key, value in options.items():
            if key not in ("max_depth", "candidate_limit", "enable_vcf"):
                raise ValueError("unsupported local experiment option: " + key)
            setattr(ai, key, value)
        conn.send(("ready", name))
        while True:
            message = conn.recv()
            if message is None:
                break
            board, last, seconds = message
            try:
                move = ai.get_move(board, last, seconds)
                conn.send(("move", move, getattr(ai, "last_stats", {}), time.perf_counter()))
            except BaseException as exc:
                conn.send(("error", "{}: {}".format(type(exc).__name__, exc), {}, time.perf_counter()))
                break
    except BaseException as exc:
        try:
            conn.send(("error", "{}: {}".format(type(exc).__name__, exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


class AIProcess:
    """One persistent AI per game. request/poll are suitable for a GUI loop."""

    def __init__(self, path, player, board_size, win_length, seed=0, options=None):
        self.path = str(Path(path).resolve())
        self.name = Path(path).stem
        self.pending = False
        self.last_stats = {}
        self._ready = False
        self._closed = False
        self._started = time.perf_counter()
        self._startup_limit = 15.0
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_ai_worker,
                                   args=(child, self.path, player, board_size,
                                         win_length, seed, options or {}), daemon=True)
        self.process.start()
        child.close()

    def poll_ready(self):
        if self._ready:
            return True, None
        try:
            if self.conn.poll():
                message = self.conn.recv()
                if message[0] == "ready":
                    self.name, self._ready = message[1], True
                    return True, None
                return False, "initialization exception: " + message[1]
        except (EOFError, OSError):
            return False, "AI process exited during initialization"
        if not self.process.is_alive():
            return False, "AI process exited during initialization"
        if time.perf_counter() - self._started >= self._startup_limit:
            self.close()
            return False, "initialization timeout"
        return False, None

    def wait_ready(self):
        while True:
            ready, error = self.poll_ready()
            if ready or error:
                return error
            self.conn.poll(0.01)

    def request(self, board, last_opponent_move, time_limit):
        if self.pending or not self._ready:
            raise RuntimeError("AI must be ready and idle")
        self._move_start = time.perf_counter()
        self._time_limit = float(time_limit)
        self.pending = True
        self.conn.send(([row[:] for row in board], last_opponent_move, time_limit))

    def poll(self):
        """None while thinking; otherwise (move, elapsed, error, diagnostics)."""
        if not self.pending:
            return None
        elapsed = time.perf_counter() - self._move_start
        try:
            if self.conn.poll():
                message = self.conn.recv()
                # perf_counter uses the same system clock across processes.
                # A GUI's polling interval must not be charged to the AI.
                elapsed = message[3] - self._move_start
                self.pending = False
                if elapsed > self._time_limit:
                    return None, elapsed, "timeout", {}
                if message[0] == "error":
                    return None, elapsed, "exception: " + message[1], {}
                self.last_stats = message[2]
                return message[1], elapsed, None, message[2]
        except (EOFError, OSError):
            self.pending = False
            return None, elapsed, "AI process exited", {}
        if elapsed >= self._time_limit:
            self.pending = False
            self.close()
            return None, elapsed, "timeout", {}
        if not self.process.is_alive():
            self.pending = False
            return None, elapsed, "AI process exited", {}
        return None

    def get_move(self, board, last_opponent_move, time_limit):
        self.request(board, last_opponent_move, time_limit)
        while True:
            result = self.poll()
            if result is not None:
                return result
            # Wait for pipe readiness, rather than burning CPU in a busy loop.
            remaining = self._time_limit - (time.perf_counter() - self._move_start)
            self.conn.poll(max(0.0, min(0.01, remaining)))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.2)
        finally:
            self.conn.close()


def play_game(ai_black, ai_white, n=15, k=5, seconds=5.0, seed=0,
              quiet=False, black_options=None, white_options=None, on_move=None):
    game = GomokuGame(n, k)
    workers = {}
    labels = {1: "{} [X]".format(Path(ai_black).stem),
              2: "{} [O]".format(Path(ai_white).stem)}
    records = []
    try:
        for side, path, options in ((1, ai_black, black_options), (2, ai_white, white_options)):
            try:
                worker = AIProcess(path, side, n, k, seed + side, options)
                workers[side] = worker
                error = worker.wait_ready()
            except Exception as exc:
                error = "initialization exception: {}".format(exc)
            if error:
                game.forfeit(side, error)
                break
            labels[side] = worker.name
        if not quiet:
            print("{} (X) vs {} (O); N={}, K={}, {:.3f}s/move".format(labels[1], labels[2], n, k, seconds))
            print(game.format_board())
        while not game.finished:
            side = game.to_move
            try:
                move, elapsed, error, stats = workers[side].get_move(
                    game.board, game.last_move(3 - side), seconds)
            except Exception as exc:
                move, elapsed, error, stats = None, 0.0, "worker error: {}".format(exc), {}
            error = error or validate_move(move, game.board)
            if error:
                game.forfeit(side, "{}: {}".format(labels[side], error))
                records.append({"player": side, "move": move, "elapsed": elapsed, "error": error})
                break
            game.play(move, elapsed)
            record = {"player": side, "move": move, "elapsed": elapsed, "stats": stats}
            records.append(record)
            if on_move:
                on_move(game, record)
            if not quiet:
                print("[{}] {} -> {} ({:.4f}s)".format(len(game.history), labels[side], move, elapsed))
                print(game.format_board())
        return {"board_size": n, "win_length": k, "time_limit": seconds,
                "seed": seed, "black": str(Path(ai_black).resolve()),
                "white": str(Path(ai_white).resolve()), "names": labels,
                "winner": game.winner, "reason": game.reason,
                "moves": len(game.history), "records": records}
    finally:
        for worker in workers.values():
            worker.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gomoku Protocol v1 tournament arena")
    parser.add_argument("ai_a")
    parser.add_argument("ai_b")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", type=Path, help="optional result JSON path")
    parser.add_argument("--record", type=Path, help="optional JSON-lines game records")
    args = parser.parse_args(argv)
    if args.board < 3 or not 3 <= args.win <= args.board:
        parser.error("require N >= 3 and 3 <= K <= N")
    if args.time <= 0 or not args.time < float("inf") or args.games < 1:
        parser.error("time must be finite and positive; games must be >= 1")
    for path in (args.ai_a, args.ai_b):
        if not Path(path).is_file():
            parser.error("AI file does not exist: " + path)
    results, score = [], [0.0, 0.0]
    for index in range(args.games):
        swap = index % 2
        black, white = (args.ai_b, args.ai_a) if swap else (args.ai_a, args.ai_b)
        result = play_game(black, white, args.board, args.win, args.time,
                           args.seed + index * 17, args.quiet)
        results.append(result)
        winner = result["winner"]
        if winner:
            winning_index = (winner - 1) ^ swap
            score[winning_index] += 1
            outcome = "{} wins".format("B" if winning_index else "A")
        else:
            score[0] += 0.5
            score[1] += 0.5
            outcome = "Draw"
        times = [r["elapsed"] for r in result["records"]]
        print("Game {}: {} [{}], {} moves, mean {:.4f}s/move".format(
            index + 1, outcome, result["reason"], result["moves"], sum(times) / max(1, len(times))))
    print("A ({}) {:.1f} : {:.1f} B ({})".format(Path(args.ai_a).stem, score[0], score[1], Path(args.ai_b).stem))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"score": score, "games": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
