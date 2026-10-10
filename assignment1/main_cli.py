#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console game supporting human-human, human-ai and ai-ai.

    python main_cli.py --mode human-ai --board 15 --win 5 --human-color white
    python main_cli.py --mode ai-ai --ai-white gemini_ai.py --time 5

Coordinates are ZERO-based: enter 'row col'. Commands: undo, restart, quit.
No third-party packages are required.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from arena import AIProcess, play_game
from gomoku_game import GomokuGame, validate_move

BASE = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis generalized Gomoku")
    parser.add_argument("--mode", choices=("human-human", "human-ai", "ai-ai"), default="human-ai")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--time", type=float, default=5.0, help="AI wall time budget per move (seconds)")
    parser.add_argument("--human-color", choices=("black", "white"), default="black")
    parser.add_argument("--ai-black", type=Path, default=BASE / "gomoku_ai.py")
    parser.add_argument("--ai-white", type=Path, default=None,
                        help="external AI file; defaults to Aegis in human-ai, random in ai-ai")
    parser.add_argument("--games", type=int, default=1, help="ai-ai games, swapping colors")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--record", type=Path, help="optional JSON game-record file")
    return parser


def validate_args(parser, args):
    if args.board < 3 or not 3 <= args.win <= args.board:
        parser.error("require N >= 3 and 3 <= K <= N")
    if args.time <= 0 or not args.time < float("inf") or args.games < 1:
        parser.error("time must be finite and positive; games >= 1")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.mode == "ai-ai":
        white = args.ai_white or BASE / "random_ai.py"
        score = [0.0, 0.0]
        results = []
        for index in range(args.games):
            swap = index % 2
            black_path, white_path = (white, args.ai_black) if swap else (args.ai_black, white)
            result = play_game(black_path, white_path, args.board, args.win, args.time,
                               args.seed + index * 17, args.quiet)
            results.append(result)
            if result["winner"]:
                score[(result["winner"] - 1) ^ swap] += 1
            else:
                score[0] += 0.5
                score[1] += 0.5
            print("Game {}: {} [{}], {} moves".format(index + 1, result["winner"], result["reason"], result["moves"]))
        print("AI A {:.1f} : {:.1f} AI B".format(*score))
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    human_side = 1 if args.human_color == "black" else 2
    ai_paths = {} if args.mode == "human-human" else {
        3 - human_side: (args.ai_black if human_side == 2 else args.ai_white or BASE / "gomoku_ai.py")}
    game = GomokuGame(args.board, args.win)
    workers = {}

    def close_workers():
        for worker in workers.values():
            worker.close()
        workers.clear()

    def start_workers():
        close_workers()
        for side, path in ai_paths.items():
            worker = AIProcess(path, side, args.board, args.win, args.seed + side)
            workers[side] = worker
            error = worker.wait_ready()
            if error:
                game.forfeit(side, error)
                print(error)

    try:
        start_workers()
        print("N={}, K={}. X is black; O is white. Coordinates start at 0.".format(args.board, args.win))
        print("Enter row col; commands: undo / restart / quit.")
        while not game.finished:
            print(game.format_board())
            side = game.to_move
            if side in workers:
                worker = workers[side]
                print("{} is thinking...".format(worker.name), flush=True)
                move, elapsed, error, stats = worker.get_move(game.board, game.last_move(3 - side), args.time)
                error = error or validate_move(move, game.board)
                if error:
                    game.forfeit(side, error)
                    break
                game.play(move, elapsed)
                print("{} -> {}, {:.3f}s, depth {}".format(worker.name, move, elapsed, stats.get("depth", "-")))
                continue
            raw = input("{} > ".format("X" if side == 1 else "O")).strip().lower()
            if raw in ("quit", "q", "exit"):
                return
            if raw in ("restart", "new"):
                game.reset()
                start_workers()
                continue
            if raw in ("undo", "u"):
                game.undo(2 if ai_paths else 1)
                start_workers()
                continue
            try:
                values = raw.replace(",", " ").split()
                if len(values) != 2:
                    raise ValueError("enter exactly two integers: row col")
                game.play(tuple(map(int, values)))
            except ValueError as exc:
                print("Invalid move:", exc)
        print(game.format_board())
        print("Draw" if game.draw else "{} wins".format("X" if game.winner == 1 else "O"), "[{}]".format(game.reason))
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            records = [{"player": m.player, "move": (m.row, m.col), "elapsed": m.elapsed} for m in game.history]
            args.record.write_text(json.dumps({"board_size": args.board, "win_length": args.win,
                                              "winner": game.winner, "reason": game.reason,
                                              "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    except (KeyboardInterrupt, EOFError):
        print("\nGame closed.")
    finally:
        close_workers()


if __name__ == "__main__":
    mp.freeze_support()
    main()
