#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible assignment experiments. Results are measured, never fabricated.

Default: 40 random-baseline games per configuration (20 in each color), four
self-play games, at least two configurations, actual AI wall times and length.
The default 0.2s budget is for convenient experiments; use --time 5 for the
official budget. Configuration, seeds and budget are printed with the results.
An optional ablation compares depth 2, depth 4 and full iterative search.

    python experiments.py --configs 9:4 15:5 --games 40 --self-games 4 --time .2
    python experiments.py --opponent gemini_ai.py --games 20 --time 5
    python experiments.py --ablation --json results.json
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time

from arena import play_game

BASE = Path(__file__).resolve().parent


def summarize(results, candidate_is_black):
    wins = losses = draws = forfeits = 0
    times, lengths, depths = [], [], []
    for result, black in zip(results, candidate_is_black):
        side = 1 if black else 2
        winner = result["winner"]
        wins += int(winner == side)
        losses += int(winner != 0 and winner != side)
        draws += int(winner == 0)
        forfeits += int(result["reason"] not in ("board full", "{}-in-a-row".format(result["win_length"])))
        lengths.append(result["moves"])
        for record in result["records"]:
            if record["player"] == side:
                times.append(record["elapsed"])
                depths.append(record.get("stats", {}).get("depth", 0))
    total = len(results)
    return {"games": total, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / max(1, total), "forfeit_games": forfeits,
            "mean_move_seconds": statistics.mean(times) if times else 0,
            "max_move_seconds": max(times, default=0),
            "mean_game_moves": statistics.mean(lengths) if lengths else 0,
            "mean_completed_depth": statistics.mean(depths) if depths else 0}


def run_series(opponent, n, k, seconds, games, seed, options=None, quiet_progress=False):
    results, colors = [], []
    for index in range(games):
        black = index % 2 == 0
        a, b = (BASE / "gomoku_ai.py", opponent) if black else (opponent, BASE / "gomoku_ai.py")
        result = play_game(a, b, n, k, seconds, seed + index * 17, True,
                           options if black else None, None if black else options)
        results.append(result)
        colors.append(black)
        if not quiet_progress:
            print("  {}/{}: {} moves, winner={}, {}".format(index + 1, games, result["moves"], result["winner"], result["reason"]), flush=True)
    return {"configuration": {"N": n, "K": k, "time_limit": seconds, "seed": seed,
                               "opponent": str(Path(opponent).resolve()), "options": options or {}},
            "all": summarize(results, colors),
            "as_black": summarize([r for r, c in zip(results, colors) if c], [True] * sum(colors)),
            "as_white": summarize([r for r, c in zip(results, colors) if not c], [False] * (len(colors) - sum(colors))),
            "games": results}


def print_summary(label, summary):
    row = summary["all"]
    print("{}: W/D/L={}/{}/{}, win={:.1%}, mean move={:.4f}s, max={:.4f}s, mean length={:.1f}, forfeits={}".format(
        label, row["wins"], row["draws"], row["losses"], row["win_rate"], row["mean_move_seconds"],
        row["max_move_seconds"], row["mean_game_moves"], row["forfeit_games"]), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measured, reproducible Gomoku experiments")
    parser.add_argument("--configs", nargs="+", default=["9:4", "15:5"], help="N:K configurations")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--self-games", type=int, default=4)
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--opponent", type=Path, default=BASE / "random_ai.py")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="print summaries only")
    parser.add_argument("--json", type=Path, help="optional measured result and full game records")
    args = parser.parse_args(argv)
    if args.games < 1 or args.self_games < 0 or not 0 < args.time < float("inf"):
        parser.error("games >= 1, self-games >= 0, time finite and positive")
    configurations = []
    for config in args.configs:
        try:
            n, k = map(int, config.split(":"))
            if n < 3 or not 3 <= k <= n:
                raise ValueError
        except ValueError:
            parser.error("invalid configuration: " + config)
        configurations.append((n, k))
    if not args.opponent.is_file():
        parser.error("opponent file does not exist")
    started = time.perf_counter()
    runs = []
    for n, k in configurations:
        print("N={}, K={}, time={}s, games={}, alternating colors".format(n, k, args.time, args.games), flush=True)
        series = run_series(args.opponent, n, k, args.time, args.games, args.seed, quiet_progress=args.quiet)
        series["kind"] = "opponent"
        runs.append(series)
        print_summary("Opponent", series)
        if args.self_games:
            series = run_series(BASE / "gomoku_ai.py", n, k, args.time, args.self_games, args.seed + 10000, quiet_progress=args.quiet)
            series["kind"] = "self-play"
            runs.append(series)
            print_summary("Self-play", series)
        if args.ablation:
            for label, options in (("depth-2", {"max_depth": 2}), ("depth-4", {"max_depth": 4}),
                                   ("no-VCF", {"enable_vcf": False})):
                series = run_series(args.opponent, n, k, args.time, args.games, args.seed, options, args.quiet)
                series["kind"] = label
                runs.append(series)
                print_summary(label, series)
    print("Total elapsed: {:.2f}s".format(time.perf_counter() - started))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"seed": args.seed, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved measured results:", args.json)


if __name__ == "__main__":
    mp.freeze_support()
    main()
