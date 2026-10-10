#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment 1 entry point. Python 3.10+; standard library only.

    python main.py                         # graphical game
    python main.py --board 9 --win 4        # configurable graphical game
    python main.py --cli --mode human-ai    # terminal game
    python main.py --cli --mode ai-ai --ai-white gemini_ai.py --games 2
    python arena.py gomoku_ai.py gemini_ai.py --time 5 --games 10 --quiet
    python test_gomoku.py                   # rules and tactical regression tests
    python experiments.py --help           # reproducible assignment experiments

The judge imports gomoku_ai.py alone; no sibling file is needed by that AI.
The GUI opens a mode-selection screen. There are no pip dependencies;
the GUI needs Python's tkinter. The Windows executable includes its runtime.
"""

import multiprocessing as mp
import sys

from main_cli import build_parser, validate_args


def main():
    args_list = sys.argv[1:]
    if args_list and args_list[0] == "--self-test":
        from desktop_check import run_checks
        run_checks(args_list[1])
        return
    if "--cli" in args_list:
        args_list.remove("--cli")
        from main_cli import main as console_main
        console_main(args_list)
        return
    parser = build_parser()
    parser.description += " (GUI; add --cli for terminal mode)"
    args = parser.parse_args(args_list)
    validate_args(parser, args)
    try:
        from gui import launch
    except ImportError as exc:
        print("GUI unavailable: {}. Use python main.py --cli instead.".format(exc), file=sys.stderr)
        raise SystemExit(1)
    try:
        launch(args)
    except Exception as exc:
        # Keep an incomplete Tcl/Tk installation from producing an opaque trace.
        import tkinter
        if isinstance(exc, tkinter.TclError):
            print("Tk could not start: {}. Use python main.py --cli or a Python installation with working Tk.".format(exc), file=sys.stderr)
            raise SystemExit(1)
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
