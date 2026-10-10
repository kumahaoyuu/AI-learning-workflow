"""Optional native/frozen app smoke check: main.py --self-test result.json."""

import json
from pathlib import Path
import sys
import time
import tkinter as tk

from gui import HomeScreen, BASE, create_root
from main_cli import build_parser


def run_checks(output_path):
    """Exercise the actual GUI and bundled AI workers, writing measured results."""
    root = create_root()
    root.withdraw()
    root.geometry("1100x820")
    checks = []
    payload = {"frozen": bool(getattr(sys, "frozen", False)), "checks": checks}

    def expect(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def pump(predicate, timeout=20):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            root.update()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError("AI did not reach expected state within test budget")

    def menu(mode, human="black"):
        args = build_parser().parse_args(["--mode", mode, "--board", "5", "--win", "3",
                                         "--time", "0.3", "--human-color", human])
        screen = HomeScreen(root, args)
        screen.start()
        root.update()
        return screen.app

    def leave(app):
        processes = [w.process for w in app.workers.values()]
        app.return_home()
        expect(all(not p.is_alive() for p in processes), "AI workers stop when leaving game")
        for child in root.winfo_children():
            child.destroy()

    try:
        expect((BASE / "gomoku_ai.py").is_file() and (BASE / "random_ai.py").is_file(), "Bundled AI files available")
        app = menu("human-human")
        expect(not app.workers, "Human-human needs no AI process")
        for cell in ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)):
            app.game.play(cell)
        app._refresh_status()
        expect(app.game.winner == 1 and "黑棋获胜" in app.status.get(), "Human-human detects and displays victory")
        app.undo()
        expect(not app.game.finished and len(app.game.history) == 4, "Undo reopens finished game")
        leave(app)

        app = menu("human-ai")
        app.game.play((2, 2))
        pump(lambda: len(app.game.history) >= 2 or app.game.finished)
        expect(len(app.game.history) == 2 and app.game.to_move == 1 and not app.game.finished, "Bundled Aegis responds to human black")
        app.undo()
        expect(not app.game.history and app.game.to_move == 1, "Human-AI undo removes a complete turn")
        leave(app)

        app = menu("human-ai", "white")
        pump(lambda: len(app.game.history) >= 1 or app.game.finished)
        expect(len(app.game.history) == 1 and app.game.to_move == 2 and not app.game.finished, "Bundled Aegis opens when human chooses white")
        leave(app)

        app = menu("ai-ai")
        pump(lambda: len(app.game.history) >= 1 or app.game.finished)
        app.toggle_pause()
        pump(lambda: app.paused or app.game.finished)
        expect(app.paused and not app.game.finished, "AI-AI pause works")
        app.toggle_pause()
        pump(lambda: app.game.finished)
        expect(app.game.reason in ("3-in-a-row", "board full"), "AI-AI resumes and completes a legal game")
        payload["ai_game_moves"] = len(app.game.history)
        payload["ai_game_winner"] = app.game.winner
        leave(app)
        payload["passed"] = True
    except BaseException as exc:
        payload["passed"] = False
        payload["error"] = str(exc)
        raise
    finally:
        if root.winfo_exists():
            root.destroy()
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
