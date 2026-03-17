"""Background command execution for the JCoder GUI."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


class CommandRunner:
    """Run CLI commands without blocking the tkinter main loop."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.events: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        proc = self._process
        return proc is not None and proc.poll() is None

    def start(self, cli_args: list[str], external_console: bool = False) -> None:
        """Start a new command run."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("A command is already running")
            if external_console:
                self._launch_external_console(cli_args)
                return
            self._thread = threading.Thread(
                target=self._run_capture,
                args=(cli_args,),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Best-effort stop for the active process."""
        proc = self._process
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _launch_external_console(self, cli_args: list[str]) -> None:
        argv = [sys.executable, str(self.repo_root / "main.py"), *cli_args]
        if os.name == "nt":
            subprocess.Popen(
                argv,
                cwd=self.repo_root,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            self.events.put({"type": "launched_external", "argv": argv})
            return
        raise RuntimeError("Interactive console launch is only implemented for Windows")

    def _run_capture(self, cli_args: list[str]) -> None:
        argv = [sys.executable, str(self.repo_root / "main.py"), *cli_args]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                argv,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._process = process
            self.events.put({"type": "started", "argv": argv, "pid": process.pid})
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put({"type": "output", "text": line})
            return_code = process.wait()
            self.events.put({"type": "finished", "return_code": return_code})
        except Exception as exc:
            self.events.put({"type": "error", "message": str(exc)})
        finally:
            self._process = None
