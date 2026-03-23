"""Shell and git operation tools mixin for ToolRegistry."""

from __future__ import annotations

import os
import subprocess
from typing import Any, List


class ShellOpsMixin:
    """Mixin providing shell execution and git tool implementations."""

    # These attributes are provided by ToolRegistry (the concrete class).
    working_dir: str

    def _resolve_path(self, path: str) -> str:  # type: ignore[override]
        ...  # provided by ToolRegistry

    # -- Shell tool --------------------------------------------------------

    def _run_command(
        self, command: str, timeout_s: int = 120
    ) -> Any:
        from agent.tools import (
            ToolResult, MAX_OUTPUT_BYTES,
            _is_command_safe, _split_command_args,
        )

        safe, reason = _is_command_safe(command)
        if not safe:
            return ToolResult(False, "", f"Safety block: {reason}")

        try:
            argv = _split_command_args(command)
            if not argv:
                return ToolResult(False, "", "Command was empty after parsing")
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=min(timeout_s, 600),
                cwd=self.working_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

            return ToolResult(
                success=proc.returncode == 0,
                output=output,
                error="" if proc.returncode == 0 else f"Exit code: {proc.returncode}",
            )
        except OSError as exc:
            return ToolResult(False, "", str(exc))
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout_s}s")

    # -- Git tools ---------------------------------------------------------

    def _git_status(self) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        parts: list[str] = []

        # Current branch
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=15,
                cwd=self.working_dir,
            )
            parts.append(f"Branch: {branch.stdout.strip()}")
        except Exception as e:
            parts.append(f"Branch: (error: {e})")

        # Porcelain status
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            status_text = status.stdout.strip()
            if status_text:
                parts.append(f"\nChanged files:\n{status_text}")
            else:
                parts.append("\nWorking tree clean.")
        except Exception as e:
            parts.append(f"\nStatus error: {e}")

        # Recent commits
        try:
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, timeout=15,
                cwd=self.working_dir,
            )
            log_text = log.stdout.strip()
            if log_text:
                parts.append(f"\nRecent commits:\n{log_text}")
        except Exception as e:
            parts.append(f"\nLog error: {e}")

        output = "\n".join(parts)
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
        return ToolResult(True, output)

    def _git_diff(
        self, staged: bool = False, path: str = ""
    ) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if path:
            cmd.append("--")
            cmd.append(path)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            if not output.strip():
                label = "staged" if staged else "unstaged"
                return ToolResult(True, f"No {label} changes.")
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
            return ToolResult(True, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git diff timed out after 30s")
        except Exception as e:
            return ToolResult(False, "", f"git diff failed: {e}")

    def _git_commit(
        self, files: List[str], message: str
    ) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        # Safety: refuse empty inputs
        if not files:
            return ToolResult(False, "", "Refusing to commit: files list is empty.")
        if not message or not message.strip():
            return ToolResult(False, "", "Refusing to commit: message is empty.")

        # Validate all file paths are within allowed dirs
        resolved_files: list[str] = []
        for f in files:
            try:
                resolved_files.append(self._resolve_path(f))
            except PermissionError as e:
                return ToolResult(False, "", str(e))

        # Stage files
        try:
            add_proc = subprocess.run(
                ["git", "add", "--"] + resolved_files,
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            if add_proc.returncode != 0:
                return ToolResult(
                    False, "",
                    f"git add failed (exit {add_proc.returncode}): "
                    f"{add_proc.stderr.strip()}",
                )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git add timed out after 30s")

        # Commit
        try:
            commit_proc = subprocess.run(
                ["git", "commit", "-m", message.strip()],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            output = commit_proc.stdout
            if commit_proc.stderr:
                output += f"\n[stderr]\n{commit_proc.stderr}"
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
            return ToolResult(
                success=commit_proc.returncode == 0,
                output=output,
                error="" if commit_proc.returncode == 0
                else f"git commit failed (exit {commit_proc.returncode})",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git commit timed out after 30s")

    # -- Test runner tool --------------------------------------------------

    def _run_tests(
        self,
        target: str = "",
        verbose: bool = True,
        timeout_s: int = 300,
    ) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        cmd = ["python", "-m", "pytest"]
        if target:
            cmd.append(target)
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short"])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=min(timeout_s, 600),
                cwd=self.working_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"

            # Extract summary line (e.g. "5 passed, 1 failed in 2.3s")
            summary_line = ""
            for line in reversed(output.splitlines()):
                stripped = line.strip()
                if "passed" in stripped or "failed" in stripped or "error" in stripped:
                    summary_line = stripped
                    break

            if summary_line:
                output += f"\n\n=== SUMMARY: {summary_line} ==="

            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

            return ToolResult(
                success=proc.returncode == 0,
                output=output,
                error="" if proc.returncode == 0
                else f"Tests failed (exit {proc.returncode})",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                False, "", f"Tests timed out after {timeout_s}s"
            )
