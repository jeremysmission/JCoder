"""
Run and supervise the JCoder download queue.

Usage:
    python scripts/run_download_queue.py --list
    python scripts/run_download_queue.py
    python scripts/run_download_queue.py --status
    python scripts/run_download_queue.py --only learn_rust
    python scripts/run_download_queue.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = PROJECT_ROOT / "config" / "download_queue.json"
LOG_DIR = PROJECT_ROOT / "logs" / "download_queue"
STATUS_PATH = LOG_DIR / "queue_status.json"
LOCK_PATH = LOG_DIR / "queue_service.lock"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

HEARTBEAT_STALE_S = 300.0

ENV_AUTOSTART = "JCODER_AUTOSTART_DOWNLOAD_QUEUE"
ENV_DISABLE_AUTOSTART = "JCODER_DISABLE_DOWNLOAD_AUTOSTART"
ENV_CHILD = "JCODER_DOWNLOAD_QUEUE_CHILD"
ENV_INCLUDE_MANUAL_AUTH = "JCODER_DOWNLOAD_QUEUE_INCLUDE_MANUAL_AUTH"
ENV_INCLUDE_OPTIONAL = "JCODER_DOWNLOAD_QUEUE_INCLUDE_OPTIONAL"
ENV_INCLUDE_GIANT = "JCODER_DOWNLOAD_QUEUE_INCLUDE_GIANT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}


def _huggingface_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    return ""


def default_service_flags() -> dict[str, bool]:
    return {
        "include_manual_auth": _env_flag(ENV_INCLUDE_MANUAL_AUTH, bool(_huggingface_token())),
        "include_optional": _env_flag(ENV_INCLUDE_OPTIONAL, True),
        "include_giant": _env_flag(ENV_INCLUDE_GIANT, True),
    }


def download_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if not (env.get("JCODER_DATA") or env.get("JCODER_DATA_DIR")):
        env["JCODER_DATA"] = str(DEFAULT_DATA_ROOT)
    return env


def continue_on_error_enabled(*, only: set[str] | None, requested: bool) -> bool:
    if requested:
        return True
    return only is None or len(only) != 1


def load_queue() -> list[dict]:
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def entry_tags(entry: dict) -> set[str]:
    return {str(tag).strip().lower() for tag in entry.get("tags", []) if str(tag).strip()}


def filter_queue(
    queue: list[dict],
    *,
    only: set[str] | None = None,
    include_manual_auth: bool = False,
    include_optional: bool = False,
    include_giant: bool = False,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    runnable: list[dict] = []
    skipped: list[tuple[dict, str]] = []

    for entry in queue:
        if only and entry.get("id") not in only:
            continue

        tags = entry_tags(entry)
        blocked: list[str] = []
        if "manual-auth" in tags and not include_manual_auth:
            blocked.append("manual-auth")
        if "optional" in tags and not include_optional:
            blocked.append("optional")
        if "giant" in tags and not include_giant:
            blocked.append("giant")

        if blocked:
            skipped.append((entry, ", ".join(blocked)))
            continue

        runnable.append(entry)

    return runnable, skipped


def _module_name_for(command_target: str) -> str | None:
    target_path = Path(command_target)
    if target_path.suffix.lower() != ".py":
        return None

    if target_path.is_absolute():
        try:
            target_path = target_path.relative_to(PROJECT_ROOT)
        except ValueError:
            return None

    return ".".join(target_path.with_suffix("").parts)


def build_command(entry: dict) -> list[str]:
    command = list(entry["command"])
    if not command:
        raise ValueError(f"Queue entry {entry.get('id', '?')} has no command")

    module_name = _module_name_for(command[0])
    if module_name:
        return [sys.executable, "-m", module_name, *command[1:]]

    script_path = Path(command[0])
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    return [sys.executable, str(script_path), *command[1:]]


def runnable_ids(queue: list[dict]) -> list[str]:
    return [str(entry.get("id", "")) for entry in queue]


def compute_queue_signature(
    queue: list[dict],
    *,
    include_manual_auth: bool,
    include_optional: bool,
    include_giant: bool,
) -> str:
    payload = {
        "flags": {
            "include_manual_auth": include_manual_auth,
            "include_optional": include_optional,
            "include_giant": include_giant,
        },
        "entries": [
            {
                "id": entry.get("id"),
                "command": entry.get("command"),
                "tags": sorted(entry_tags(entry)),
            }
            for entry in queue
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_status() -> dict[str, Any]:
    return _load_json(STATUS_PATH)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def write_status(payload: dict[str, Any]) -> None:
    snapshot = dict(payload)
    snapshot["updated_at"] = _utc_now()
    _write_json(STATUS_PATH, snapshot)


def load_lock() -> dict[str, Any]:
    return _load_json(LOCK_PATH)


def _pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _status_is_live(status: dict[str, Any], *, stale_after_s: float = HEARTBEAT_STALE_S) -> bool:
    if status.get("status") != "running":
        return False

    pid = int(status.get("pid") or 0)
    heartbeat = float(status.get("heartbeat_epoch") or 0.0)
    if not _pid_is_alive(pid) or heartbeat <= 0.0:
        return False
    return (time.time() - heartbeat) <= stale_after_s


def service_is_running(*, stale_after_s: float = HEARTBEAT_STALE_S) -> bool:
    status = load_status()
    if _status_is_live(status, stale_after_s=stale_after_s):
        return True

    lock_data = load_lock()
    pid = int(lock_data.get("pid") or 0)
    created_epoch = float(lock_data.get("created_epoch") or 0.0)
    if not _pid_is_alive(pid):
        return False
    if created_epoch <= 0.0:
        return True
    return (time.time() - created_epoch) <= stale_after_s


def _recover_stale_lock() -> bool:
    if not LOCK_PATH.exists():
        return False
    if service_is_running():
        return False

    lock_data = load_lock()
    lock_pid = int(lock_data.get("pid") or 0)
    if lock_pid and _pid_is_alive(lock_pid):
        return False

    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        return True
    return True


def acquire_service_lock() -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for _ in range(2):
        try:
            handle = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _recover_stale_lock():
                continue
            return False

        lock_payload = {
            "pid": os.getpid(),
            "created_at": _utc_now(),
            "created_epoch": time.time(),
        }
        with os.fdopen(handle, "w", encoding="utf-8") as fp:
            json.dump(lock_payload, fp, indent=2, sort_keys=True)
        return True

    return False


def release_service_lock() -> None:
    lock_data = load_lock()
    lock_pid = int(lock_data.get("pid") or 0)
    if lock_pid not in {0, os.getpid()}:
        return
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        return


def build_service_command(
    *,
    include_manual_auth: bool,
    include_optional: bool,
    include_giant: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.run_download_queue",
        "--continue-on-error",
        "--service",
    ]
    if include_manual_auth:
        command.append("--include-manual-auth")
    if include_optional:
        command.append("--include-optional")
    if include_giant:
        command.append("--include-giant")
    return command


def should_launch_queue_service(
    status: dict[str, Any],
    *,
    queue_signature: str,
    runnable_queue_ids: list[str],
) -> bool:
    if not runnable_queue_ids:
        return False
    if _status_is_live(status):
        return False
    if (
        status.get("status") == "completed"
        and status.get("queue_signature") == queue_signature
        and status.get("runnable_ids") == runnable_queue_ids
        and not status.get("failures")
    ):
        return False
    return True


def launch_background_queue(
    *,
    include_manual_auth: bool,
    include_optional: bool,
    include_giant: bool,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOG_DIR / f"run_download_queue_autostart_{timestamp}.out.log"
    stderr_path = LOG_DIR / f"run_download_queue_autostart_{timestamp}.err.log"
    command = build_service_command(
        include_manual_auth=include_manual_auth,
        include_optional=include_optional,
        include_giant=include_giant,
    )
    env = download_env(os.environ.copy())
    env[ENV_CHILD] = "1"

    if os.name == "nt":
        def _ps_quote(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        file_path = _ps_quote(command[0])
        arguments = ", ".join(_ps_quote(arg) for arg in command[1:])
        working_dir = _ps_quote(str(PROJECT_ROOT))
        stdout_arg = _ps_quote(str(stdout_path))
        stderr_arg = _ps_quote(str(stderr_path))
        script = (
            "$p = Start-Process "
            f"-FilePath {file_path} "
            f"-ArgumentList @({arguments}) "
            f"-WorkingDirectory {working_dir} "
            f"-RedirectStandardOutput {stdout_arg} "
            f"-RedirectStandardError {stderr_arg} "
            "-WindowStyle Hidden -PassThru; "
            "Write-Output $p.Id"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        pid = int(result.stdout.strip().splitlines()[-1])
        return {
            "action": "launched",
            "pid": pid,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=os.name != "nt",
        )

    return {
        "action": "launched",
        "pid": process.pid,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _autostart_enabled() -> bool:
    if os.environ.get(ENV_CHILD):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if _env_flag(ENV_DISABLE_AUTOSTART, False):
        return False
    return _env_flag(ENV_AUTOSTART, True)


def ensure_queue_running(
    *,
    include_manual_auth: bool | None = None,
    include_optional: bool | None = None,
    include_giant: bool | None = None,
) -> dict[str, Any]:
    if not _autostart_enabled():
        return {"action": "disabled"}

    defaults = default_service_flags()
    include_manual_auth = defaults["include_manual_auth"] if include_manual_auth is None else include_manual_auth
    include_optional = defaults["include_optional"] if include_optional is None else include_optional
    include_giant = defaults["include_giant"] if include_giant is None else include_giant

    if service_is_running():
        return {"action": "noop", "reason": "running"}

    queue, _ = filter_queue(
        load_queue(),
        include_manual_auth=include_manual_auth,
        include_optional=include_optional,
        include_giant=include_giant,
    )
    queue_ids = runnable_ids(queue)
    queue_signature = compute_queue_signature(
        queue,
        include_manual_auth=include_manual_auth,
        include_optional=include_optional,
        include_giant=include_giant,
    )
    status = load_status()

    if not should_launch_queue_service(
        status,
        queue_signature=queue_signature,
        runnable_queue_ids=queue_ids,
    ):
        return {"action": "noop", "reason": "up_to_date"}

    return launch_background_queue(
        include_manual_auth=include_manual_auth,
        include_optional=include_optional,
        include_giant=include_giant,
    )


def print_status(status: dict[str, Any]) -> None:
    if not status:
        print("No downloader queue status recorded yet.")
        return

    print(f"status: {status.get('status', 'unknown')}")
    if status.get("pid"):
        print(f"pid: {status['pid']}")
    if status.get("mode"):
        print(f"mode: {status['mode']}")
    if status.get("started_at"):
        print(f"started_at: {status['started_at']}")
    if status.get("updated_at"):
        print(f"updated_at: {status['updated_at']}")
    if status.get("active_entry_id"):
        print(f"active_entry_id: {status['active_entry_id']}")
    runnable = status.get("runnable_ids") or []
    completed = status.get("completed_ids") or []
    failures = status.get("failures") or []
    print(f"runnable_jobs: {len(runnable)}")
    print(f"completed_jobs: {len(completed)}")
    print(f"failed_jobs: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"  {failure.get('id')}: exit={failure.get('exit_code')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queued JCoder download backlog jobs")
    parser.add_argument("--list", action="store_true", help="List queued jobs and exit")
    parser.add_argument("--status", action="store_true", help="Show queue worker status and exit")
    parser.add_argument("--only", action="append", default=[], help="Run only the specified queue id")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument(
        "--include-manual-auth",
        action="store_true",
        help="Include jobs tagged manual-auth (disabled by default)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include jobs tagged optional (disabled by default)",
    )
    parser.add_argument(
        "--include-giant",
        action="store_true",
        help="Include jobs tagged giant (disabled by default)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later queue items after a failure",
    )
    parser.add_argument("--service", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.status:
        print_status(load_status())
        return 0

    wanted = set(args.only) if args.only else None
    continue_on_error = continue_on_error_enabled(only=wanted, requested=args.continue_on_error)
    queue, skipped = filter_queue(
        load_queue(),
        only=wanted,
        include_manual_auth=args.include_manual_auth,
        include_optional=args.include_optional,
        include_giant=args.include_giant,
    )

    if args.list:
        for entry in queue:
            cmd = " ".join(build_command(entry))
            print(f"{entry['id']}: {entry['description']}")
            print(f"  {cmd}")
        if skipped:
            print("\nSkipped by default:")
            for entry, reason in skipped:
                print(f"  {entry['id']}: {reason}")
        return 0

    queue_ids = runnable_ids(queue)
    queue_signature = compute_queue_signature(
        queue,
        include_manual_auth=args.include_manual_auth,
        include_optional=args.include_optional,
        include_giant=args.include_giant,
    )

    acquired_lock = False
    status_payload: dict[str, Any] = {}
    failures: list[tuple[str, int]] = []
    child_env = download_env()

    try:
        if not args.dry_run:
            acquired_lock = acquire_service_lock()
            if not acquired_lock:
                print("Downloader queue is already running.")
                print_status(load_status())
                return 0

            status_payload = {
                "status": "running",
                "mode": "service" if args.service else "manual",
                "pid": os.getpid(),
                "started_at": _utc_now(),
                "heartbeat_epoch": time.time(),
                "queue_signature": queue_signature,
                "runnable_ids": queue_ids,
                "completed_ids": [],
                "failed_ids": [],
                "active_entry_id": "",
                "continue_on_error": continue_on_error,
                "skip_reasons": {entry["id"]: reason for entry, reason in skipped},
                "failures": [],
            }
            write_status(status_payload)

        if skipped:
            print("Skipping queue entries by default:")
            for entry, reason in skipped:
                print(f"  {entry['id']}: {reason}")
            print("")

        for index, entry in enumerate(queue, start=1):
            cmd = build_command(entry)
            print(f"[{index}/{len(queue)}] {entry['id']}: {entry['description']}")
            print("  " + " ".join(cmd))

            if args.dry_run:
                continue

            status_payload["active_entry_id"] = entry["id"]
            status_payload["heartbeat_epoch"] = time.time()
            write_status(status_payload)

            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, env=child_env)
            if result.returncode != 0:
                failures.append((entry["id"], result.returncode))
                status_payload["failed_ids"].append(entry["id"])
                print(f"  [FAIL] exit={result.returncode}")
                if not continue_on_error:
                    break
            else:
                status_payload["completed_ids"].append(entry["id"])
                print("  [OK]")

            status_payload["heartbeat_epoch"] = time.time()
            status_payload["failures"] = [
                {"id": job_id, "exit_code": code} for job_id, code in failures
            ]
            write_status(status_payload)

        if failures:
            print("\nQueue completed with failures:")
            for job_id, code in failures:
                print(f"  {job_id}: exit={code}")
            return 1

        print("\nQueue completed successfully.")
        return 0
    finally:
        if acquired_lock:
            status_payload["status"] = "failed" if failures else "completed"
            status_payload["active_entry_id"] = ""
            status_payload["heartbeat_epoch"] = time.time()
            status_payload["finished_at"] = _utc_now()
            status_payload["failures"] = [
                {"id": job_id, "exit_code": code} for job_id, code in failures
            ]
            write_status(status_payload)
            release_service_lock()


if __name__ == "__main__":
    raise SystemExit(main())
