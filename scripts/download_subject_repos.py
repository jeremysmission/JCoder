from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_ROOT = Path(os.environ.get("JCODER_DOWNLOAD_DIR", r"data\downloads")) / "subject_repos"


@dataclass(frozen=True)
class RepoSource:
    key: str
    label: str
    url: str
    target_dir: str


REPO_SOURCES = [
    RepoSource("openjdk_jdk", "OpenJDK JDK", "https://github.com/openjdk/jdk.git", "openjdk_jdk"),
    RepoSource("dotnet_runtime", ".NET Runtime", "https://github.com/dotnet/runtime.git", "dotnet_runtime"),
    RepoSource("rust_lang_rust", "Rust Compiler", "https://github.com/rust-lang/rust.git", "rust_lang_rust"),
    RepoSource("golang_go", "Go Language Repo", "https://github.com/golang/go.git", "golang_go"),
    RepoSource("microsoft_typescript", "TypeScript Compiler", "https://github.com/microsoft/TypeScript.git", "microsoft_typescript"),
    RepoSource("typescript_website", "TypeScript Website", "https://github.com/microsoft/TypeScript-Website.git", "typescript_website"),
    RepoSource("mdn_content", "MDN Content", "https://github.com/mdn/content.git", "mdn_content"),
    RepoSource("kubernetes_website", "Kubernetes Website", "https://github.com/kubernetes/website.git", "kubernetes_website"),
    RepoSource("docker_docs", "Docker Docs", "https://github.com/docker/docs.git", "docker_docs"),
    RepoSource("react_dev", "React Dev", "https://github.com/reactjs/react.dev.git", "react_dev"),
    RepoSource("fastapi", "FastAPI", "https://github.com/fastapi/fastapi.git", "fastapi"),
    RepoSource("django", "Django", "https://github.com/django/django.git", "django"),
    RepoSource("langchain", "LangChain", "https://github.com/langchain-ai/langchain.git", "langchain"),
    RepoSource("autogen", "AutoGen", "https://github.com/microsoft/autogen.git", "autogen"),
    RepoSource("openai_python", "OpenAI Python", "https://github.com/openai/openai-python.git", "openai_python"),
]
SOURCE_BY_KEY = {source.key: source for source in REPO_SOURCES}


def repo_destination(source: RepoSource) -> Path:
    return DOWNLOAD_ROOT / source.target_dir


def selected_sources(only: list[str] | None = None) -> list[RepoSource]:
    if not only:
        return list(REPO_SOURCES)

    selected: list[RepoSource] = []
    seen: set[str] = set()
    for raw_key in only:
        key = raw_key.strip()
        if not key or key in seen:
            continue
        if key not in SOURCE_BY_KEY:
            raise KeyError(key)
        seen.add(key)
        selected.append(SOURCE_BY_KEY[key])
    return selected


def build_clone_command(source: RepoSource, destination: Path) -> list[str]:
    return ["git", "clone", "--depth", "1", source.url, str(destination)]


def build_update_command(destination: Path) -> list[str]:
    return ["git", "-C", str(destination), "pull", "--ff-only"]


def _repo_size_gb(path: Path) -> float:
    total_bytes = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 ** 3)


def _run_command(command: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False, env=env)
    return result.returncode


def clone_or_update(source: RepoSource) -> bool:
    destination = repo_destination(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (destination / ".git").exists():
        print(f"  [UPDATE] {source.key}: {destination}")
        return_code = _run_command(build_update_command(destination))
    elif destination.exists() and any(destination.iterdir()):
        print(f"  [WARN] {source.key}: destination exists without .git -> {destination}")
        return False
    else:
        print(f"  [CLONE] {source.key}: {source.url}")
        return_code = _run_command(build_clone_command(source, destination))

    if return_code != 0:
        print(f"  [FAIL] {source.key}: exit={return_code}")
        return False

    size_gb = _repo_size_gb(destination)
    print(f"  [OK] {source.key}: {destination} ({size_gb:.2f} GB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download large official subject repositories into the JCoder queue"
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated list of source keys to download",
    )
    args = parser.parse_args()

    selected_keys = [item.strip() for item in args.only.split(",") if item.strip()]
    try:
        sources = selected_sources(selected_keys or None)
    except KeyError as exc:
        print(f"[FAIL] Unknown source key: {exc.args[0]}")
        return 1

    print("=" * 60)
    print("JCoder Subject Repository Download")
    print(f"Targets: {', '.join(source.key for source in sources)}")
    print(f"Destination root: {DOWNLOAD_ROOT}")
    print("=" * 60)

    started_at = time.monotonic()
    failures: list[str] = []
    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}/{len(sources)}] {source.label}")
        ok = clone_or_update(source)
        if not ok:
            failures.append(source.key)

    elapsed = time.monotonic() - started_at
    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with failures: {', '.join(failures)}")
    else:
        print("Completed successfully.")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
