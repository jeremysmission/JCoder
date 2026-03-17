from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTEST_TEMP_ROOT = PROJECT_ROOT / ".tmp_pytest"
PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
for _env_var in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_env_var] = str(PYTEST_TEMP_ROOT)
tempfile.tempdir = str(PYTEST_TEMP_ROOT)

from agent.memory import AgentMemory
from core.network_gate import NetworkGate


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite3"


class _MockLLM:
    def __init__(self, responses: list[Any], *, default: str = "") -> None:
        self._responses = list(responses)
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str = "",
        context_chunks: Any = None,
        system_prompt: str = "",
        temperature: float = 0,
        max_tokens: int = 256,
    ) -> str:
        call = {
            "question": question,
            "context_chunks": context_chunks,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.calls.append(call)
        if not self._responses:
            return self._default

        response = self._responses.pop(0)
        if callable(response):
            return response(**call)
        return str(response)


@pytest.fixture
def mock_llm() -> Callable[..., _MockLLM]:
    def factory(*responses: Any, default: str = "") -> _MockLLM:
        return _MockLLM(list(responses), default=default)

    return factory


@pytest.fixture
def mock_gate() -> NetworkGate:
    return NetworkGate(mode="offline")


@pytest.fixture
def agent_memory(tmp_path: Path) -> AgentMemory:
    memory = AgentMemory(
        embedding_engine=None,
        index_dir=str(tmp_path / "idx"),
        index_name="test_memory",
        knowledge_dir=str(tmp_path / "knowledge"),
        dimension=768,
    )
    try:
        yield memory
    finally:
        memory.close()


@pytest.fixture
def closables() -> Callable[[Any], Any]:
    resources: list[Any] = []

    def register(resource: Any) -> Any:
        resources.append(resource)
        return resource

    try:
        yield register
    finally:
        while resources:
            resource = resources.pop()
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
