"""
Prompt Templates and FIM Support
---------------------------------
System prompts for each agent mode, Fill-in-Middle formatting for
code completion models, and a PromptBuilder that wires them together.

Modes: agent, qa, review, explain, debug, refactor, fim
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# System prompts for different modes
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are JCoder, an autonomous coding agent. You solve tasks by reading, \
writing, searching, and executing code with the tools provided.

Rules (follow strictly):

1. MEMORY FIRST. Call memory_search before starting work -- check if this \
task (or something similar) was solved before. Reuse what works.

2. PLAN. Break complex tasks into small, verifiable steps. State your plan \
before acting.

3. READ BEFORE MODIFYING. Never assume file contents. Read the file first, \
then edit only what needs to change.

4. MINIMAL CHANGES. Do not refactor unrelated code, rename things that \
work, or add features not requested.

5. TEST AFTER CHANGES. Run relevant tests or a quick verification after \
every modification. If tests fail, diagnose and fix before moving on.

6. USE TOOLS. You have file I/O, shell, search, and a code knowledge base. \
Use them -- do not guess from memory.

7. HANDLE ERRORS. If a tool call fails, read the error, adjust your \
approach, and retry with a different strategy. Never repeat the exact \
same failing call.

8. STORE DISCOVERIES. When you learn something useful (a pattern, a fix, \
a gotcha), call memory_store so future runs benefit.

9. SIGNAL COMPLETION. When fully done and verified, call task_complete \
with a concise summary: what changed, which files, how it was verified.

10. BE CONCISE. Cite file paths and line numbers. Skip lengthy narration.\
"""

CODE_QA_PROMPT = """\
You are a code Q&A assistant. Answer the user's question using ONLY \
the provided context. Do not use outside knowledge.

Rules:
- If the context does not contain enough information, say "I don't have \
enough context to answer this."
- Include code in fenced blocks with the correct language tag.
- Include necessary imports in code examples.
- Cite source files by path and line number when referencing specific code.
- Keep answers focused and direct.\
"""

CODE_REVIEW_PROMPT = """\
You are a senior code reviewer. Review the provided code thoroughly.

Check for:
1. Bugs -- logic errors, off-by-one, null/None handling, race conditions
2. Security -- injection, path traversal, secrets in code, OWASP Top 10
3. Performance -- unnecessary allocations, O(n^2) where O(n) is possible, \
missing caching, blocking calls in async code
4. Maintainability -- unclear naming, missing error handling, god classes

For each finding:
- State the severity: CRITICAL / WARNING / INFO
- Quote the problematic code with file path and line number
- Provide a concrete fix as a code block
- Explain why it matters in one sentence

Start with a one-line overall assessment, then list findings by severity.\
"""

CODE_EXPLAIN_PROMPT = """\
You are a code explainer. Make the provided code understandable to \
someone learning the codebase.

Structure your explanation:
1. HIGH-LEVEL SUMMARY -- What does this code do and why does it exist? \
One paragraph.
2. KEY SECTIONS -- Walk through the important parts in order. For each, \
explain what it does and why.
3. DATA FLOW -- How does data move through this code? What goes in, \
what comes out?
4. GOTCHAS -- Note any non-obvious behavior, implicit assumptions, or \
things that could surprise a reader.

Use simple language. Avoid jargon unless you define it. Reference line \
numbers when discussing specific sections.\
"""

DEBUG_PROMPT = """\
You are a debugging expert. Analyze the error and help fix it.

Approach:
1. READ THE ERROR -- Parse the error message and full stack trace carefully.
2. IDENTIFY ROOT CAUSE -- What actually went wrong, not just the symptom. \
Trace the call chain back to the origin.
3. EXPLAIN WHY -- Why did this error happen? What condition triggered it?
4. PROVIDE THE FIX -- Show the corrected code in a fenced block with the \
language tag. Include the file path.
5. PREVENT RECURRENCE -- Suggest a test or guard that would catch this \
before it happens again.

If the error message is ambiguous, list the most likely causes ranked by \
probability.\
"""

REFACTOR_PROMPT = """\
You are a refactoring specialist. Improve code structure without changing \
behavior.

Rules:
- PRESERVE BEHAVIOR. The refactored code must produce identical outputs \
for all inputs. No feature additions, no feature removals.
- IDENTIFY SMELLS. Name the specific code smell (long method, duplicated \
logic, feature envy, primitive obsession, etc.).
- SHOW BEFORE AND AFTER. For each change, show the original and refactored \
code side by side in fenced blocks.
- KEEP IT SIMPLE. Prefer straightforward solutions over clever ones. Do \
not introduce design patterns unless they clearly reduce complexity.
- ONE THING AT A TIME. Each suggested change should be independent and \
reviewable on its own.\
"""

CODE_GROUNDED_PROMPT = """\
You are JCoder, a code assistant that answers using ONLY the retrieved \
code knowledge below. You do not use outside knowledge.

Rules (priority order):

1. SOURCE-BOUNDED. Base your answer strictly on the provided context \
chunks. If the context does not contain enough information, say \
"I don't have enough context to answer this" -- never guess or fabricate.

2. CODE FIRST. Show working code in a fenced block with the correct \
language tag. Include all necessary imports. The code must be \
syntactically valid and runnable.

3. CITE SOURCES. After every factual claim or code example, cite the \
source chunk: [Source: chunk_name or path]. If multiple chunks agree, \
cite the most specific one.

4. EXPLAIN CONCISELY. After the code, add a brief (2-4 sentence) \
explanation of what it does and why this approach works. Avoid filler.

5. AMBIGUITY. If the question could mean multiple things, address the \
most common interpretation first, then briefly note alternatives.

6. SECURITY. Never reproduce hardcoded secrets, API keys, or credentials \
from source chunks even if they appear in the context.

7. LANGUAGE MATCH. Match the programming language of your answer to \
what the user asked about. If unclear, use Python.

8. VERSION AWARENESS. If the context shows version-specific APIs, note \
which version(s) the code targets.

9. LIMITATIONS. If your answer is incomplete (e.g., the context covers \
the API but not error handling), explicitly state what is missing.\
"""

# Map mode names to their prompts
_MODE_PROMPTS: Dict[str, str] = {
    "agent": AGENT_SYSTEM_PROMPT,
    "qa": CODE_QA_PROMPT,
    "code": CODE_GROUNDED_PROMPT,
    "review": CODE_REVIEW_PROMPT,
    "explain": CODE_EXPLAIN_PROMPT,
    "debug": DEBUG_PROMPT,
    "refactor": REFACTOR_PROMPT,
}


# ---------------------------------------------------------------------------
# FIM (Fill-in-Middle) support
# ---------------------------------------------------------------------------

class FIMFormatter:
    """Format code completion prompts for Fill-in-Middle models.

    Each model family uses different special tokens. This class wraps
    the formatting so callers never deal with raw tokens.

    Devstral format:  [SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]
    CodeLlama format: <PRE>{prefix} <SUF>{suffix} <MID>
    StarCoder format: <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>
    """

    FORMATS: Dict[str, Dict[str, str]] = {
        "devstral": {
            "prefix": "[PREFIX]",
            "suffix": "[SUFFIX]",
            "middle": "[MIDDLE]",
        },
        "codellama": {
            "prefix": "<PRE>",
            "suffix": " <SUF>",
            "middle": " <MID>",
        },
        "starcoder": {
            "prefix": "<fim_prefix>",
            "suffix": "<fim_suffix>",
            "middle": "<fim_middle>",
        },
        "deepseek": {
            "prefix": "<|fim_begin|>",
            "suffix": "<|fim_hole|>",
            "middle": "<|fim_end|>",
        },
        "generic": {
            "prefix": "### PREFIX\n",
            "suffix": "\n### SUFFIX\n",
            "middle": "\n### MIDDLE\n",
        },
    }

    def __init__(self, model_format: str = "devstral"):
        if model_format not in self.FORMATS:
            model_format = "generic"
        self._fmt = self.FORMATS[model_format]
        self._model_format = model_format
        # Build a set of all tokens for this format (used by extract)
        self._all_tokens = set(self._fmt.values())

    def format_completion(self, prefix: str, suffix: str) -> str:
        """Format a FIM prompt from code before and after the cursor.

        Parameters
        ----------
        prefix : str
            Code before the cursor (what the model sees as already written).
        suffix : str
            Code after the cursor (what comes next).

        Returns
        -------
        str
            The formatted FIM prompt string ready to send to the model.
        """
        fmt = self._fmt
        # Devstral puts suffix before prefix
        if self._model_format == "devstral":
            return f"{fmt['suffix']}{suffix}{fmt['prefix']}{prefix}{fmt['middle']}"
        # All other formats: prefix, suffix, middle
        return f"{fmt['prefix']}{prefix}{fmt['suffix']}{suffix}{fmt['middle']}"

    def extract_completion(self, response: str) -> str:
        """Extract the generated middle from model response.

        Strips any format tokens the model may have echoed back and
        returns clean code.
        """
        text = response
        for token in self._all_tokens:
            text = text.replace(token, "")
        # Strip leading/trailing whitespace that some models add around
        # the completion, but preserve internal whitespace
        return text.strip("\n")

    @classmethod
    def supported_formats(cls) -> List[str]:
        """Return list of supported FIM format names."""
        return list(cls.FORMATS.keys())


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Builds prompts by combining system prompt + context + user query.

    Produces an OpenAI-format message list ready for any LLM backend.
    """

    def __init__(self, mode: str = "agent"):
        if mode == "fim":
            self._system_prompt = ""
        elif mode not in _MODE_PROMPTS:
            raise ValueError(
                f"Unknown mode {mode!r}. "
                f"Available: {', '.join(self.available_modes())}"
            )
        else:
            self._system_prompt = _MODE_PROMPTS[mode]
        self._mode = mode
        self._fim = FIMFormatter() if mode == "fim" else None

    @property
    def mode(self) -> str:
        return self._mode

    def build_messages(
        self,
        query: str,
        context: str = "",
        code: str = "",
        error: str = "",
        *,
        fim_prefix: str = "",
        fim_suffix: str = "",
        fim_format: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build a complete message list for the LLM.

        Parameters
        ----------
        query : str
            The user's question or task description.
        context : str
            Retrieved RAG context (code chunks, docs).
        code : str
            Specific code snippet being discussed (review, explain, debug).
        error : str
            Error message or stack trace (debug mode).
        fim_prefix : str
            Code before cursor (FIM mode only).
        fim_suffix : str
            Code after cursor (FIM mode only).
        fim_format : str, optional
            Override FIM model format (default: devstral).

        Returns
        -------
        list[dict]
            OpenAI-format messages: [system, user].
        """
        # FIM mode: single raw prompt, no system message
        if self._mode == "fim":
            fmt = FIMFormatter(fim_format or "devstral")
            return [{"role": "user", "content": fmt.format_completion(fim_prefix, fim_suffix)}]

        # Build user content from available parts
        parts: List[str] = []
        if context:
            if self._mode == "code":
                parts.append(
                    "Retrieved Knowledge Chunks "
                    "(use ONLY these to answer):\n" + context
                )
            else:
                parts.append(f"Context:\n{context}")
        if code:
            parts.append(f"Code:\n```\n{code}\n```")
        if error:
            parts.append(f"Error:\n```\n{error}\n```")
        parts.append(query)

        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages

    @staticmethod
    def available_modes() -> List[str]:
        """Return list of available prompt modes."""
        return sorted(list(_MODE_PROMPTS.keys()) + ["fim"])
