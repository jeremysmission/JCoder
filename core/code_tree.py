"""
CodeTree: Agent-Guided Tree Search for Code Generation
-------------------------------------------------------
Tree search with Thinker/Solver/Critic agents and dual feedback
(execution results + LLM critique) with dynamic pruning.

Based on:
- CodeTree (arxiv 2411.04329, NAACL 2025): HumanEval+ 86%, CodeContests 43%
- Key insight: dual feedback (test execution + LLM critique) with
  tree-structured exploration beats flat Best-of-N sampling

Architecture:
- Thinker: decomposes problem into solution strategies
- Solver: generates code for each strategy
- Critic: evaluates candidates with dual feedback
- Tree: explores promising branches, prunes dead ends

Selection uses dynamic pruning: branches with low scores after
K expansions are pruned. Beam width controls parallelism.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NodeStatus(Enum):
    PENDING = "pending"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    SELECTED = "selected"


@dataclass
class CodeNode:
    """A node in the code generation tree."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    strategy: str
    code: str = ""
    syntax_valid: bool = False
    exec_score: float = 0.0
    critique_score: float = 0.0
    combined_score: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    children: List[int] = field(default_factory=list)
    feedback: str = ""
    generation_ms: float = 0.0


@dataclass
class TreeSearchResult:
    """Result from CodeTree search."""
    answer: str
    total_nodes: int
    max_depth: int
    pruned_count: int
    selected_node_id: int
    selected_score: float
    all_scores: List[float] = field(default_factory=list)
    total_ms: float = 0.0
    strategies_explored: int = 0


# ---------------------------------------------------------------------------
# Verification (dual feedback)
# ---------------------------------------------------------------------------

def _check_syntax(code: str) -> Tuple[bool, str]:
    """Parse code and return (valid, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def _structural_score(code: str) -> float:
    """Score code structure: function count, nesting, length."""
    lines = code.strip().splitlines()
    if not lines:
        return 0.0

    scores = []

    # Has functions or classes (structured code)
    has_defs = bool(re.search(r"^(def |class )", code, re.MULTILINE))
    scores.append(0.3 if has_defs else 0.0)

    # Reasonable length (not trivially short)
    scores.append(min(0.2, len(lines) / 50.0))

    # Has docstrings or comments
    has_docs = bool(re.search(r'""".*?"""|\'\'\'.*?\'\'\'|#\s+\w', code, re.DOTALL))
    scores.append(0.1 if has_docs else 0.0)

    # Low nesting depth (clean code)
    max_indent = max(
        (len(line) - len(line.lstrip())) for line in lines if line.strip()
    )
    nesting_penalty = max(0, (max_indent - 16) / 32.0)
    scores.append(max(0, 0.2 - nesting_penalty))

    # Has error handling
    has_try = "try:" in code
    scores.append(0.1 if has_try else 0.0)

    # Has type hints
    has_types = bool(re.search(r":\s*(int|str|float|bool|List|Dict|Optional)", code))
    scores.append(0.1 if has_types else 0.0)

    return min(1.0, sum(scores))


def _exec_feedback(code: str, test_fn: Optional[Callable] = None) -> Tuple[float, str]:
    """
    Run execution-based feedback.

    If test_fn is provided, call it with the code and return (score, feedback).
    Otherwise, fall back to syntax + structural analysis.
    """
    syntax_ok, syntax_err = _check_syntax(code)
    if not syntax_ok:
        return 0.0, f"Syntax error: {syntax_err}"

    if test_fn is not None:
        try:
            score = test_fn(code)
            feedback = "passed" if score >= 0.8 else "partial"
            return min(1.0, max(0.0, score)), feedback
        except Exception as e:
            return 0.1, f"Execution error: {e}"

    # Fallback: structural analysis
    struct_score = _structural_score(code)
    return struct_score, "structural analysis only"


def _critique_feedback(
    code: str,
    problem: str,
    critique_fn: Optional[Callable] = None,
) -> Tuple[float, str]:
    """
    LLM-based critique of generated code.

    If critique_fn is provided, call it with (code, problem).
    Otherwise, fall back to heuristic critique.
    """
    if critique_fn is not None:
        try:
            score, feedback = critique_fn(code, problem)
            return min(1.0, max(0.0, score)), feedback
        except Exception as e:
            return 0.5, f"Critique error: {e}"

    # Heuristic critique (no LLM needed)
    issues = []
    score = 1.0

    if not code.strip():
        return 0.0, "Empty code"

    # Check for common anti-patterns
    if "pass" == code.strip():
        issues.append("trivial implementation")
        score -= 0.4

    if code.count("TODO") > 0:
        issues.append(f"{code.count('TODO')} TODOs found")
        score -= 0.1 * code.count("TODO")

    if "print(" in code and "return" not in code:
        issues.append("prints but no return value")
        score -= 0.15

    # Check relevance to problem
    problem_words = set(re.findall(r"[a-z]+", problem.lower()))
    code_words = set(re.findall(r"[a-z]+", code.lower()))
    overlap = len(problem_words & code_words)
    if overlap < 2:
        issues.append("low relevance to problem")
        score -= 0.2

    feedback = "; ".join(issues) if issues else "looks good"
    return max(0.0, score), feedback


# ---------------------------------------------------------------------------
# Strategy decomposition (Thinker)
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGIES = [
    "direct implementation",
    "helper function decomposition",
    "class-based with encapsulation",
]


def decompose_strategies(
    problem: str,
    strategy_fn: Optional[Callable] = None,
    max_strategies: int = 3,
) -> List[str]:
    """
    Thinker agent: decompose a problem into solution strategies.

    If strategy_fn is provided (LLM-backed), use it.
    Otherwise, return default strategy set.
    """
    if strategy_fn is not None:
        try:
            strategies = strategy_fn(problem)
            if isinstance(strategies, list) and strategies:
                return strategies[:max_strategies]
        except Exception:
            pass

    # Heuristic strategy selection based on problem keywords
    strategies = list(_DEFAULT_STRATEGIES)

    if any(kw in problem.lower() for kw in ["sort", "search", "optimize"]):
        strategies.append("iterative with early exit")

    if any(kw in problem.lower() for kw in ["tree", "graph", "recursive"]):
        strategies.append("recursive with memoization")

    if any(kw in problem.lower() for kw in ["parse", "validate", "format"]):
        strategies.append("regex-based parsing")

    return strategies[:max_strategies]


# ---------------------------------------------------------------------------
# CodeTree search engine
# ---------------------------------------------------------------------------

class CodeTreeSearch:
    """
    Tree search engine for code generation.

    Explores a tree of solution strategies, generating code at each
    node and evaluating with dual feedback (execution + critique).
    Prunes low-scoring branches and selects the best candidate.
    """

    def __init__(
        self,
        generate_fn: Callable[[str, str], str],
        beam_width: int = 3,
        max_depth: int = 3,
        prune_threshold: float = 0.3,
        exec_weight: float = 0.6,
        critique_weight: float = 0.4,
        test_fn: Optional[Callable] = None,
        critique_fn: Optional[Callable] = None,
        strategy_fn: Optional[Callable] = None,
    ):
        """
        Args:
            generate_fn: (problem, strategy) -> code string
            beam_width: Max children per node (controls branching)
            max_depth: Maximum tree depth
            prune_threshold: Prune branches below this score
            exec_weight: Weight for execution feedback in combined score
            critique_weight: Weight for critique feedback in combined score
            test_fn: Optional execution test function (code -> score)
            critique_fn: Optional LLM critique function (code, problem -> (score, feedback))
            strategy_fn: Optional LLM strategy decomposition (problem -> [strategies])
        """
        self.generate_fn = generate_fn
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold
        self.exec_weight = exec_weight
        self.critique_weight = critique_weight
        self.test_fn = test_fn
        self.critique_fn = critique_fn
        self.strategy_fn = strategy_fn

        self._nodes: Dict[int, CodeNode] = {}
        self._next_id = 0

    def _new_node(self, parent_id: Optional[int], depth: int,
                  strategy: str) -> CodeNode:
        nid = self._next_id
        self._next_id += 1
        node = CodeNode(node_id=nid, parent_id=parent_id,
                        depth=depth, strategy=strategy)
        self._nodes[nid] = node
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children.append(nid)
        return node

    def _evaluate_node(self, node: CodeNode, problem: str) -> None:
        """Dual feedback evaluation: execution + critique."""
        exec_score, exec_fb = _exec_feedback(node.code, self.test_fn)
        crit_score, crit_fb = _critique_feedback(
            node.code, problem, self.critique_fn)

        node.syntax_valid = exec_score > 0
        node.exec_score = exec_score
        node.critique_score = crit_score
        node.combined_score = (
            self.exec_weight * exec_score
            + self.critique_weight * crit_score
        )
        node.feedback = f"exec: {exec_fb} | critique: {crit_fb}"
        node.status = NodeStatus.EXPANDED

    def _refine_strategy(self, strategy: str, feedback: str,
                         depth: int) -> str:
        """Create a refined strategy based on feedback."""
        prefix = f"Iteration {depth}: "
        if "syntax error" in feedback.lower():
            return prefix + f"fix syntax in {strategy}"
        if "partial" in feedback.lower():
            return prefix + f"improve {strategy} based on test failures"
        if "low relevance" in feedback.lower():
            return prefix + f"align {strategy} more closely with requirements"
        return prefix + f"refine {strategy}"

    def search(self, problem: str) -> TreeSearchResult:
        """
        Run tree search on a code generation problem.

        Phase 1 (Thinker): Decompose into strategies
        Phase 2 (Solver): Generate code per strategy
        Phase 3 (Critic): Evaluate with dual feedback
        Phase 4 (Refine): Expand promising branches, prune weak ones
        Phase 5 (Select): Return best candidate
        """
        t0 = time.monotonic()
        self._nodes.clear()
        self._next_id = 0

        # Phase 1: Decompose into strategies
        strategies = decompose_strategies(
            problem, self.strategy_fn, self.beam_width)

        # Phase 2+3: Generate and evaluate root candidates
        for strategy in strategies:
            node = self._new_node(parent_id=None, depth=0, strategy=strategy)
            gen_t0 = time.monotonic()
            try:
                node.code = self.generate_fn(problem, strategy)
            except Exception as e:
                node.code = f"# Generation failed: {e}"
            node.generation_ms = (time.monotonic() - gen_t0) * 1000
            self._evaluate_node(node, problem)

        # Phase 4: Iterative refinement (depth 1 to max_depth)
        for depth in range(1, self.max_depth):
            # Select top beam_width nodes from current frontier
            frontier = [
                n for n in self._nodes.values()
                if n.depth == depth - 1
                and n.status == NodeStatus.EXPANDED
                and n.combined_score >= self.prune_threshold
            ]
            frontier.sort(key=lambda n: n.combined_score, reverse=True)
            frontier = frontier[:self.beam_width]

            if not frontier:
                break

            # Prune nodes not in frontier
            for n in self._nodes.values():
                if (n.depth == depth - 1
                        and n.status == NodeStatus.EXPANDED
                        and n not in frontier):
                    n.status = NodeStatus.PRUNED

            # Expand each frontier node
            for parent in frontier:
                refined = self._refine_strategy(
                    parent.strategy, parent.feedback, depth)
                child = self._new_node(
                    parent_id=parent.node_id, depth=depth,
                    strategy=refined)
                gen_t0 = time.monotonic()
                try:
                    child.code = self.generate_fn(problem, refined)
                except Exception as e:
                    child.code = f"# Generation failed: {e}"
                child.generation_ms = (time.monotonic() - gen_t0) * 1000
                self._evaluate_node(child, problem)

        # Phase 5: Select best
        all_expanded = [
            n for n in self._nodes.values()
            if n.status in (NodeStatus.EXPANDED, NodeStatus.SELECTED)
        ]
        if not all_expanded:
            return TreeSearchResult(
                answer="", total_nodes=len(self._nodes),
                max_depth=0, pruned_count=0, selected_node_id=-1,
                selected_score=0.0, total_ms=(time.monotonic() - t0) * 1000,
            )

        best = max(all_expanded, key=lambda n: n.combined_score)
        best.status = NodeStatus.SELECTED

        pruned = sum(
            1 for n in self._nodes.values()
            if n.status == NodeStatus.PRUNED
        )
        max_d = max(n.depth for n in self._nodes.values())

        return TreeSearchResult(
            answer=best.code,
            total_nodes=len(self._nodes),
            max_depth=max_d,
            pruned_count=pruned,
            selected_node_id=best.node_id,
            selected_score=best.combined_score,
            all_scores=[n.combined_score for n in all_expanded],
            total_ms=(time.monotonic() - t0) * 1000,
            strategies_explored=len(strategies),
        )

    def get_tree_report(self) -> Dict:
        """Return a structured report of the search tree."""
        nodes = []
        for n in self._nodes.values():
            nodes.append({
                "id": n.node_id,
                "parent": n.parent_id,
                "depth": n.depth,
                "strategy": n.strategy,
                "score": round(n.combined_score, 3),
                "exec": round(n.exec_score, 3),
                "critique": round(n.critique_score, 3),
                "status": n.status.value,
                "code_len": len(n.code),
            })
        return {
            "total_nodes": len(self._nodes),
            "nodes": nodes,
        }
