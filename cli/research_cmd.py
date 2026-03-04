"""Click CLI commands for enhanced research pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group("research")
def research_cmd():
    """Enhanced research pipeline with PRISMA tracking and bias control."""
    pass


@research_cmd.command("sprint")
@click.argument("topic")
@click.option("--devils-advocate", is_flag=True, default=False,
              help="Enable devil's advocate bias checking")
@click.option("--verify-claims", is_flag=True, default=False,
              help="Enable cross-source claim verification")
@click.option("--max-papers", default=5, type=int,
              help="Max papers to deep-digest")
@click.pass_context
def sprint(ctx, topic, devils_advocate, verify_claims, max_papers):
    """Run enhanced research sprint on TOPIC."""
    config = ctx.obj["config"]
    mock = ctx.obj.get("mock", False)

    from core.config import _ms_to_seconds
    from core.mock_backend import MockLLM
    from core.network_gate import NetworkGate
    from core.research_sprint import ResearchSprinter, SprintConfig
    from core.runtime import Runtime

    gate = NetworkGate(mode=config.network.mode, allowlist=config.network.allowlist)

    if mock:
        runtime = MockLLM()
    else:
        runtime = Runtime(
            config.llm,
            _ms_to_seconds(config.policies.timeout_generate_ms),
            gate=gate,
        )

    sprint_config = SprintConfig(
        focus_topics=[topic],
        max_papers_to_digest=max_papers,
        output_dir=config.research.output_dir,
        prisma_enabled=config.research.prisma_enabled,
        credibility_scoring=config.research.credibility_scoring,
        devils_advocate=devils_advocate or config.research.devils_advocate,
        claim_verification=verify_claims or config.research.claim_verification,
        satellite_cutoff=config.research.satellite_cutoff,
        drone_cutoff=config.research.drone_cutoff,
        max_deep_dive=config.research.max_deep_dive,
        max_counter_queries=config.research.max_counter_queries,
        max_verify_claims=config.research.max_verify_claims,
        synthesis_max_themes=config.research.synthesis_max_themes,
    )

    sprinter = ResearchSprinter(
        runtime=runtime,
        config=sprint_config,
    )

    console.print(f"[bold]Starting research sprint: {topic}[/bold]")
    result = sprinter.run_sprint(topics=[topic])

    # Display results
    console.print(f"\n[bold green][OK] Sprint complete: {result.sprint_id}")
    console.print(f"  Duration:    {result.duration_seconds:.1f}s")
    console.print(f"  Discovered:  {result.papers_discovered}")
    console.print(f"  Screened:    {result.papers_triaged}")
    console.print(f"  Digested:    {result.papers_digested}")
    console.print(f"  Prototypes:  {result.prototypes_generated}")
    console.print(f"  Report:      {result.report_path}")

    if result.prisma_flow:
        console.print("\n[bold]PRISMA Flow:[/bold]")
        for stage, count in result.prisma_flow.items():
            console.print(f"  {stage}: {count}")

    if result.verification_count > 0:
        console.print(f"\n  Verified claims: {result.verification_count}")

    if result.balance_ratio >= 0:
        console.print(f"  Balance ratio:   {result.balance_ratio:.2f}")

    if result.synthesis_markdown:
        console.print(f"\n[bold]Synthesis Matrix:[/bold]")
        console.print(result.synthesis_markdown)


@research_cmd.command("verify")
@click.argument("claim")
@click.pass_context
def verify(ctx, claim):
    """Cross-source verification of a CLAIM."""
    config = ctx.obj["config"]
    mock = ctx.obj.get("mock", False)

    from core.claim_verifier import ClaimVerifier
    from core.config import _ms_to_seconds
    from core.mock_backend import MockLLM
    from core.network_gate import NetworkGate
    from core.runtime import Runtime

    gate = NetworkGate(mode=config.network.mode, allowlist=config.network.allowlist)

    if mock:
        runtime = MockLLM()
    else:
        runtime = Runtime(
            config.llm,
            _ms_to_seconds(config.policies.timeout_generate_ms),
            gate=gate,
        )

    verifier = ClaimVerifier(runtime=runtime)
    result = verifier.verify(claim)

    status = "[bold green]VERIFIED" if result.verified else "[bold red]UNVERIFIED"
    console.print(f"\nClaim: {claim}")
    console.print(f"Status: {status}")
    console.print(f"Confidence: {result.confidence:.2f}")
    console.print(f"Method: {result.verification_method}")
    console.print(f"Corroborating: {len(result.corroborating_sources)}")
    console.print(f"Contradicting: {len(result.contradicting_sources)}")


@research_cmd.command("status")
@click.pass_context
def status(ctx):
    """Show PRISMA pipeline status from latest sprint."""
    config = ctx.obj["config"]
    research_dir = Path(config.research.output_dir)

    if not research_dir.exists():
        console.print("[WARN] No research output directory found.")
        return

    # Find latest sprint
    sprint_dirs = sorted(research_dir.glob("sprint_*"), reverse=True)
    if not sprint_dirs:
        console.print("[WARN] No sprint results found.")
        return

    latest = sprint_dirs[0]
    console.print(f"[bold]Latest sprint: {latest.name}[/bold]\n")

    # Show PRISMA flow if available
    prisma_db = latest / "prisma.db"
    if prisma_db.exists():
        from core.prisma_tracker import PrismaTracker
        tracker = PrismaTracker(db_path=str(prisma_db))
        console.print(tracker.flow_diagram_text())
        tracker.close()

    # Show synthesis matrix if available
    synthesis = latest / "synthesis_matrix.md"
    if synthesis.exists():
        console.print(f"\n[bold]Synthesis Matrix:[/bold]")
        console.print(synthesis.read_text(encoding="utf-8"))

    # Show report if available
    report = latest / "report.md"
    if report.exists():
        console.print(f"\n[bold]Report:[/bold] {report}")
