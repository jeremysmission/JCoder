"""CLI command: harvest -- fetch external research sources for ingestion."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from core.network_gate import NetworkGate
from research.harvester import ResearchHarvester

console = Console()


@click.command(name="harvest")
@click.option("--repo-root", default=".", help="Repo root for output directory")
@click.pass_context
def harvest_cmd(ctx, repo_root: str):
    """
    Harvest external research sources into markdown files for ingestion.

    SAFETY:
    - Requires network.mode == allowlist
    - Requires allowlist hosts configured
    - All requests enforced by NetworkGate.guard()
    """
    config = ctx.obj["config"]

    if not hasattr(config, "network"):
        raise click.ClickException(
            "Config missing 'network' section. Add network.mode and "
            "network.allowlist to config/default.yaml")

    if config.network.mode != "allowlist":
        raise click.ClickException(
            f"Harvest blocked: network.mode is '{config.network.mode}', "
            "must be 'allowlist'. This prevents accidental outbound traffic.")

    if not config.network.allowlist:
        raise click.ClickException(
            "Harvest blocked: network.allowlist is empty. Add hosts like "
            "api.github.com, pypi.org, hn.algolia.com, export.arxiv.org")

    gate = NetworkGate(mode=config.network.mode, allowlist=config.network.allowlist)

    harvester = ResearchHarvester(gate=gate, cfg=config.harvester, repo_root=repo_root)
    try:
        with console.status("Harvesting external sources..."):
            manifest = harvester.run()

        console.print("[bold green][OK] Harvest complete.")
        console.print(f"Written: {manifest['written_count']} files")
        if manifest["error_count"]:
            console.print(f"[yellow]Errors: {manifest['error_count']}[/yellow]")
            for e in manifest["errors"][:10]:
                console.print(f"  - {e}")
        out_dir = Path(repo_root) / config.harvester.out_dir
        console.print(f"Manifest: {out_dir / '_harvest_manifest.json'}")
    finally:
        harvester.close()
