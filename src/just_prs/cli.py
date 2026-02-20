"""Typer CLI for just-prs: PGS Catalog exploration and PRS computation."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

import polars as pl

from just_prs.catalog import PGSCatalogClient
from just_prs.ftp import METADATA_FILES, bulk_download_scoring_parquets, download_all_metadata, download_metadata_sheet, list_all_pgs_ids
from just_prs.hf import pull_cleaned_parquets
from just_prs.models import PRSResult
from just_prs.normalize import VcfFilterConfig, normalize_vcf
from just_prs.prs import compute_prs, compute_prs_batch
from just_prs.prs_catalog import PRSCatalog
from just_prs.scoring import DEFAULT_CACHE_DIR, download_scoring_file

app = typer.Typer(
    name="just-prs",
    help="Polars-bio based tool to compute polygenic risk scores from PGS Catalog.",
    no_args_is_help=True,
)

catalog_app = typer.Typer(
    name="catalog",
    help="Explore the PGS Catalog: search scores, traits, and download scoring files.",
    no_args_is_help=True,
)
app.add_typer(catalog_app, name="catalog")

scores_app = typer.Typer(
    name="scores",
    help="Search and inspect PGS scores.",
    no_args_is_help=True,
)
catalog_app.add_typer(scores_app, name="scores")

traits_app = typer.Typer(
    name="traits",
    help="Search and inspect traits.",
    no_args_is_help=True,
)
catalog_app.add_typer(traits_app, name="traits")

bulk_app = typer.Typer(
    name="bulk",
    help="Bulk FTP downloads: metadata CSVs and scoring files as parquet.",
    no_args_is_help=True,
)
catalog_app.add_typer(bulk_app, name="bulk")

console = Console()


@scores_app.command("list")
def scores_list(
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Max results to show (default: 100)"),
    ] = 100,
    all_scores: Annotated[
        bool,
        typer.Option("--all", "-a", help="Fetch and list all scores in catalog"),
    ] = False,
) -> None:
    """List PGS Catalog scores (paginated from /score/all)."""
    max_results: int | None = None if all_scores else (limit or 100)
    with PGSCatalogClient() as client:
        results = list(
            client.iter_all_scores(page_size=50, max_results=max_results)
        )

    if not results:
        console.print("[yellow]No scores found.[/yellow]")
        return

    table = Table(
        title=f"PGS Catalog Scores ({len(results)} shown)"
        + (" (all)" if all_scores else "")
    )
    table.add_column("PGS ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Trait", style="magenta")
    table.add_column("Variants", justify="right")
    table.add_column("Publication")

    for s in results:
        pub = s.publication.firstauthor if s.publication else ""
        table.add_row(
            s.id,
            s.name or "",
            s.trait_reported or "",
            str(s.variants_number or ""),
            pub or "",
        )

    console.print(table)


@scores_app.command("search")
def scores_search(
    term: Annotated[str, typer.Option("--term", "-t", help="Search term")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 25,
) -> None:
    """Search PGS Catalog scores by term (trait name, gene, etc.)."""
    with PGSCatalogClient() as client:
        results = client.search_scores(term, limit=limit)

    if not results:
        console.print(f"[yellow]No scores found for '{term}'[/yellow]")
        return

    table = Table(title=f"PGS Scores matching '{term}'")
    table.add_column("PGS ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Trait", style="magenta")
    table.add_column("Variants", justify="right")
    table.add_column("Publication")

    for s in results:
        pub = s.publication.firstauthor if s.publication else ""
        table.add_row(
            s.id,
            s.name or "",
            s.trait_reported or "",
            str(s.variants_number or ""),
            pub or "",
        )

    console.print(table)


@scores_app.command("info")
def scores_info(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
) -> None:
    """Show detailed information about a PGS score."""
    with PGSCatalogClient() as client:
        score = client.get_score(pgs_id)

    console.print(f"\n[bold cyan]{score.id}[/bold cyan] - {score.name or 'N/A'}")
    console.print(f"  Trait: [magenta]{score.trait_reported or 'N/A'}[/magenta]")
    console.print(f"  Variants: {score.variants_number or 'N/A'}")
    console.print(f"  Genome Build: {score.variants_genomebuild or 'N/A'}")
    console.print(f"  Weight Type: {score.weight_type or 'N/A'}")

    if score.publication:
        pub = score.publication
        console.print(f"  Publication: {pub.title or 'N/A'}")
        console.print(f"    Author: {pub.firstauthor or 'N/A'}")
        console.print(f"    Journal: {pub.journal or 'N/A'} ({pub.date_publication or 'N/A'})")
        if pub.PMID:
            console.print(f"    PMID: {pub.PMID}")
        if pub.doi:
            console.print(f"    DOI: {pub.doi}")

    if score.ftp_harmonized_scoring_files:
        console.print("  Harmonized files:")
        for build, info in score.ftp_harmonized_scoring_files.items():
            url = info.positions or "N/A"
            console.print(f"    {build}: {url}")

    if score.license:
        console.print(f"  License: {score.license[:100]}...")
    console.print()


@traits_app.command("search")
def traits_search(
    term: Annotated[str, typer.Option("--term", "-t", help="Search term")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 25,
) -> None:
    """Search PGS Catalog traits by term."""
    with PGSCatalogClient() as client:
        results = client.search_traits(term, limit=limit)

    if not results:
        console.print(f"[yellow]No traits found for '{term}'[/yellow]")
        return

    table = Table(title=f"Traits matching '{term}'")
    table.add_column("EFO ID", style="cyan")
    table.add_column("Label", style="green")
    table.add_column("Scores", justify="right", style="magenta")
    table.add_column("Categories")

    for t in results:
        table.add_row(
            t.id,
            t.label or "",
            str(len(t.associated_pgs_ids)),
            ", ".join(t.trait_categories) if t.trait_categories else "",
        )

    console.print(table)


@traits_app.command("info")
def traits_info(
    efo_id: Annotated[str, typer.Argument(help="EFO trait ID (e.g. EFO_0001645)")],
) -> None:
    """Show detailed information about a trait."""
    with PGSCatalogClient() as client:
        trait = client.get_trait(efo_id)

    console.print(f"\n[bold cyan]{trait.id}[/bold cyan] - {trait.label or 'N/A'}")
    if trait.description:
        desc = trait.description[:200] + "..." if len(trait.description) > 200 else trait.description
        console.print(f"  Description: {desc}")
    if trait.url:
        console.print(f"  URL: {trait.url}")
    if trait.trait_categories:
        console.print(f"  Categories: {', '.join(trait.trait_categories)}")
    if trait.associated_pgs_ids:
        console.print(f"  Associated PGS IDs ({len(trait.associated_pgs_ids)}):")
        for chunk_start in range(0, len(trait.associated_pgs_ids), 10):
            chunk = trait.associated_pgs_ids[chunk_start:chunk_start + 10]
            console.print(f"    {', '.join(chunk)}")
    console.print()


@bulk_app.command("metadata")
def bulk_metadata(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to write parquet files"),
    ] = Path("./output/pgs_metadata"),
    sheet: Annotated[
        Optional[str],
        typer.Option(
            "--sheet",
            "-s",
            help=f"Single sheet to download. One of: {', '.join(METADATA_FILES)}. "
            "Omit to download all sheets.",
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-download and overwrite existing files")
    ] = False,
) -> None:
    """Bulk-download PGS Catalog metadata CSVs from EBI FTP and save as parquet.

    Downloads pre-built catalog-wide CSVs in a single HTTP request each —
    far faster than paginating the REST API.
    """
    if sheet is not None:
        if sheet not in METADATA_FILES:
            console.print(
                f"[red]Unknown sheet '{sheet}'. "
                f"Choose from: {', '.join(METADATA_FILES)}[/red]"
            )
            raise typer.Exit(code=1)
        output_path = output_dir / f"{sheet}.parquet"
        console.print(f"Downloading sheet [cyan]{sheet}[/cyan] → {output_path} ...")
        df = download_metadata_sheet(sheet, output_path, overwrite=overwrite)  # type: ignore[arg-type]
        console.print(f"[green]Saved {len(df):,} rows to {output_path}[/green]")
    else:
        console.print(f"Downloading all {len(METADATA_FILES)} metadata sheets → {output_dir} ...")
        results = download_all_metadata(output_dir, overwrite=overwrite)
        table = Table(title="PGS Catalog Metadata Downloaded")
        table.add_column("Sheet", style="cyan")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Parquet", style="dim")
        for name, df in results.items():
            table.add_row(name, f"{len(df):,}", str(output_dir / f"{name}.parquet"))
        console.print(table)


@bulk_app.command("scores")
def bulk_scores(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to write per-score parquet files"),
    ] = Path("./output/pgs_scores"),
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    pgs_ids: Annotated[
        Optional[str],
        typer.Option(
            "--ids",
            help="Comma-separated PGS IDs to download. Omit to download all.",
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-download and overwrite existing parquet files")
    ] = False,
) -> None:
    """Bulk-download PGS scoring files from EBI FTP and save each as parquet.

    When --ids is omitted, fetches the full ID list from pgs_scores_list.txt
    first (one request), then streams each scoring file without storing
    intermediate .gz files locally.
    """
    ids: list[str] | None = None
    if pgs_ids is not None:
        ids = [p.strip() for p in pgs_ids.split(",") if p.strip()]

    if ids is None:
        console.print("Fetching full PGS ID list from EBI FTP...")
        ids = list_all_pgs_ids()
        console.print(f"Found [cyan]{len(ids):,}[/cyan] scores. Starting download → {output_dir}")
    else:
        console.print(f"Downloading [cyan]{len(ids)}[/cyan] score(s) → {output_dir}")

    paths = bulk_download_scoring_parquets(
        output_dir=output_dir,
        genome_build=build,
        pgs_ids=ids,
        overwrite=overwrite,
    )
    console.print(f"[green]Done. {len(paths):,} parquet files in {output_dir}[/green]")


@bulk_app.command("ids")
def bulk_ids() -> None:
    """Print the full list of PGS IDs from EBI FTP (pgs_scores_list.txt)."""
    ids = list_all_pgs_ids()
    for pgs_id in ids:
        console.print(pgs_id)
    console.print(f"\n[dim]Total: {len(ids):,} scores[/dim]")


@bulk_app.command("clean-metadata")
def bulk_clean_metadata(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to write cleaned parquet files"),
    ] = Path("./output/pgs_metadata"),
) -> None:
    """Download raw metadata from EBI FTP, run the cleanup pipeline, and save cleaned parquets.

    Produces three files: scores.parquet, performance.parquet, best_performance.parquet.
    These contain normalized genome builds, snake_case columns, and parsed numeric metrics.
    """
    console.print(f"Building cleaned metadata parquets → {output_dir} ...")
    catalog = PRSCatalog()
    paths = catalog.build_cleaned_parquets(output_dir=output_dir)

    table = Table(title="Cleaned PGS Metadata")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Parquet", style="dim")
    for name, path in paths.items():
        df = pl.read_parquet(path)
        table.add_row(name, f"{df.height:,}", str(path))
    console.print(table)


@bulk_app.command("push-hf")
def bulk_push_hf(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory containing cleaned parquets to push"),
    ] = Path("./output/pgs_metadata"),
    repo_id: Annotated[
        str,
        typer.Option("--repo", "-r", help="HuggingFace dataset repo ID"),
    ] = "just-dna-seq/polygenic_risk_scores",
) -> None:
    """Push cleaned metadata parquets to a HuggingFace dataset repository.

    Builds cleaned parquets first if they don't exist in output-dir.
    Token is read from .env file or HF_TOKEN environment variable.
    """
    catalog = PRSCatalog()
    if not all((output_dir / f).exists() for f in ("scores.parquet", "performance.parquet", "best_performance.parquet")):
        console.print(f"Cleaned parquets not found in {output_dir}, building them first...")
        catalog.build_cleaned_parquets(output_dir=output_dir)

    console.print(f"Pushing cleaned parquets to [cyan]{repo_id}[/cyan] ...")
    from just_prs.hf import push_cleaned_parquets
    push_cleaned_parquets(output_dir, repo_id=repo_id)
    console.print(f"[green]Pushed to https://huggingface.co/datasets/{repo_id}[/green]")


@bulk_app.command("pull-hf")
def bulk_pull_hf(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to save pulled parquet files"),
    ] = Path("./output/pgs_metadata"),
    repo_id: Annotated[
        str,
        typer.Option("--repo", "-r", help="HuggingFace dataset repo ID"),
    ] = "just-dna-seq/polygenic_risk_scores",
) -> None:
    """Pull cleaned metadata parquets from a HuggingFace dataset repository.

    Downloads scores.parquet, performance.parquet, and best_performance.parquet
    from the data/ folder of the HF repo into the output directory.
    """
    console.print(f"Pulling cleaned parquets from [cyan]{repo_id}[/cyan] → {output_dir} ...")
    downloaded = pull_cleaned_parquets(output_dir, repo_id=repo_id)

    table = Table(title="Pulled from HuggingFace")
    table.add_column("File", style="cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Path", style="dim")
    for path in downloaded:
        df = pl.read_parquet(path)
        table.add_row(path.name, f"{df.height:,}", str(path))
    console.print(table)


@catalog_app.command("download")
def catalog_download(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory")
    ] = Path("./output/scores"),
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build")
    ] = "GRCh38",
) -> None:
    """Download a PGS scoring file."""
    console.print(f"Downloading {pgs_id} ({build})...")
    path = download_scoring_file(pgs_id, output_dir, genome_build=build)
    console.print(f"[green]Saved to {path}[/green]")


@app.command("compute")
def compute(
    vcf: Annotated[Path, typer.Option("--vcf", "-v", help="Path to VCF file")],
    pgs_id: Annotated[
        str, typer.Option("--pgs-id", "-p", help="PGS ID(s), comma-separated")
    ],
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build")
    ] = "GRCh38",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Cache directory for scoring files")
    ] = DEFAULT_CACHE_DIR,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSON file")
    ] = None,
) -> None:
    """Compute polygenic risk score(s) for a VCF file."""
    pgs_ids = [pid.strip() for pid in pgs_id.split(",")]

    console.print(f"Computing PRS for {len(pgs_ids)} score(s) on {vcf}...")

    results: list[PRSResult]
    if len(pgs_ids) == 1:
        with PGSCatalogClient() as client:
            score_info = client.get_score(pgs_ids[0])
        result = compute_prs(
            vcf_path=vcf,
            scoring_file=pgs_ids[0],
            genome_build=build,
            cache_dir=cache_dir,
            pgs_id=pgs_ids[0],
            trait_reported=score_info.trait_reported,
        )
        results = [result]
    else:
        results = compute_prs_batch(
            vcf_path=vcf,
            pgs_ids=pgs_ids,
            genome_build=build,
            cache_dir=cache_dir,
        )

    table = Table(title="PRS Results")
    table.add_column("PGS ID", style="cyan")
    table.add_column("Trait", style="magenta")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Matched", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Match Rate", justify="right")

    for r in results:
        table.add_row(
            r.pgs_id,
            r.trait_reported or "",
            f"{r.score:.6f}",
            str(r.variants_matched),
            str(r.variants_total),
            f"{r.match_rate:.1%}",
        )

    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in results]
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[green]Results saved to {output}[/green]")


@app.command("normalize")
def normalize(
    vcf: Annotated[Path, typer.Option("--vcf", "-v", help="Path to VCF file (.vcf or .vcf.gz)")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output Parquet path. Defaults to data/output/results/<stem>.parquet"),
    ] = None,
    pass_filters: Annotated[
        Optional[str],
        typer.Option("--pass-filters", help='Comma-separated FILTER values to keep (e.g. "PASS,.")'),
    ] = None,
    min_depth: Annotated[
        Optional[int],
        typer.Option("--min-depth", help="Minimum DP (read depth) to keep a variant"),
    ] = None,
    min_qual: Annotated[
        Optional[float],
        typer.Option("--min-qual", help="Minimum QUAL score to keep a variant"),
    ] = None,
    sex: Annotated[
        Optional[str],
        typer.Option("--sex", help='Sample sex ("Male" or "Female"). Warns if Female has chrY variants.'),
    ] = None,
    format_fields_str: Annotated[
        Optional[str],
        typer.Option("--format-fields", help='Comma-separated FORMAT fields to include (default "GT,DP")'),
    ] = None,
) -> None:
    """Normalize a VCF file: strip chr prefix, compute genotype, apply quality filters, write Parquet."""
    if output is None:
        output = Path("data/output/results") / (vcf.stem.removesuffix(".vcf") + ".parquet")

    config = VcfFilterConfig(
        pass_filters=[f.strip() for f in pass_filters.split(",") if f.strip()] if pass_filters else None,
        min_depth=min_depth,
        min_qual=min_qual,
        sex=sex,
    )

    format_fields: list[str] | None = None
    if format_fields_str is not None:
        format_fields = [f.strip() for f in format_fields_str.split(",") if f.strip()]

    console.print(f"Normalizing [cyan]{vcf}[/cyan] → {output} ...")
    if config.pass_filters:
        console.print(f"  FILTER keep: {config.pass_filters}")
    if config.min_depth is not None:
        console.print(f"  Min DP: {config.min_depth}")
    if config.min_qual is not None:
        console.print(f"  Min QUAL: {config.min_qual}")
    if config.sex:
        console.print(f"  Sex: {config.sex}")

    result_path = normalize_vcf(vcf, output, config=config, format_fields=format_fields)

    df = pl.read_parquet(result_path)
    console.print(f"[green]Wrote {df.height:,} variants ({len(df.columns)} columns) to {result_path}[/green]")

    table = Table(title="Normalized VCF Summary")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="green")
    for col_name, col_type in zip(df.columns, df.dtypes):
        table.add_row(col_name, str(col_type))
    console.print(table)


@app.command("pipeline")
def pipeline(
    host: Annotated[str, typer.Option("--host", help="Dagster webserver host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Dagster webserver port")] = 3010,
) -> None:
    """Launch the Dagster reference-panel pipeline UI and auto-trigger the full pipeline job."""
    launch_pipeline(host=host, port=port)


def _find_project_root() -> Path:
    """Walk upward from cwd to find the uv workspace root (contains [tool.uv.workspace])."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text():
            return candidate
    return current


def _ensure_dagster_yaml(dagster_home: Path) -> None:
    """Write dagster.yaml with telemetry disabled if missing."""
    yaml_path = dagster_home / "dagster.yaml"
    if yaml_path.exists():
        return
    yaml_path.write_text("telemetry:\n  enabled: false\n")
    console.print(f"[dim]Created {yaml_path}[/dim]")


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port (SIGTERM, then SIGKILL)."""
    import signal
    import subprocess
    import time

    result = subprocess.run(
        ["lsof", "-t", f"-iTCP:{port}"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
    if pids:
        console.print(f"[yellow]Port {port} in use by PIDs {pids}, terminating...[/yellow]")
        for pid in pids:
            import os as _os
            _os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        # SIGKILL anything that survived
        result2 = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}"],
            capture_output=True, text=True,
        )
        for pid in [int(p) for p in result2.stdout.strip().splitlines() if p.strip()]:
            import os as _os
            _os.kill(pid, signal.SIGKILL)


def _cancel_orphaned_runs() -> None:
    """Cancel any runs stuck in STARTED or NOT_STARTED from a previous session."""
    from dagster import DagsterInstance, DagsterRunStatus, RunsFilter

    with DagsterInstance.get() as instance:
        stuck = instance.get_run_records(
            filters=RunsFilter(statuses=[DagsterRunStatus.STARTED, DagsterRunStatus.NOT_STARTED])
        )
        for record in stuck:
            console.print(
                f"[yellow]Cancelling orphaned run {record.dagster_run.run_id[:8]}...[/yellow]"
            )
            instance.report_run_canceled(
                record.dagster_run,
                message="Orphaned run from previous session",
            )


def _wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Block until a TCP connection to host:port succeeds, or timeout expires."""
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _run_job_in_background(job_name: str) -> None:
    """Execute a named asset job in a background thread via execute_in_process."""
    import threading

    from dagster import DagsterInstance
    from prs_pipeline.definitions import defs

    job_def = defs.get_job_def(job_name)

    def _execute() -> None:
        with DagsterInstance.get() as instance:
            console.print(f"[green]Starting job '{job_name}' in background...[/green]")
            result = job_def.execute_in_process(instance=instance)
            if result.success:
                console.print(f"[green]Job '{job_name}' completed successfully.[/green]")
            else:
                console.print(f"[red]Job '{job_name}' failed.[/red]")

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()


def launch_pipeline(host: str = "0.0.0.0", port: int = 3010) -> None:
    """Start the Dagster dev server and submit the full_reference_pipeline job.

    1. Set DAGSTER_HOME to data/output/dagster inside the workspace root.
    2. Generate dagster.yaml (telemetry off) if missing.
    3. Kill any orphaned process holding the target port.
    4. Cancel any orphaned runs from previous sessions.
    5. Start dagster dev as a background Popen.
    6. Wait for the webserver to be ready (poll the port).
    7. Submit the full_reference_pipeline job to the daemon.
    8. Forward SIGINT/SIGTERM to dagster and wait for it to exit.
    """
    import os
    import signal
    import subprocess
    import sys
    from importlib.util import find_spec

    if find_spec("prs_pipeline") is None:
        console.print(
            "[red]prs-pipeline package not installed. "
            "Install it with: uv sync --all-packages[/red]"
        )
        raise typer.Exit(code=1)

    project_root = _find_project_root()
    dagster_home = project_root / "data" / "output" / "dagster"
    dagster_home.mkdir(parents=True, exist_ok=True)
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    console.print(f"[dim]DAGSTER_HOME={dagster_home}[/dim]")

    _ensure_dagster_yaml(dagster_home)
    _kill_port(port)
    _cancel_orphaned_runs()

    dagster_bin = str(Path(sys.executable).parent / "dagster")
    console.print(f"Starting Dagster dev at http://{host}:{port} ...")
    proc = subprocess.Popen(
        [dagster_bin, "dev", "-m", "prs_pipeline.definitions",
         "--host", host, "--port", str(port)],
    )

    def _forward_signal(sig: int, _frame: object) -> None:
        proc.send_signal(sig)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    if _wait_for_port(connect_host, port, timeout=90):
        console.print(f"[green]Dagster UI ready at http://{host}:{port}[/green]")
        try:
            _run_job_in_background("full_reference_pipeline")
        except Exception as exc:
            console.print(f"[yellow]Could not start job: {exc}[/yellow]")
            console.print("[yellow]You can start it manually from the Dagster UI.[/yellow]")
    else:
        console.print("[yellow]Timed out waiting for Dagster webserver — UI may still be starting.[/yellow]")

    proc.wait()


@app.command("ui")
def ui() -> None:
    """Launch the PRS web UI (Reflex app)."""
    launch_ui()


def launch_ui() -> None:
    """Start the Reflex dev server for prs-ui."""
    import os
    from importlib.util import find_spec

    if find_spec("prs_ui") is None:
        console.print(
            "[red]prs-ui package not installed. "
            "Install it with: uv sync --all-packages[/red]"
        )
        raise typer.Exit(code=1)

    prs_ui_dir = Path(find_spec("prs_ui").origin).resolve().parent.parent  # type: ignore[union-attr]
    console.print(f"Starting PRS UI from {prs_ui_dir} ...")
    os.chdir(prs_ui_dir)
    from reflex.reflex import cli as reflex_cli
    reflex_cli(["run"])


def run() -> None:
    """Entry point for the CLI."""
    app()
