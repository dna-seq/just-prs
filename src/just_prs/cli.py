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
    ] = Path("./pgs_metadata"),
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
    ] = Path("./pgs_scores"),
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
    ] = Path("./pgs_metadata"),
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
    ] = Path("./pgs_metadata"),
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
    ] = Path("./pgs_metadata"),
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
    ] = Path("./scores"),
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


def run() -> None:
    """Entry point for the CLI."""
    app()
