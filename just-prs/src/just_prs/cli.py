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
from just_prs.scoring import DEFAULT_CACHE_DIR, download_scoring_file, resolve_cache_dir

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

reference_app = typer.Typer(
    name="reference",
    help="Reference panel operations: score PGS IDs against population panels (1000G, HGDP+1kGP) using pgenlib + polars.",
    no_args_is_help=True,
)
app.add_typer(reference_app, name="reference")

pgen_app = typer.Typer(
    name="pgen",
    help="PLINK2 binary format (.pgen/.pvar/.psam) operations — pure Python via pgenlib.",
    no_args_is_help=True,
)
app.add_typer(pgen_app, name="pgen")

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

    def _cli_progress(payload: dict[str, int]) -> None:
        completed = payload["completed"]
        total = payload["total"]
        percent = (completed / total) * 100 if total > 0 else 0
        console.print(f"Progress: [cyan]{completed}/{total}[/cyan] ([yellow]{percent:.1f}%[/yellow])")

    paths = bulk_download_scoring_parquets(
        output_dir=output_dir,
        genome_build=build,
        pgs_ids=ids,
        overwrite=overwrite,
        progress_callback=_cli_progress,
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


@bulk_app.command("push-catalog")
def bulk_push_catalog(
    repo_id: Annotated[
        str,
        typer.Option("--repo", "-r", help="HuggingFace dataset repo ID"),
    ] = "just-dna-seq/pgs-catalog",
    build: Annotated[
        str,
        typer.Option("--build", "-b", help="Genome build for scoring files"),
    ] = "GRCh38",
    ids: Annotated[
        Optional[str],
        typer.Option("--ids", help="Comma-separated PGS IDs (default: all from PGS Catalog)"),
    ] = None,
    skip_download: Annotated[
        bool,
        typer.Option("--skip-download", help="Skip downloading, only convert and upload what is cached"),
    ] = False,
    delete_gz: Annotated[
        bool,
        typer.Option("--delete-gz", help="Delete .txt.gz files after verified parquet conversion"),
    ] = False,
) -> None:
    """Download, convert, and upload PGS Catalog scoring files + metadata to HuggingFace.

    \b
    Full pipeline in one command:
    1. Download all scoring .txt.gz files from EBI FTP (skips existing)
    2. Convert each .txt.gz to parquet (skips existing)
    3. Build cleaned metadata parquets if missing
    4. Upload everything to the HF dataset repo

    Token is read from .env file or HF_TOKEN environment variable.
    """
    from just_prs.ftp import bulk_download_scoring_files
    from just_prs.hf import push_pgs_catalog
    from just_prs.scoring import _scoring_parquet_cache_path, parse_scoring_file

    cache = resolve_cache_dir()
    scores_dir = cache / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = cache / "metadata"

    pgs_ids: list[str]
    if ids is not None:
        pgs_ids = [p.strip() for p in ids.split(",") if p.strip()]
        console.print(f"Using {len(pgs_ids)} specified PGS IDs")
    else:
        console.print("Fetching full PGS ID list from EBI FTP...")
        pgs_ids = list_all_pgs_ids()
        console.print(f"Found [cyan]{len(pgs_ids):,}[/cyan] PGS IDs in catalog")

    if not skip_download:
        console.print(f"\n[bold]Step 1/4:[/bold] Downloading scoring files ({build})...")
        
        def _download_progress(payload: dict[str, int]) -> None:
            completed = payload["completed"]
            total = payload["total"]
            downloaded = payload["downloaded"]
            cached = payload["cached"]
            percent = (completed / total) * 100 if total > 0 else 0
            console.print(
                f"Progress: [cyan]{completed}/{total}[/cyan] ([yellow]{percent:.1f}%[/yellow]) "
                f"dl={downloaded} cached={cached}"
            )
        
        result = bulk_download_scoring_files(
            pgs_ids=pgs_ids,
            output_dir=scores_dir,
            genome_build=build,
            progress_callback=_download_progress,
        )
        console.print(
            f"  downloaded={result.downloaded}, cached={result.cached}, "
            f"failed={result.failed}"
        )

    console.print(f"\n[bold]Step 2/4:[/bold] Converting .txt.gz to parquet...")
    gz_files = sorted(scores_dir.glob("*_hmPOS_*.txt.gz"))
    converted = 0
    already_cached = 0
    failed = 0
    deleted = 0
    
    import time
    last_log_time = time.monotonic()
    total_gz = len(gz_files)
    
    for i, gz_path in enumerate(gz_files):
        parquet_path = _scoring_parquet_cache_path(gz_path)
        if parquet_path.exists():
            already_cached += 1
            if delete_gz:
                gz_path.unlink()
                deleted += 1
        else:
            try:
                lf = parse_scoring_file(gz_path)
                lf.select(pl.len()).collect()
                if parquet_path.exists():
                    converted += 1
                    if delete_gz:
                        gz_path.unlink()
                        deleted += 1
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                console.print(f"  [red]{gz_path.name}: {exc}[/red]")
                
        current_time = time.monotonic()
        if (i + 1) % 500 == 0 or (current_time - last_log_time > 15.0) or (i + 1) == total_gz:
            last_log_time = current_time
            percent = ((i + 1) / total_gz) * 100 if total_gz > 0 else 0
            console.print(f"  Parquet conversion: [cyan]{i + 1}/{total_gz}[/cyan] ([yellow]{percent:.1f}%[/yellow]) "
                          f"converted={converted} cached={already_cached} failed={failed}")
    console.print(
        f"  converted={converted}, already_cached={already_cached}, "
        f"failed={failed}, deleted_gz={deleted}"
    )

    console.print(f"\n[bold]Step 3/4:[/bold] Building cleaned metadata...")
    if not all((metadata_dir / f).exists() for f in ("scores.parquet", "performance.parquet", "best_performance.parquet")):
        catalog = PRSCatalog()
        catalog.build_cleaned_parquets(output_dir=metadata_dir)
        console.print("  Built cleaned metadata parquets")
    else:
        console.print("  Cleaned metadata already exists, skipping")

    scoring_parquets = [
        p for p in scores_dir.glob("*_hmPOS_*.parquet")
        if p.name != "conversion_failures.parquet"
    ]
    console.print(
        f"\n[bold]Step 4/4:[/bold] Uploading to [cyan]{repo_id}[/cyan]...\n"
        f"  metadata: {metadata_dir} (3 parquets)\n"
        f"  scores:   {scores_dir} ({len(scoring_parquets):,} parquets)"
    )
    push_pgs_catalog(
        metadata_dir=metadata_dir,
        scores_dir=scores_dir,
        repo_id=repo_id,
    )
    console.print(f"\n[green]Done. Pushed to https://huggingface.co/datasets/{repo_id}[/green]")


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


def _print_reference_score_table(pgs_id: str, result_df: pl.DataFrame) -> None:
    """Print per-superpopulation reference panel score statistics."""
    table = Table(title=f"Reference Panel Scores: {pgs_id}")
    table.add_column("Superpop", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    stats = (
        result_df.group_by("superpop")
        .agg(
            pl.col("score").count().alias("n"),
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("superpop")
    )
    for row in stats.iter_rows(named=True):
        table.add_row(
            row["superpop"],
            str(row["n"]),
            f"{row['mean']:.8f}",
            f"{row['std']:.8f}",
            f"{row['min']:.8f}",
            f"{row['max']:.8f}",
        )
    console.print(table)
    console.print(f"[green]Total samples: {result_df.height}[/green]")


def _print_polars_timing_table(done: dict) -> None:
    """Print timing breakdown for the polars scoring engine."""
    timing_table = Table(title="Timing & Resources")
    timing_table.add_column("Metric", style="cyan")
    timing_table.add_column("Value", justify="right", style="green")

    timing_table.add_row("Total time", f"{done.get('total_elapsed_sec', '?')}s")
    timing_table.add_row("  Scoring file parse", f"{done.get('scoring_sec', '?')}s")
    timing_table.add_row("  Variant matching", f"{done.get('pvar_sec', '?')}s")
    timing_table.add_row("  Read genotypes (pgenlib)", f"{done.get('genotypes_sec', '?')}s")
    timing_table.add_row("  PRS compute (numpy)", f"{done.get('compute_sec', '?')}s")
    timing_table.add_row("  Population join", f"{done.get('join_sec', '?')}s")
    timing_table.add_row("Variants in score", str(done.get("variants_total", "?")))
    timing_table.add_row("Variants matched", str(done.get("variants_matched", "?")))
    console.print(timing_table)


def _validate_pgs_id(pgs_id: str) -> str:
    """Validate and normalize a PGS ID (must match PGS\\d{6,} pattern)."""
    import re
    pgs_id = pgs_id.strip().upper()
    if not re.fullmatch(r"PGS\d{6,}", pgs_id):
        console.print(
            f"[red]Invalid PGS ID: '{pgs_id}'. "
            f"Expected format: PGS followed by 6+ digits (e.g. PGS000001).[/red]"
        )
        raise typer.Exit(code=1)
    return pgs_id


@reference_app.command("score")
def reference_score(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Score a PGS ID against the 1000G reference panel using pgenlib + polars.

    Reads genotypes directly from the .pgen binary via pgenlib, computes dosage
    in numpy, and matches scoring weights in polars.
    """
    from just_prs.reference import compute_reference_prs_polars, reference_panel_dir

    pgs_id = _validate_pgs_id(pgs_id)
    cache = cache_dir or resolve_cache_dir()
    ref_dir = reference_panel_dir(cache)
    if not ref_dir.exists():
        console.print(
            f"[red]Reference panel not found at {ref_dir}.[/red]\n"
            "Download it first with: prs reference download"
        )
        raise typer.Exit(code=1)

    scoring_file = download_scoring_file(pgs_id, cache / "scores", genome_build=build)

    out_dir = cache / "reference_scores" / f"{pgs_id}_polars"
    console.print(
        f"Scoring [cyan]{pgs_id}[/cyan] ({build}) against 1000G using [bold]pgenlib + polars[/bold] engine..."
    )

    from eliot import add_destinations, remove_destination

    timing_msgs: list[dict] = []
    def _capture(msg: dict) -> None:
        mt = msg.get("message_type", "")
        if "polars_phase" in mt or mt == "reference:polars_score_done":
            timing_msgs.append(msg)

    add_destinations(_capture)
    result_df = compute_reference_prs_polars(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=out_dir,
        genome_build=build,
    )
    remove_destination(_capture)

    _print_reference_score_table(pgs_id, result_df)

    scores_parquet = out_dir / "scores.parquet"
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Scores parquet: {scores_parquet}")
    console.print(f"  Output dir:     {out_dir}")

    done = next(
        (m for m in timing_msgs if m.get("message_type") == "reference:polars_score_done"),
        None,
    )
    if done:
        _print_polars_timing_table(done)


@reference_app.command("score-plink2")
def reference_score_plink2(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Score a PGS ID against the 1000G reference panel via PLINK2 --score.

    Requires a PLINK2 binary. For a dependency-free alternative, use 'prs reference score'.
    """
    from just_prs.reference import compute_reference_prs_plink2, reference_panel_dir

    pgs_id = _validate_pgs_id(pgs_id)
    cache = cache_dir or resolve_cache_dir()
    ref_dir = reference_panel_dir(cache)
    if not ref_dir.exists():
        console.print(
            f"[red]Reference panel not found at {ref_dir}.[/red]\n"
            "Download it first with: prs reference download"
        )
        raise typer.Exit(code=1)

    plink2_bin = cache / "plink2" / "plink2"
    if not plink2_bin.exists():
        console.print(
            f"[red]PLINK2 binary not found at {plink2_bin}.[/red]\n"
            "Download it first or set it up via the pipeline."
        )
        raise typer.Exit(code=1)

    scoring_file = download_scoring_file(pgs_id, cache / "scores", genome_build=build)

    out_dir = cache / "reference_scores" / pgs_id
    console.print(f"Scoring [cyan]{pgs_id}[/cyan] ({build}) against 1000G via [bold]PLINK2[/bold]...")

    from eliot import add_destinations, remove_destination

    timing_msgs: list[dict] = []
    def _capture(msg: dict) -> None:
        mt = msg.get("message_type", "")
        if "phase" in mt or mt == "reference:plink2_score_done":
            timing_msgs.append(msg)

    add_destinations(_capture)
    result_df = compute_reference_prs_plink2(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=out_dir,
        plink2_bin=plink2_bin,
        genome_build=build,
    )
    remove_destination(_capture)

    _print_reference_score_table(pgs_id, result_df)

    scores_parquet = out_dir / "scores.parquet"
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Scores parquet: {scores_parquet}")
    console.print(f"  Output dir:     {out_dir}")

    done = next((m for m in timing_msgs if m.get("message_type") == "reference:plink2_score_done"), None)
    if done:
        timing_table = Table(title="Timing & Resources (PLINK2)")
        timing_table.add_column("Metric", style="cyan")
        timing_table.add_column("Value", justify="right", style="green")

        timing_table.add_row("Total time", f"{done.get('total_elapsed_sec', '?')}s")
        timing_table.add_row("  Score input prep", f"{done.get('prepare_sec', '?')}s")
        timing_table.add_row("  PLINK2 execution", f"{done.get('plink2_sec', '?')}s")
        timing_table.add_row("  Result parsing", f"{done.get('parse_sec', '?')}s")
        if "variants_loaded" in done:
            timing_table.add_row("Panel variants", f"{done['variants_loaded']:,}")
        if "variants_matched" in done:
            timing_table.add_row("Variants matched", str(done["variants_matched"]))
        if "variants_skipped" in done:
            timing_table.add_row("Variants skipped", str(done["variants_skipped"]))
        if "ram_available_mb" in done:
            timing_table.add_row("RAM available", f"{done['ram_available_mb']:,} MiB")
        if "ram_reserved_mb" in done:
            timing_table.add_row("RAM reserved", f"{done['ram_reserved_mb']:,} MiB")
        if "threads_used" in done:
            timing_table.add_row("Threads", str(done["threads_used"]))
        console.print(timing_table)


@reference_app.command("test-score")
def reference_test_score(
    pgs_ids: Annotated[
        str,
        typer.Option(
            "--pgs-ids", "-p",
            help="Comma-separated PGS IDs to test (default: PGS000001,PGS000002,PGS000004,PGS000010)",
        ),
    ] = "PGS000001,PGS000002,PGS000004,PGS000010",
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build")
    ] = "GRCh38",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Test reference panel scoring for multiple PGS IDs and report results.

    Uses the polars engine (no PLINK2 needed). Validates output and prints
    a summary table. Non-zero exit code if any score fails.
    """
    from just_prs.reference import (
        SUPERPOPULATIONS,
        compute_reference_prs_polars,
        reference_panel_dir,
    )

    cache = cache_dir or resolve_cache_dir()
    ref_dir = reference_panel_dir(cache)
    if not ref_dir.exists():
        console.print(f"[red]Reference panel not found at {ref_dir}.[/red]")
        raise typer.Exit(code=1)

    ids = [_validate_pgs_id(pid) for pid in pgs_ids.split(",") if pid.strip()]
    console.print(f"Testing {len(ids)} PGS IDs against 1000G reference panel ({build})...\n")

    results_table = Table(title="Reference Score Test Results")
    results_table.add_column("PGS ID", style="cyan")
    results_table.add_column("Samples", justify="right")
    results_table.add_column("Superpops", justify="right")
    results_table.add_column("Mean", justify="right", style="green")
    results_table.add_column("Std", justify="right")
    results_table.add_column("Status", style="bold")

    n_pass = 0
    n_fail = 0

    for pgs_id in ids:
        status = "[green]PASS[/green]"
        scoring_file = download_scoring_file(pgs_id, cache / "scores", genome_build=build)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result_df = compute_reference_prs_polars(
                pgs_id=pgs_id,
                scoring_file=scoring_file,
                ref_dir=ref_dir,
                out_dir=out_dir,
                genome_build=build,
            )

        n_samples = result_df.height
        superpops = set(result_df["superpop"].unique().to_list())
        mean_score = result_df["score"].mean()
        std_score = result_df["score"].std()

        issues: list[str] = []
        if n_samples != 3202:
            issues.append(f"expected 3202 samples, got {n_samples}")
        if superpops != set(SUPERPOPULATIONS):
            issues.append(f"missing superpops: {set(SUPERPOPULATIONS) - superpops}")
        if std_score is None or std_score < 1e-10:
            issues.append("zero variance")

        if issues:
            status = f"[red]FAIL: {'; '.join(issues)}[/red]"
            n_fail += 1
        else:
            n_pass += 1

        results_table.add_row(
            pgs_id,
            str(n_samples),
            str(len(superpops)),
            f"{mean_score:.8f}" if mean_score is not None else "N/A",
            f"{std_score:.8f}" if std_score is not None else "N/A",
            status,
        )

    console.print(results_table)
    console.print(f"\n[bold]{n_pass} passed, {n_fail} failed[/bold]")
    console.print(f"[dim]Note: test-score uses temp dirs. For persistent results use 'prs reference score' or 'prs reference score-batch'.[/dim]")

    if n_fail > 0:
        raise typer.Exit(code=1)


@reference_app.command("score-batch")
def reference_score_batch(
    pgs_ids: Annotated[
        Optional[str],
        typer.Option(
            "--pgs-ids", "-p",
            help="Comma-separated PGS IDs. If omitted, scores all IDs from the PGS Catalog.",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Score only the first N PGS IDs (0 = all)")
    ] = 0,
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    panel: Annotated[
        str, typer.Option("--panel", help="Reference panel identifier (1000g or hgdp_1kg)")
    ] = "1000g",
    skip_existing: Annotated[
        bool, typer.Option("--skip-existing/--no-skip-existing", help="Skip PGS IDs already scored")
    ] = True,
    match_threshold: Annotated[
        float, typer.Option("--match-threshold", help="Flag scores with match rate below this")
    ] = 0.1,
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Score multiple PGS IDs against a reference panel in batch.

    Downloads scoring files and computes PRS for all specified PGS IDs
    using pgenlib + polars. Failures are logged and reported but do not
    abort the batch. Produces reference distributions and a quality report.

    \b
    Examples:
      prs reference score-batch                              # all PGS IDs
      prs reference score-batch --pgs-ids PGS000001,PGS000002
      prs reference score-batch --limit 50
      prs reference score-batch --panel hgdp_1kg
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

    from just_prs.ftp import list_all_pgs_ids
    from just_prs.reference import (
        DEFAULT_PANEL,
        REFERENCE_PANELS,
        compute_reference_prs_batch,
        reference_panel_dir,
    )

    if panel not in REFERENCE_PANELS:
        console.print(f"[red]Unknown panel {panel!r}. Known: {list(REFERENCE_PANELS)}[/red]")
        raise typer.Exit(code=1)

    cache = cache_dir or resolve_cache_dir()
    ref_dir = reference_panel_dir(cache, panel=panel)
    if not ref_dir.exists():
        console.print(
            f"[red]Reference panel {panel!r} not found at {ref_dir}.[/red]\n"
            "Download it first with: prs reference download"
        )
        raise typer.Exit(code=1)

    if pgs_ids:
        ids = [_validate_pgs_id(pid) for pid in pgs_ids.split(",") if pid.strip()]
    else:
        console.print("Fetching PGS ID list from PGS Catalog...")
        ids = list_all_pgs_ids()
        console.print(f"Found {len(ids)} PGS IDs.")

    if limit > 0:
        ids = ids[:limit]

    panel_desc = REFERENCE_PANELS[panel]["description"]
    console.print(
        f"\nScoring [cyan]{len(ids)}[/cyan] PGS IDs against [bold]{panel}[/bold] "
        f"({panel_desc}, {build})...\n"
    )

    result = compute_reference_prs_batch(
        pgs_ids=ids,
        ref_dir=ref_dir,
        cache_dir=cache,
        genome_build=build,
        panel=panel,
        skip_existing=skip_existing,
        match_rate_threshold=match_threshold,
    )

    # Summary table
    summary_table = Table(title="Batch Scoring Summary")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")

    status_counts: dict[str, int] = {}
    for o in result.outcomes:
        status_counts[o.status] = status_counts.get(o.status, 0) + 1

    status_styles = {"ok": "green", "failed": "red", "low_match": "yellow", "zero_variance": "yellow"}
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        style = status_styles.get(status, "dim")
        summary_table.add_row(f"[{style}]{status}[/{style}]", str(count))

    console.print(summary_table)

    # Show failed IDs
    failed = [o for o in result.outcomes if o.status == "failed"]
    if failed:
        console.print(f"\n[red bold]Failed ({len(failed)}):[/red bold]")
        for o in failed[:30]:
            err_short = (o.error or "unknown")[:120]
            console.print(f"  {o.pgs_id}  [dim]{err_short}[/dim]")
        if len(failed) > 30:
            console.print(f"  [dim]... and {len(failed) - 30} more[/dim]")

    # Show problematic IDs
    problematic = [o for o in result.outcomes if o.status in ("low_match", "zero_variance")]
    if problematic:
        console.print(f"\n[yellow bold]Problematic ({len(problematic)}):[/yellow bold]")
        for o in problematic[:20]:
            detail = f"match_rate={o.match_rate}" if o.match_rate is not None else f"std={o.score_std}"
            console.print(f"  {o.pgs_id}  [{o.status}] {detail}")
        if len(problematic) > 20:
            console.print(f"  [dim]... and {len(problematic) - 20} more[/dim]")

    percentiles_dir = cache / "percentiles"
    scores_dir = cache / "reference_scores" / panel
    console.print(f"\n[bold]Output files:[/bold]")
    console.print(f"  Per-PGS scores:  {scores_dir}/{{PGS_ID}}/scores.parquet")
    console.print(f"  Distributions:   {percentiles_dir / f'{panel}_distributions.parquet'}")
    console.print(f"  Quality report:  {percentiles_dir / f'{panel}_quality.parquet'}")

    if result.distributions_df.height > 0:
        console.print(
            f"  {result.distributions_df['pgs_id'].n_unique()} PGS IDs x "
            f"{result.distributions_df['superpopulation'].n_unique()} superpopulations = "
            f"{result.distributions_df.height} distribution rows"
        )


@reference_app.command("download")
def reference_download(
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-download even if already present")
    ] = False,
    panel: Annotated[
        str, typer.Option("--panel", help="Reference panel to download (1000g or hgdp_1kg)")
    ] = "1000g",
) -> None:
    """Download a reference panel from the PGS Catalog FTP."""
    from just_prs.reference import REFERENCE_PANELS, download_reference_panel

    if panel not in REFERENCE_PANELS:
        console.print(f"[red]Unknown panel {panel!r}. Known: {list(REFERENCE_PANELS)}[/red]")
        raise typer.Exit(code=1)

    cache = cache_dir or resolve_cache_dir()
    info = REFERENCE_PANELS[panel]
    console.print(f"Downloading {panel} reference panel ({info['description']})...")
    dest = download_reference_panel(cache_dir=cache, overwrite=overwrite, panel=panel)
    console.print(f"[green]Reference panel ready at {dest}[/green]")


@reference_app.command("compare")
def reference_compare(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Run both PLINK2 and polars scoring engines and compare results side-by-side.

    Scores the same PGS ID against the 1000G reference panel using both engines,
    then reports per-superpopulation statistics, per-sample correlation, and timing.
    """
    from just_prs.reference import (
        compute_reference_prs_plink2,
        compute_reference_prs_polars,
        reference_panel_dir,
    )

    pgs_id = _validate_pgs_id(pgs_id)
    cache = cache_dir or resolve_cache_dir()
    ref_dir = reference_panel_dir(cache)
    if not ref_dir.exists():
        console.print(f"[red]Reference panel not found at {ref_dir}.[/red]")
        raise typer.Exit(code=1)

    plink2_bin = cache / "plink2" / "plink2"
    if not plink2_bin.exists():
        console.print(f"[red]PLINK2 binary not found at {plink2_bin}.[/red]")
        raise typer.Exit(code=1)

    scoring_file = download_scoring_file(pgs_id, cache / "scores", genome_build=build)

    # --- Run PLINK2 engine ---
    console.print(f"\n[bold]Engine 1: PLINK2 --score[/bold]")
    import time as _time

    t0 = _time.monotonic()
    plink2_df = compute_reference_prs_plink2(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=cache / "reference_scores" / pgs_id,
        plink2_bin=plink2_bin,
        genome_build=build,
    )
    t_plink2 = _time.monotonic() - t0
    console.print(f"  Completed in [green]{t_plink2:.2f}s[/green] ({plink2_df.height} samples)")

    # --- Run polars engine ---
    console.print(f"[bold]Engine 2: pgenlib + polars[/bold]")
    t0 = _time.monotonic()
    polars_df = compute_reference_prs_polars(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=cache / "reference_scores" / f"{pgs_id}_polars",
        genome_build=build,
    )
    t_polars = _time.monotonic() - t0
    console.print(f"  Completed in [green]{t_polars:.2f}s[/green] ({polars_df.height} samples)")

    # --- Per-superpopulation comparison ---
    console.print()
    cmp_table = Table(title=f"Per-Superpopulation Comparison: {pgs_id}")
    cmp_table.add_column("Superpop", style="cyan")
    cmp_table.add_column("N", justify="right")
    cmp_table.add_column("PLINK2 Mean", justify="right", style="green")
    cmp_table.add_column("Polars Mean", justify="right", style="blue")
    cmp_table.add_column("PLINK2 Std", justify="right")
    cmp_table.add_column("Polars Std", justify="right")
    cmp_table.add_column("Mean Diff", justify="right", style="yellow")

    for engine_label, df in [("plink2", plink2_df), ("polars", polars_df)]:
        if "superpop" not in df.columns:
            console.print(f"[red]{engine_label} result missing 'superpop' column[/red]")
            raise typer.Exit(code=1)

    plink2_stats = (
        plink2_df.group_by("superpop")
        .agg(
            pl.col("score").count().alias("n"),
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
        )
        .sort("superpop")
    )
    polars_stats = (
        polars_df.group_by("superpop")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
        )
        .sort("superpop")
    )
    merged_stats = plink2_stats.join(
        polars_stats, on="superpop", suffix="_polars"
    )

    for row in merged_stats.iter_rows(named=True):
        diff = abs(row["mean"] - row["mean_polars"]) if row["mean"] and row["mean_polars"] else 0.0
        cmp_table.add_row(
            row["superpop"],
            str(row["n"]),
            f"{row['mean']:.8f}",
            f"{row['mean_polars']:.8f}",
            f"{row['std']:.8f}",
            f"{row['std_polars']:.8f}",
            f"{diff:.2e}",
        )
    console.print(cmp_table)

    # --- Per-sample correlation ---
    merged = plink2_df.select(
        pl.col("iid"), pl.col("score").alias("score_plink2")
    ).join(
        polars_df.select(pl.col("iid"), pl.col("score").alias("score_polars")),
        on="iid",
        how="inner",
    )

    if merged.height > 0:
        corr = merged.select(
            pl.corr("score_plink2", "score_polars").alias("pearson_r")
        ).item()
        max_abs_diff = merged.select(
            (pl.col("score_plink2") - pl.col("score_polars")).abs().max().alias("max_diff")
        ).item()
        mean_abs_diff = merged.select(
            (pl.col("score_plink2") - pl.col("score_polars")).abs().mean().alias("mean_diff")
        ).item()

        summary_table = Table(title="Score Agreement")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Samples matched", str(merged.height))
        summary_table.add_row("Pearson r", f"{corr:.10f}" if corr is not None else "N/A")
        summary_table.add_row("Max |diff|", f"{max_abs_diff:.2e}" if max_abs_diff is not None else "N/A")
        summary_table.add_row("Mean |diff|", f"{mean_abs_diff:.2e}" if mean_abs_diff is not None else "N/A")
        summary_table.add_row("PLINK2 time", f"{t_plink2:.2f}s")
        summary_table.add_row("Polars time", f"{t_polars:.2f}s")
        speedup = t_plink2 / t_polars if t_polars > 0 else float("inf")
        faster = "polars" if t_polars < t_plink2 else "PLINK2"
        summary_table.add_row("Faster engine", f"{faster} ({speedup:.2f}x)" if faster == "polars" else f"{faster} ({1/speedup:.2f}x)")
        console.print(summary_table)
    else:
        console.print("[yellow]No samples could be matched between engines.[/yellow]")


# ---------------------------------------------------------------------------
# pgen subcommand group — standalone pgen/pvar/psam file operations
# ---------------------------------------------------------------------------

@pgen_app.command("read-pvar")
def pgen_read_pvar(
    pvar_path: Annotated[Path, typer.Argument(help="Path to .pvar.zst file")],
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max rows to display")
    ] = 20,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Save full table as parquet")
    ] = None,
) -> None:
    """Parse a .pvar.zst variant file and display its contents.

    Reads the binary .pvar.zst directly using zstandard, caches the result
    as a parquet file for fast subsequent reads.
    """
    from just_prs.reference import parse_pvar

    console.print(f"Parsing [cyan]{pvar_path}[/cyan] ...")
    df = parse_pvar(pvar_path)
    console.print(f"[green]Loaded {df.height:,} variants[/green]\n")

    table = Table(title=f"Variant table ({min(limit, df.height)} of {df.height:,} rows)")
    for col in df.columns:
        table.add_column(col, style="cyan" if col == "chrom" else "")
    for row in df.head(limit).iter_rows():
        table.add_row(*[str(v) for v in row])
    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        console.print(f"[green]Saved {df.height:,} rows to {output}[/green]")


@pgen_app.command("read-psam")
def pgen_read_psam(
    psam_path: Annotated[Path, typer.Argument(help="Path to .psam file")],
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max rows to display")
    ] = 20,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Save full table as parquet")
    ] = None,
) -> None:
    """Parse a .psam sample file and display its contents.

    Reads sample IDs and population labels from a PLINK2 binary fileset.
    """
    from just_prs.reference import parse_psam

    console.print(f"Parsing [cyan]{psam_path}[/cyan] ...")
    df = parse_psam(psam_path)
    console.print(f"[green]Loaded {df.height:,} samples[/green]")

    superpop_counts = df.group_by("superpop").len().sort("superpop")
    summary = Table(title="Superpopulation summary")
    summary.add_column("Superpop", style="cyan")
    summary.add_column("N", justify="right", style="green")
    for row in superpop_counts.iter_rows(named=True):
        summary.add_row(row["superpop"], str(row["len"]))
    console.print(summary)

    detail = Table(title=f"Samples ({min(limit, df.height)} of {df.height:,} rows)")
    for col in df.columns:
        detail.add_column(col)
    for row in df.head(limit).iter_rows():
        detail.add_row(*[str(v) for v in row])
    console.print(detail)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        console.print(f"[green]Saved {df.height:,} rows to {output}[/green]")


@pgen_app.command("genotypes")
def pgen_genotypes(
    pgen_path: Annotated[Path, typer.Argument(help="Path to .pgen file")],
    pvar_path: Annotated[Path, typer.Argument(help="Path to .pvar.zst file")],
    psam_path: Annotated[Path, typer.Argument(help="Path to .psam file")],
    chrom: Annotated[
        Optional[str], typer.Option("--chrom", "-c", help="Filter to this chromosome")
    ] = None,
    pos_start: Annotated[
        Optional[int], typer.Option("--start", help="Start position (inclusive)")
    ] = None,
    pos_end: Annotated[
        Optional[int], typer.Option("--end", help="End position (inclusive)")
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Max variants to extract")
    ] = 100,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Save genotypes as parquet")
    ] = None,
) -> None:
    """Extract genotypes from a .pgen file for specific regions.

    Reads genotype data directly via pgenlib.
    Output values: 0 = hom-ref, 1 = het, 2 = hom-alt, -9 = missing.
    """
    import numpy as np

    from just_prs.reference import parse_psam, parse_pvar, read_pgen_genotypes

    console.print(f"Loading variant table from [cyan]{pvar_path}[/cyan] ...")
    pvar_df = parse_pvar(pvar_path)

    filtered = pvar_df
    if chrom is not None:
        filtered = filtered.filter(pl.col("chrom") == chrom.replace("chr", ""))
    if pos_start is not None:
        filtered = filtered.filter(pl.col("POS") >= pos_start)
    if pos_end is not None:
        filtered = filtered.filter(pl.col("POS") <= pos_end)

    if filtered.height == 0:
        console.print("[yellow]No variants match the specified region.[/yellow]")
        raise typer.Exit(code=0)

    filtered = filtered.head(limit)
    console.print(f"Extracting genotypes for {filtered.height} variants ...")

    psam_df = parse_psam(psam_path)
    variant_indices = filtered["variant_idx"].cast(pl.UInt32).to_numpy()

    geno_matrix = read_pgen_genotypes(
        pgen_path=pgen_path,
        pvar_zst_path=pvar_path,
        variant_indices=variant_indices,
        n_samples=psam_df.height,
    )

    n_het = int(np.sum(geno_matrix == 1))
    n_hom_alt = int(np.sum(geno_matrix == 2))
    n_missing = int(np.sum(geno_matrix == -9))
    n_total = int(geno_matrix.size)

    summary = Table(title="Genotype summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")
    summary.add_row("Variants", str(geno_matrix.shape[0]))
    summary.add_row("Samples", str(geno_matrix.shape[1]))
    summary.add_row("Het (1)", f"{n_het:,} ({n_het/n_total:.1%})")
    summary.add_row("Hom-alt (2)", f"{n_hom_alt:,} ({n_hom_alt/n_total:.1%})")
    summary.add_row("Missing (-9)", f"{n_missing:,} ({n_missing/n_total:.1%})")
    console.print(summary)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        iids = psam_df["iid"].to_list()
        rows: list[dict[str, object]] = []
        for i, var_row in enumerate(filtered.iter_rows(named=True)):
            row_data: dict[str, object] = {
                "chrom": var_row["chrom"],
                "pos": var_row["POS"],
                "ref": var_row["REF"],
                "alt": var_row["ALT"],
            }
            for j, iid in enumerate(iids):
                row_data[iid] = int(geno_matrix[i, j])
            rows.append(row_data)
        out_df = pl.DataFrame(rows)
        out_df.write_parquet(output)
        console.print(f"[green]Saved {out_df.height} variants x {psam_df.height} samples to {output}[/green]")


@pgen_app.command("score")
def pgen_score(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    pgen_dir: Annotated[Path, typer.Argument(help="Directory containing .pgen/.pvar.zst/.psam files")],
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)")
    ] = "GRCh38",
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Save scores as parquet")
    ] = None,
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory for scoring file download")
    ] = None,
) -> None:
    """Score a PGS ID against any .pgen/.pvar.zst/.psam dataset.

    Unlike 'prs reference score' which targets the 1000G panel specifically,
    this command works with any PLINK2 binary fileset.
    """
    from just_prs.reference import compute_reference_prs_polars

    pgs_id = _validate_pgs_id(pgs_id)
    cache = cache_dir or resolve_cache_dir()

    scoring_file = download_scoring_file(pgs_id, cache / "scores", genome_build=build)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        console.print(
            f"Scoring [cyan]{pgs_id}[/cyan] ({build}) against [cyan]{pgen_dir}[/cyan] ..."
        )

        from eliot import add_destinations, remove_destination

        timing_msgs: list[dict] = []
        def _capture(msg: dict) -> None:
            mt = msg.get("message_type", "")
            if "polars_phase" in mt or mt == "reference:polars_score_done":
                timing_msgs.append(msg)

        add_destinations(_capture)
        result_df = compute_reference_prs_polars(
            pgs_id=pgs_id,
            scoring_file=scoring_file,
            ref_dir=pgen_dir,
            out_dir=out_dir,
            genome_build=build,
        )
        remove_destination(_capture)

    _print_reference_score_table(pgs_id, result_df)

    done = next(
        (m for m in timing_msgs if m.get("message_type") == "reference:polars_score_done"),
        None,
    )
    if done:
        _print_polars_timing_table(done)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output)
        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Scores parquet: {output}")
    else:
        console.print(f"\n[dim]Tip: use --output/-o to save results as parquet.[/dim]")


def run() -> None:
    """Entry point for the CLI."""
    app()
