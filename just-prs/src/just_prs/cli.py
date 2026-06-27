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

ancestry_app = typer.Typer(
    name="ancestry",
    help="Infer a sample's genetic ancestry (super-population) and check score×sample×panel coherence.",
    no_args_is_help=True,
)
app.add_typer(ancestry_app, name="ancestry")

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
            try:
                pl.scan_parquet(parquet_path).collect_schema()
                already_cached += 1
                if delete_gz:
                    gz_path.unlink()
                    deleted += 1
            except Exception as exc:
                console.print(f"  [yellow]{parquet_path.name}: corrupt cache removed ({exc})[/yellow]")
                parquet_path.unlink(missing_ok=True)
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
                except Exception as parse_exc:
                    failed += 1
                    console.print(f"  [red]{gz_path.name}: {parse_exc}[/red]")
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
    ] = "just-dna-seq/pgs-catalog",
) -> None:
    """Pull cleaned metadata parquets from a HuggingFace dataset repository.

    Downloads scores.parquet, performance.parquet, and best_performance.parquet
    from the data/metadata/ folder of the HF repo into the output directory.
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
    vcf: Annotated[str, typer.Option("--vcf", "-v", help="Path to VCF file or alias name (e.g. 'anton', 'livia')")],
    pgs_id: Annotated[
        Optional[str], typer.Option("--pgs-id", "-p", help="PGS ID(s), comma-separated")
    ] = None,
    scoring_file: Annotated[
        Optional[Path],
        typer.Option(
            "--scoring-file", "-s",
            help="Local scoring file (.txt.gz or .parquet) instead of a PGS Catalog ID",
        ),
    ] = None,
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build")
    ] = "GRCh38",
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Cache directory for scoring files")
    ] = DEFAULT_CACHE_DIR,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output JSON file")
    ] = None,
    genotype_input_mode: Annotated[
        str,
        typer.Option(
            "--genotype-input-mode",
            help=(
                "How absent scoring loci are interpreted: auto, variant_only, "
                "all_sites, or plink_present_only"
            ),
        ),
    ] = "auto",
    reference_restoration: Annotated[
        str,
        typer.Option(
            "--reference-restoration",
            help=(
                "Restore a scoring variant's missing reference allele from the "
                "precomputed reference-allele universe so absent loci score as "
                "homozygous-reference within a scope: 'off' (default; old behavior), "
                "'wgs' (whole universe — for genome-wide variant-only WGS), or a chip "
                "id (e.g. 'gsa_v3' — only chip-typed positions, for arrays)."
            ),
        ),
    ] = "off",
) -> None:
    """Compute polygenic risk score(s) for a VCF file.

    Use --pgs-id to score PGS Catalog models, or --scoring-file to score a
    custom local scoring file (.txt.gz or .parquet with effect_allele,
    effect_weight, and position columns).
    """
    from just_prs.chip_coverage import Chip
    from just_prs.prs import RestorationScope

    vcf_path = _resolve_vcf(vcf, cache_dir)

    if scoring_file and pgs_id:
        console.print("[red]Provide either --pgs-id or --scoring-file, not both.[/red]")
        raise typer.Exit(code=1)
    if not scoring_file and not pgs_id:
        console.print("[red]Provide --pgs-id or --scoring-file.[/red]")
        raise typer.Exit(code=1)

    # Parse the restoration scope: off -> False, wgs -> True, else a Chip.
    choice = reference_restoration.strip().lower()
    scope: RestorationScope
    if choice in ("off", "false", "none"):
        scope = False
    elif choice in ("wgs", "true", "universe"):
        scope = True
    else:
        try:
            scope = Chip(choice)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--reference-restoration must be 'off', 'wgs', or a chip id "
                f"({', '.join(c.value for c in Chip)}); got {reference_restoration!r}"
            ) from exc

    # Resolve (and lazily pull) the reference-allele universe once when on.
    # Build-aware: GRCh37 restoration must use the GRCh37 universe, not GRCh38.
    universe_path: Optional[Path] = None
    if scope is not False:
        from just_prs.hf import (
            pull_reference_allele_universe,
            reference_allele_universe_filename,
        )
        from just_prs.scoring import resolve_cache_dir

        ref_dir = resolve_cache_dir() / "reference"
        candidate = ref_dir / reference_allele_universe_filename(build)
        if not candidate.exists():
            try:
                pull_reference_allele_universe(ref_dir, genome_build=build)
            except Exception as exc:
                console.print(
                    f"[yellow]Could not fetch reference-allele universe ({exc}); "
                    f"proceeding without restoration.[/yellow]"
                )
        universe_path = candidate if candidate.exists() else None
        if universe_path is None:
            console.print(
                "[yellow]Reference-allele universe unavailable; "
                "--reference-restoration is a no-op this run.[/yellow]"
            )
            scope = False

    # Detect the sample's build from the VCF header so a build mismatch fails
    # loudly instead of silently scoring ~0 variants. Unknown build -> no guard
    # (never guesses; e.g. a normalized parquet has no header).
    sample_build: Optional[str] = None
    try:
        from just_prs.vcf import detect_genome_build

        sample_build = detect_genome_build(vcf_path)
    except Exception:
        sample_build = None

    if scoring_file:
        label = scoring_file.stem
        console.print(f"Computing PRS from custom scoring file [cyan]{scoring_file}[/cyan] on {vcf_path}...")
        result = compute_prs(
            vcf_path=vcf_path,
            scoring_file=scoring_file,
            genome_build=build,
            cache_dir=cache_dir,
            pgs_id=label,
            genotype_input_mode=genotype_input_mode,
            reference_restoration=scope,
            reference_universe_path=universe_path,
            sample_build=sample_build,
        )
        results: list[PRSResult] = [result]
    else:
        pgs_ids = [pid.strip() for pid in pgs_id.split(",")]
        console.print(f"Computing PRS for {len(pgs_ids)} score(s) on {vcf_path}...")

        if len(pgs_ids) == 1:
            with PGSCatalogClient() as client:
                score_info = client.get_score(pgs_ids[0])
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_ids[0],
                genome_build=build,
                cache_dir=cache_dir,
                pgs_id=pgs_ids[0],
                trait_reported=score_info.trait_reported,
                genotype_input_mode=genotype_input_mode,
                reference_restoration=scope,
                reference_universe_path=universe_path,
                sample_build=sample_build,
            )
            results = [result]
        else:
            batch = compute_prs_batch(
                vcf_path=vcf_path,
                pgs_ids=pgs_ids,
                genome_build=build,
                cache_dir=cache_dir,
                genotype_input_mode=genotype_input_mode,
                reference_restoration=scope,
                reference_universe_path=universe_path,
                sample_build=sample_build,
            )
            results = batch.results
            if batch.failed_ids:
                console.print(
                    f"[yellow]Warning: {batch.n_failed}/{batch.n_total} scores failed: "
                    f"{', '.join(batch.failed_ids)}[/yellow]"
                )

    table = Table(title="PRS Results")
    table.add_column("PGS ID", style="cyan")
    table.add_column("Trait", style="magenta")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Matched", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Match Rate", justify="right")
    table.add_column("Assumed Ref", justify="right")
    table.add_column("Unavailable", justify="right")
    table.add_column("Mode", justify="right")

    for r in results:
        table.add_row(
            r.pgs_id,
            r.trait_reported or "",
            f"{r.score:.6f}",
            str(r.variants_matched),
            str(r.variants_total),
            f"{r.match_rate:.1%}",
            str(r.variants_assumed_hom_ref),
            str(r.variants_unscorable_absent),
            r.genotype_input_mode,
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


@app.command("normalize-array")
def normalize_array_cmd(
    array: Annotated[
        Path,
        typer.Option("--array", "-a", help="Path to a 23andMe/AncestryDNA raw file (.txt/.txt.gz/.csv/.zip)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output Parquet path. Defaults to data/output/results/<stem>.parquet"),
    ] = None,
    build: Annotated[
        str,
        typer.Option("--build", "-b", help="Genome build of the array coordinates (GRCh37 for 23andMe v5 / AncestryDNA v2)"),
    ] = "GRCh37",
    array_format: Annotated[
        Optional[str],
        typer.Option("--format", help='Force vendor format ("23andme" or "ancestrydna"); auto-detected when omitted'),
    ] = None,
) -> None:
    """Normalize a consumer genotyping-array file (23andMe/AncestryDNA) to Parquet for PRS computation."""
    from just_prs.arrays import normalize_array

    if output is None:
        stem = array.name
        for suffix in (".zip", ".gz", ".csv", ".txt"):
            stem = stem.removesuffix(suffix)
        output = Path("data/output/results") / f"{stem}.parquet"

    console.print(f"Normalizing array [cyan]{array}[/cyan] (build {build}) → {output} ...")
    result_path = normalize_array(array, output, genome_build=build, array_format=array_format)

    df = pl.read_parquet(result_path)
    console.print(f"[green]Wrote {df.height:,} typed variants to {result_path}[/green]")
    console.print(
        "Compute a PRS with: "
        f"[cyan]prs compute --vcf {result_path} --pgs-id PGS000001 --build {build}[/cyan]"
    )


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


def _fmt_dist(d: dict) -> str:
    return ", ".join(f"{k} {v:.0%}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v > 0.005)


def _print_single(res, label: str) -> None:
    console.print(f"[bold]{label}:[/bold] {res.superpopulation}  "
                  f"(conf {res.confidence:.2f}, cov {res.coverage:.1%}, "
                  f"{res.n_variants_used:,}/{res.n_variants_model:,} sites, {res.panel}/{res.genome_build})")
    if res.probabilities:
        console.print(f"    KNN (fraposa):  {_fmt_dist(res.probabilities)}")
    if res.mixture:
        console.print(f"    mixture (PCA-NNLS):  {_fmt_dist(res.mixture)}")


@ancestry_app.command("infer")
def ancestry_infer(
    vcf: Annotated[str, typer.Option("--vcf", "-v", help="VCF/array path or alias (e.g. 'anton', 'livia')")],
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="label | mixture | prive (21-group proportions) | consensus | all"),
    ] = "all",
    panel: Annotated[str, typer.Option("--panel", help="Reference panel for label/mixture modes: 1000g or hgdp_1kg")] = "1000g",
    panels: Annotated[
        str, typer.Option("--panels", help="Comma-separated panels fused in consensus/all modes")
    ] = "1000g,hgdp_1kg",
    prive: Annotated[
        bool, typer.Option("--prive/--no-prive", help="Fold the Privé 21-group reference into consensus/all (must be built locally)")
    ] = False,
    build: Annotated[
        Optional[str], typer.Option("--build", "-b", help="Sample genome build (auto-detected; default GRCh38)")
    ] = None,
    cache_dir: Annotated[Optional[Path], typer.Option("--cache-dir", help="Cache directory (base, default ~/.cache/just-prs)")] = None,
) -> None:
    """Infer a sample's genetic ancestry. Modes: label, mixture, prive, consensus, all (default).

    \b
    - label:     single-panel KNN (fraposa) hard super-population call + posterior.
    - mixture:   single-panel PCA-NNLS ancestry proportions.
    - prive:     Privé/bigsnpr 21-group proportions (finer within-continent; GRCh37 ref, lifted).
    - consensus: Bayesian product-of-experts fusing every panel's KNN + mixture (+ Privé with --prive).
    - all:       per-panel label + mixture, the Privé breakdown (with --prive), then the consensus.
    """
    vcf_path = _resolve_vcf(vcf, cache_dir)
    catalog = PRSCatalog(cache_dir=cache_dir)
    panel_list = [p.strip() for p in panels.split(",") if p.strip()]

    if mode in ("label", "mixture"):
        res = catalog.infer_ancestry(vcf_path, panel=panel, sample_build=build)
        if mode == "label":
            console.print(f"[bold]Ancestry ({panel}):[/bold] {res.superpopulation}  "
                          f"(conf {res.confidence:.2f}, cov {res.coverage:.1%})")
            if res.probabilities:
                console.print(f"    {_fmt_dist(res.probabilities)}")
        else:
            console.print(f"[bold]Mixture ({panel}):[/bold] {_fmt_dist(res.mixture or {})}")
        return

    if mode == "prive":
        pr = catalog.infer_ancestry_prive(vcf_path, sample_build=build)
        if pr is None:
            console.print("[yellow]Privé reference not built locally (see just_prs.ancestry.prive.build_prive_reference).[/yellow]")
            raise typer.Exit(code=1)
        console.print(f"[bold]Privé continental:[/bold] {_fmt_dist(pr['continental'])}  "
                      f"({pr['n_variants_used']:,} variants)")
        console.print(f"    21-group: {_fmt_dist(pr['proportions'])}")
        return

    # consensus / all: fuse across panels (+ Privé with --prive)
    con = catalog.infer_ancestry_consensus(
        vcf_path, panels=tuple(panel_list), sample_build=build, include_prive=prive
    )
    if mode == "all":
        for p in panel_list:
            r = con.per_panel.get(p)
            if r is not None:
                _print_single(r, f"Panel {p}")
        prive_m = next((m for m in con.methods if m["panel"] == "prive"), None)
        if prive_m is not None:
            console.print(f"[bold]Privé (21-group → continental):[/bold] {_fmt_dist(prive_m['distribution'])}")
    color = "green" if con.confidence >= 0.8 else "yellow"
    console.print(f"[bold {color}]Consensus:[/bold {color}] {con.consensus_superpopulation}  "
                  f"(posterior {con.confidence:.2f}, fused {len(con.methods)} methods across {len(con.per_panel)} panels)")
    console.print(f"    {_fmt_dist(con.posterior)}")


@ancestry_app.command("check")
def ancestry_check(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    vcf: Annotated[str, typer.Option("--vcf", "-v", help="VCF/array path or alias")],
    panel: Annotated[str, typer.Option("--panel", help="Reference panel: 1000g or hgdp_1kg")] = "1000g",
    build: Annotated[Optional[str], typer.Option("--build", "-b", help="Sample genome build")] = None,
    cache_dir: Annotated[Optional[Path], typer.Option("--cache-dir", help="Cache directory (base, default ~/.cache/just-prs)")] = None,
) -> None:
    """Check score×sample×panel ancestry coherence and print a plain-English verdict."""
    vcf_path = _resolve_vcf(vcf, cache_dir)
    catalog = PRSCatalog(cache_dir=cache_dir)
    anc = catalog.infer_ancestry(vcf_path, panel=panel, sample_build=build)
    verdict = catalog.assess_ancestry_coherence(pgs_id, anc)
    color = "green" if verdict.reliable else "yellow"
    console.print(f"[bold]Sample ancestry:[/bold] {anc.superpopulation}  "
                  f"[bold]score dev:[/bold] {verdict.dev_ancestry or 'n/a'}  "
                  f"[bold]panel:[/bold] {verdict.panel_ancestry or 'n/a'}")
    console.print(f"[{color}]Coherence: {verdict.level}[/{color}] — {verdict.message}")


# ---------------------------------------------------------------------------
# VCF aliases — named shortcuts for frequently used genotype files
# ---------------------------------------------------------------------------

BUILTIN_ALIASES: dict[str, str] = {
    "anton": "~/.cache/just-prs/genomes/antonkulaga.vcf",
    "livia": "~/.cache/just-prs/genomes/SIMHIFQTILQ.hard-filtered.vcf.gz",
}

BUILTIN_ZENODO_URLS: dict[str, tuple[str, str]] = {
    "anton": (
        "https://zenodo.org/api/records/18370498/files/antonkulaga.vcf/content",
        "antonkulaga.vcf",
    ),
    "livia": (
        "https://zenodo.org/api/records/19487816/files/SIMHIFQTILQ.hard-filtered.vcf.gz/content",
        "SIMHIFQTILQ.hard-filtered.vcf.gz",
    ),
}


def _aliases_path(cache_dir: Path | None = None) -> Path:
    cache = cache_dir or resolve_cache_dir()
    return cache / "vcf_aliases.json"


def _load_aliases(cache_dir: Path | None = None) -> dict[str, str]:
    path = _aliases_path(cache_dir)
    user_aliases: dict[str, str] = {}
    if path.exists():
        user_aliases = json.loads(path.read_text())
    merged = {**BUILTIN_ALIASES, **user_aliases}
    return merged


def _save_user_aliases(aliases: dict[str, str], cache_dir: Path | None = None) -> None:
    path = _aliases_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aliases, indent=2))


def _download_builtin_vcf(alias: str, dest: Path) -> None:
    """Download a built-in alias VCF from Zenodo."""
    import httpx

    url, _filename = BUILTIN_ZENODO_URLS[alias]
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Downloading [cyan]{alias}[/cyan] genome from Zenodo...")
    console.print(f"  URL: {url}")
    console.print(f"  Destination: {dest}")

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_pct_reported = -10
            with open(tmp_dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        if pct >= last_pct_reported + 10:
                            console.print(f"  {downloaded // (1024*1024)} / {total // (1024*1024)} MB ({pct}%)")
                            last_pct_reported = pct
        if total and downloaded < total:
            tmp_dest.unlink(missing_ok=True)
            console.print(f"[red]Download incomplete: got {downloaded} of {total} bytes.[/red]")
            raise typer.Exit(code=1)
        tmp_dest.rename(dest)
    except Exception:
        tmp_dest.unlink(missing_ok=True)
        raise
    console.print(f"[green]Downloaded {dest.name} ({downloaded // (1024*1024)} MB)[/green]")


def _resolve_vcf(vcf_or_alias: str, cache_dir: Path | None = None) -> Path:
    """Resolve a VCF path or alias name to an existing file Path.

    For built-in aliases (anton, livia), auto-downloads from Zenodo if not cached.
    """
    p = Path(vcf_or_alias).expanduser()
    if p.exists():
        return p

    aliases = _load_aliases(cache_dir)
    lower = vcf_or_alias.lower().strip()
    if lower in aliases:
        resolved = Path(aliases[lower]).expanduser()
        if not resolved.exists() and lower in BUILTIN_ZENODO_URLS:
            _download_builtin_vcf(lower, resolved)
        if not resolved.exists():
            console.print(
                f"[red]Alias '{vcf_or_alias}' points to {resolved} but the file does not exist.[/red]\n"
                f"Set it with: prs alias set {lower} /path/to/file.vcf.gz"
            )
            raise typer.Exit(code=1)
        return resolved

    console.print(
        f"[red]'{vcf_or_alias}' is neither an existing file nor a known alias.[/red]\n"
        f"Known aliases: {', '.join(aliases) or '(none)'}\n"
        f"Add one with: prs alias set {lower} /path/to/file.vcf.gz"
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# PRS result cache — keyed by (vcf mtime+size, pgs_id, genome_build, ancestry)
# ---------------------------------------------------------------------------

def _results_cache_dir(cache_dir: Path | None = None) -> Path:
    cache = cache_dir or resolve_cache_dir()
    d = cache / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vcf_fingerprint(vcf_path: Path) -> str:
    """Cheap fingerprint: file size + mtime (avoids hashing multi-GB VCFs)."""
    stat = vcf_path.stat()
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def _result_cache_key(vcf_path: Path, pgs_id: str, build: str, ancestry: str) -> str:
    fp = _vcf_fingerprint(vcf_path)
    return f"{pgs_id}_{build}_{ancestry}_{fp}"


def _load_result_cache(cache_dir: Path | None = None) -> dict:
    d = _results_cache_dir(cache_dir)
    p = d / "prs_results_cache.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_result_cache(cache: dict, cache_dir: Path | None = None) -> None:
    d = _results_cache_dir(cache_dir)
    p = d / "prs_results_cache.json"
    p.write_text(json.dumps(cache, indent=2))


def _get_cached_result(
    vcf_path: Path, pgs_id: str, build: str, ancestry: str,
    cache_dir: Path | None = None,
) -> dict | None:
    cache = _load_result_cache(cache_dir)
    key = _result_cache_key(vcf_path, pgs_id, build, ancestry)
    hit = cache.get(key)
    if hit is not None:
        z = hit.get("z_score")
        if z is not None and abs(z) > 10:
            hit["z_score"] = None
            hit["percentile"] = None
            hit.pop("risk_ratio", None)
            hit.pop("absolute_risk", None)
            hit.pop("population_prevalence", None)
            hit.pop("risk_method", None)
            cache[key] = hit
            _save_result_cache(cache, cache_dir)
    return hit


def _put_cached_result(
    vcf_path: Path, pgs_id: str, build: str, ancestry: str,
    result: dict, cache_dir: Path | None = None,
) -> None:
    cache = _load_result_cache(cache_dir)
    key = _result_cache_key(vcf_path, pgs_id, build, ancestry)
    cache[key] = result
    _save_result_cache(cache, cache_dir)


alias_app = typer.Typer(
    name="alias",
    help="Manage VCF aliases — named shortcuts for frequently used genotype files.",
    no_args_is_help=True,
)
app.add_typer(alias_app, name="alias")


@alias_app.command("list")
def alias_list(
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """List all VCF aliases (built-in and user-defined)."""
    aliases = _load_aliases(cache_dir)
    user_path = _aliases_path(cache_dir)
    user_aliases: dict[str, str] = {}
    if user_path.exists():
        user_aliases = json.loads(user_path.read_text())

    table = Table(title="VCF Aliases")
    table.add_column("Alias", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Exists", justify="center")
    table.add_column("Source", style="dim")

    for name, path_str in sorted(aliases.items()):
        resolved = Path(path_str).expanduser()
        if resolved.exists():
            exists = "[green]yes[/green]"
        elif name in BUILTIN_ZENODO_URLS:
            exists = "[yellow]auto-download[/yellow]"
        else:
            exists = "[red]no[/red]"
        source = "user" if name in user_aliases else "built-in"
        table.add_row(name, path_str, exists, source)

    console.print(table)
    not_cached = [n for n in BUILTIN_ZENODO_URLS if not Path(aliases.get(n, "")).expanduser().exists()]
    if not_cached:
        console.print(f"\n[dim]Built-in aliases ({', '.join(not_cached)}) will auto-download from Zenodo on first use.[/dim]")
    console.print(f"[dim]User aliases file: {user_path}[/dim]")


@alias_app.command("set")
def alias_set(
    name: Annotated[str, typer.Argument(help="Alias name (e.g. 'livia')")],
    vcf_path: Annotated[Path, typer.Argument(help="Path to VCF file")],
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Set or update a VCF alias."""
    if not vcf_path.exists():
        console.print(f"[yellow]Warning: {vcf_path} does not exist yet.[/yellow]")

    user_path = _aliases_path(cache_dir)
    user_aliases: dict[str, str] = {}
    if user_path.exists():
        user_aliases = json.loads(user_path.read_text())

    user_aliases[name.lower().strip()] = str(vcf_path.expanduser().resolve())
    _save_user_aliases(user_aliases, cache_dir)
    console.print(f"[green]Alias '{name}' → {vcf_path}[/green]")


@alias_app.command("remove")
def alias_remove(
    name: Annotated[str, typer.Argument(help="Alias name to remove")],
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Remove a user-defined VCF alias."""
    user_path = _aliases_path(cache_dir)
    user_aliases: dict[str, str] = {}
    if user_path.exists():
        user_aliases = json.loads(user_path.read_text())

    key = name.lower().strip()
    if key in BUILTIN_ALIASES and key not in user_aliases:
        console.print(f"[yellow]'{name}' is a built-in alias and cannot be removed (but you can override it with 'prs alias set').[/yellow]")
        raise typer.Exit(code=1)

    if key not in user_aliases:
        console.print(f"[yellow]Alias '{name}' not found in user aliases.[/yellow]")
        raise typer.Exit(code=1)

    del user_aliases[key]
    _save_user_aliases(user_aliases, cache_dir)
    console.print(f"[green]Removed alias '{name}'.[/green]")


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
    console.print(f"  Issue report:    {percentiles_dir / f'{panel}_distribution_quality_issues.parquet'}")

    if result.distributions_df.height > 0:
        console.print(
            f"  {result.distributions_df['pgs_id'].n_unique()} PGS IDs x "
            f"{result.distributions_df['superpopulation'].n_unique()} superpopulations = "
            f"{result.distributions_df.height} distribution rows"
        )
    if result.distribution_issues_df.height > 0:
        n_errors = result.distribution_issues_df.filter(pl.col("severity") == "ERROR").height
        n_warnings = result.distribution_issues_df.filter(pl.col("severity") == "WARN").height
        console.print(
            f"  [yellow]{result.distribution_issues_df.height} distribution issues "
            f"(errors={n_errors}, warnings={n_warnings})[/yellow]"
        )


@reference_app.command("backfill-quality")
def reference_backfill_quality(
    pgs_ids: Annotated[
        Optional[str],
        typer.Option("--pgs-ids", help="Comma-separated PGS IDs to backfill. Defaults to all cached IDs."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Limit number of IDs for testing (0 = no limit)."),
    ] = 0,
    build: Annotated[
        str,
        typer.Option("--build", "-b", help="Genome build (GRCh37 or GRCh38)."),
    ] = "GRCh38",
    panel: Annotated[
        str,
        typer.Option("--panel", help="Reference panel (1000g or hgdp_1kg)."),
    ] = "1000g",
    match_threshold: Annotated[
        float,
        typer.Option("--match-threshold", help="Flag scores with match rate below this."),
    ] = 0.1,
    no_rewrite_scores: Annotated[
        bool,
        typer.Option("--no-rewrite-scores", help="Do not add match metadata columns to cached per-PGS scores.parquet files."),
    ] = False,
    output_subdir: Annotated[
        Optional[str],
        typer.Option("--output-subdir", help="Write percentiles under cache/percentiles/<subdir>; required for partial test runs."),
    ] = None,
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory."),
    ] = None,
) -> None:
    """Backfill reference quality metadata from cached scores without pgen scoring.

    This repairs `{panel}_quality.parquet` after older runs that produced
    distributions but left `variants_total`, `variants_matched`, and
    `match_rate` null. It reuses cached per-PGS scores and only recomputes
    scoring-file-to-pvar match counts.
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from just_prs.reference import (
        DEFAULT_PANEL,
        REFERENCE_PANELS,
        backfill_reference_quality_metadata,
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

    scores_dir = cache / "reference_scores" / panel
    if pgs_ids:
        ids = [_validate_pgs_id(pid) for pid in pgs_ids.split(",") if pid.strip()]
    else:
        ids = sorted(p.name for p in scores_dir.iterdir() if (p / "scores.parquet").exists()) if scores_dir.exists() else []
    if limit > 0:
        ids = ids[:limit]
    if not ids:
        console.print(f"[red]No cached per-PGS scores found under {scores_dir}.[/red]")
        raise typer.Exit(code=1)
    if (pgs_ids or limit > 0) and not output_subdir:
        console.print(
            "[red]Partial backfills would overwrite the main percentile files.[/red]\n"
            "Pass --output-subdir test for trial runs, or omit --pgs-ids/--limit to repair all cached IDs."
        )
        raise typer.Exit(code=1)

    panel_desc = REFERENCE_PANELS.get(panel, REFERENCE_PANELS[DEFAULT_PANEL])["description"]
    console.print(
        f"\nBackfilling quality metadata for [cyan]{len(ids)}[/cyan] cached PGS IDs "
        f"against [bold]{panel}[/bold] ({panel_desc}, {build}).\n"
        "[dim]This does not read pgen genotypes or recompute PRS scores.[/dim]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Resolving panel...", total=len(ids))

        def _progress(update: dict[str, object]) -> None:
            processed = int(update.get("processed", 0) or 0)
            last_pgs_id = str(update.get("last_pgs_id", ""))
            last_status = str(update.get("last_status", ""))
            progress.update(
                task_id,
                completed=processed,
                description=f"{last_pgs_id} {last_status}".strip(),
            )

        result = backfill_reference_quality_metadata(
            pgs_ids=ids,
            ref_dir=ref_dir,
            cache_dir=cache,
            genome_build=build,
            panel=panel,
            match_rate_threshold=match_threshold,
            output_subdir=output_subdir,
            rewrite_score_parquets=not no_rewrite_scores,
            progress_callback=_progress,
        )

    percentiles_dir = cache / "percentiles"
    if output_subdir:
        percentiles_dir = percentiles_dir / output_subdir
    n_missing = result.quality_df.select(
        (
            pl.col("variants_total").is_null()
            | pl.col("variants_matched").is_null()
            | pl.col("match_rate").is_null()
        ).sum().alias("n_missing")
    )["n_missing"][0]
    n_errors = result.distribution_issues_df.filter(pl.col("severity") == "ERROR").height
    n_warnings = result.distribution_issues_df.filter(pl.col("severity") == "WARN").height

    console.print("\n[bold green]Backfill complete.[/bold green]")
    console.print(f"  Quality report:  {percentiles_dir / f'{panel}_quality.parquet'}")
    console.print(f"  Issue report:    {percentiles_dir / f'{panel}_distribution_quality_issues.parquet'}")
    console.print(f"  Distributions:   {percentiles_dir / f'{panel}_distributions.parquet'}")
    console.print(f"  Missing match metadata rows: {n_missing}")
    console.print(f"  Audit issues: errors={n_errors}, warnings={n_warnings}")


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
    match_mode: Annotated[
        str,
        typer.Option(
            "--match-mode",
            help=(
                "Variant matching strategy for the polars engine: "
                "'position' matches on chromosome+position+alleles (default), "
                "'id' matches on synthetic CHROM:POS:REF:ALT IDs for PLINK-parity."
            ),
            case_sensitive=False,
        ),
    ] = "position",
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
    match_mode = match_mode.lower()
    if match_mode not in {"position", "id"}:
        console.print(
            f"[red]Invalid --match-mode '{match_mode}'. Expected 'position' or 'id'.[/red]"
        )
        raise typer.Exit(code=1)
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
    console.print(f"[bold]Engine 2: pgenlib + polars ({match_mode} match)[/bold]")
    t0 = _time.monotonic()
    polars_df = compute_reference_prs_polars(
        pgs_id=pgs_id,
        scoring_file=scoring_file,
        ref_dir=ref_dir,
        out_dir=cache / "reference_scores" / f"{pgs_id}_polars",
        genome_build=build,
        match_mode=match_mode,
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


# ---------------------------------------------------------------------------
# plot subcommand group — Altair chart generation
# ---------------------------------------------------------------------------

plot_app = typer.Typer(
    name="plot",
    help="Generate PRS charts: bell curves, multi-ancestry overlays, trait plots, and percentile strips.",
    no_args_is_help=True,
)
app.add_typer(plot_app, name="plot")


def _load_distributions(
    cache_dir: Path | None = None,
    panel: str = "1000g",
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load reference distributions and quality parquets from cache or HuggingFace."""
    cache = cache_dir or resolve_cache_dir()
    pctl_dir = cache / "percentiles"

    dist_path = pctl_dir / f"{panel}_distributions.parquet"
    quality_path = pctl_dir / f"{panel}_quality.parquet"

    if not dist_path.exists():
        console.print(f"Distributions not cached locally; pulling from HuggingFace...")
        catalog = PRSCatalog(cache_dir=cache)
        ref_dists = catalog.reference_distributions()
        if ref_dists is None or ref_dists.height == 0:
            console.print("[red]No reference distributions available. Run the pipeline or check HuggingFace.[/red]")
            raise typer.Exit(code=1)
        return ref_dists, pl.read_parquet(quality_path) if quality_path.exists() else None

    dists = pl.read_parquet(dist_path)
    quality = pl.read_parquet(quality_path) if quality_path.exists() else None
    return dists, quality


def _load_user_results(results_path: Path) -> list[dict]:
    """Load user PRS results from a JSON file."""
    import json as _json
    data = _json.loads(results_path.read_text())
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        console.print("[red]Results JSON must be a list of objects (or {\"results\": [...]}).[/red]")
        raise typer.Exit(code=1)
    return data


def _compute_trait_results(
    trait: str,
    vcf_path: Path,
    distributions_df: pl.DataFrame,
    ancestry: str,
    build: str,
    max_scores: int,
    cache: Path,
    fuzzy: bool = False,
    no_cache: bool = False,
) -> list[dict]:
    """Search for PGS models matching a trait, compute PRS, and return result dicts.

    Exact match on trait_reported first. Falls back to substring match only with fuzzy=True.
    Always lists the matched trait names before computing.
    Uses a file-based result cache keyed by (vcf mtime+size, pgs_id, build, ancestry).
    """
    catalog = PRSCatalog(cache_dir=cache)
    term = trait.strip().lower()

    all_scores = catalog.scores(genome_build=build, include_harmonized=True)

    if fuzzy:
        fuzzy_df = catalog.search(trait, genome_build=build).collect()
        if fuzzy_df.height == 0:
            console.print(f"[red]No PGS scores found for trait '{trait}' ({build}).[/red]")
            raise typer.Exit(code=1)
        scores_df = fuzzy_df
        match_type = "fuzzy"
    else:
        exact_df = all_scores.filter(
            pl.col("trait_reported").str.to_lowercase().eq(term)
        ).collect()

        if exact_df.height > 0:
            scores_df = exact_df
            match_type = "exact"
        else:
            fuzzy_df = catalog.search(trait, genome_build=build).collect()
            if fuzzy_df.height == 0:
                console.print(f"[red]No PGS scores found for trait '{trait}' ({build}).[/red]")
                raise typer.Exit(code=1)

            trait_names = fuzzy_df["trait_reported"].unique().sort().to_list()
            console.print(f"[yellow]No exact match for '{trait}'. Found {fuzzy_df.height} scores with partial matches:[/yellow]")
            for t in trait_names:
                n = fuzzy_df.filter(pl.col("trait_reported") == t).height
                console.print(f"  - {t} ({n} scores)")
            console.print(f"\n[dim]Add --fuzzy to compute PRS for these partial matches.[/dim]")
            raise typer.Exit(code=1)

    trait_names = scores_df["trait_reported"].unique().sort().to_list()
    pgs_ids = scores_df["pgs_id"].head(max_scores).to_list()

    pgs_to_trait = {
        row["pgs_id"]: row["trait_reported"]
        for row in scores_df.select("pgs_id", "trait_reported").iter_rows(named=True)
    }
    name_cols = ["pgs_id", "name"] if "name" in scores_df.columns else ["pgs_id"]
    pgs_to_name = {
        row["pgs_id"]: row.get("name", "")
        for row in scores_df.select(name_cols).iter_rows(named=True)
    } if "name" in scores_df.columns else {}

    quality_cols = [c for c in ("pgs_id", "quality_label", "synthetic_score", "is_harmonized") if c in scores_df.columns]
    pgs_to_quality: dict[str, dict] = {}
    if "quality_label" in scores_df.columns:
        for row in scores_df.select(quality_cols).iter_rows(named=True):
            pgs_to_quality[row["pgs_id"]] = row

    best_perf_df = catalog.best_performance().collect()
    perf_cols = [c for c in ("pgs_id", "auroc_estimate", "or_estimate", "hr_estimate",
                              "cindex_estimate", "n_individuals", "ancestry_broad") if c in best_perf_df.columns]
    pgs_to_perf: dict[str, dict] = {}
    if best_perf_df.height > 0:
        for row in best_perf_df.select(perf_cols).iter_rows(named=True):
            pgs_to_perf[row["pgs_id"]] = row

    console.print(f"Found [cyan]{scores_df.height}[/cyan] scores ({match_type} match) for {len(trait_names)} trait(s):")
    for t in trait_names:
        n = scores_df.filter(pl.col("trait_reported") == t).height
        console.print(f"  - {t} ({n} scores)")

    from just_prs.prs import compute_prs_duckdb

    def attach_risk_context(rd: dict) -> None:
        z_score = rd.get("z_score")
        if z_score is None:
            percentile = rd.get("percentile")
            if percentile is None:
                return
            try:
                from statistics import NormalDist

                p = max(0.001, min(99.999, float(percentile))) / 100.0
                z_score = NormalDist().inv_cdf(p)
                rd["z_score"] = z_score
            except (TypeError, ValueError):
                return
        try:
            bundle = catalog.absolute_risk_bundle(str(rd["pgs_id"]), float(z_score))
        except Exception:
            return

        if bundle.best_estimate is not None:
            be = bundle.best_estimate
            rd["absolute_risk"] = be.absolute_risk
            rd["risk_ratio"] = be.risk_ratio
            rd["population_prevalence"] = be.population_prevalence
            rd["risk_method"] = be.method_label

        h2_estimates = [est for est in bundle.estimates if est.h2_value is not None]
        if h2_estimates:
            h_parts: list[str] = []
            h_metrics: list[dict[str, str]] = []
            for est in h2_estimates:
                population = est.ancestry or "combined"
                source = est.h2_source or est.method_label
                h_parts.append(f"{population} h²={est.h2_value:.3f} ({source})")
                h_metrics.append({
                    "population": population,
                    "h2": f"{est.h2_value:.3f}",
                    "source": source,
                    "risk": f"{est.absolute_risk * 100:.1f}%",
                    "ratio": f"{est.risk_ratio:.2f}x",
                    "confidence": est.confidence,
                })
            rd["heritability"] = "; ".join(h_parts)
            rd["heritability_metrics"] = h_metrics
            rd["heritability_detail"] = (
                "h² is population-level heritability: the fraction of trait variation "
                "statistically associated with genetic differences in a studied population, "
                "not an individual causal percentage."
            )
        else:
            rd.setdefault("heritability", "No mapped h²")
            rd.setdefault("heritability_detail", bundle.heritability_detail)

    result_dicts: list[dict] = []
    failed: list[str] = []
    cached_count = 0

    for i, pgs_id in enumerate(pgs_ids, 1):
        trait_name = pgs_to_trait.get(pgs_id, "?")

        if not no_cache:
            hit = _get_cached_result(vcf_path, pgs_id, build, ancestry, cache)
            if hit is not None:
                cached_count += 1
                hit.setdefault("trait_reported", trait_name)
                hit.setdefault("score_name", pgs_to_name.get(pgs_id, ""))
                attach_risk_context(hit)
                result_dicts.append(hit)
                console.print(
                    f"[{i}/{len(pgs_ids)}] [cyan]{pgs_id}[/cyan] — {trait_name}  "
                    f"[dim](cached) score={hit['score']:.6f}  percentile={hit.get('percentile')}[/dim]"
                )
                continue

        console.print(f"[{i}/{len(pgs_ids)}] [cyan]{pgs_id}[/cyan] — {trait_name}")
        try:
            r = compute_prs_duckdb(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=build,
                cache_dir=cache,
                pgs_id=pgs_id,
                trait_reported=trait_name,
            )
            pctl_result = catalog.percentile_full(
                r.score, r.pgs_id, ancestry=ancestry,
                weight_mass_coverage=r.weight_mass_coverage,
                user_match_rate=r.match_rate,
            )
            pctl = pctl_result.percentile
            z_score = pctl_result.z_score

            if not pctl_result.reliable:
                console.print(f"  [yellow]⚠ {pctl_result.caveat}[/yellow]")

            rd: dict = {
                "pgs_id": r.pgs_id,
                "score_name": pgs_to_name.get(r.pgs_id, ""),
                "score": r.score,
                "percentile": pctl,
                "z_score": z_score,
                "match_rate": r.match_rate,
                "variants_total": r.variants_total,
                "variants_matched": r.variants_matched,
                "trait_reported": trait_name,
                "reliable": pctl_result.reliable,
            }

            q = pgs_to_quality.get(r.pgs_id, {})
            if q.get("quality_label"):
                rd["quality_label"] = q["quality_label"]
            if q.get("synthetic_score") is not None:
                rd["synthetic_quality"] = q["synthetic_score"]
            if q.get("is_harmonized"):
                rd["is_harmonized"] = q["is_harmonized"]

            p = pgs_to_perf.get(r.pgs_id, {})
            if p.get("auroc_estimate") is not None:
                rd["auroc"] = p["auroc_estimate"]
            if p.get("or_estimate") is not None:
                rd["or_estimate"] = p["or_estimate"]
            if p.get("ancestry_broad"):
                rd["eval_ancestry"] = p["ancestry_broad"]

            attach_risk_context(rd)

            result_dicts.append(rd)
            _put_cached_result(vcf_path, pgs_id, build, ancestry, rd, cache)

            line = f"  score={r.score:.6f}  matched={r.variants_matched}/{r.variants_total}  percentile={pctl}"
            if "risk_ratio" in rd:
                line += f"  risk={rd['risk_ratio']:.2f}x"
            if "absolute_risk" in rd:
                line += f"  abs_risk={rd['absolute_risk']:.1%}"
            console.print(line)
        except Exception as exc:
            failed.append(pgs_id)
            console.print(f"  [red]failed: {exc}[/red]")

    if failed:
        console.print(f"\n[yellow]{len(failed)}/{len(pgs_ids)} scores failed: {', '.join(failed)}[/yellow]")
    summary_parts = [f"[green]{len(result_dicts)}[/green] scores"]
    if cached_count:
        summary_parts.append(f"{cached_count} from cache")
    if len(result_dicts) - cached_count > 0:
        summary_parts.append(f"{len(result_dicts) - cached_count} computed")
    console.print(f"\n{', '.join(summary_parts)}.")

    for rd in result_dicts:
        pid = rd.get("pgs_id", "")
        if not rd.get("quality_label"):
            q = pgs_to_quality.get(pid, {})
            if q.get("quality_label"):
                rd["quality_label"] = q["quality_label"]
            if q.get("synthetic_score") is not None:
                rd.setdefault("synthetic_quality", q["synthetic_score"])
            if q.get("is_harmonized"):
                rd.setdefault("is_harmonized", q["is_harmonized"])
        if not rd.get("auroc"):
            p = pgs_to_perf.get(pid, {})
            if p.get("auroc_estimate") is not None:
                rd.setdefault("auroc", p["auroc_estimate"])
            if p.get("or_estimate") is not None:
                rd.setdefault("or_estimate", p["or_estimate"])
            if p.get("ancestry_broad"):
                rd.setdefault("eval_ancestry", p["ancestry_broad"])

    scored = [r for r in result_dicts if r.get("percentile") is not None]
    if scored:
        pctls = sorted([r["percentile"] for r in scored])
        median_pctl = pctls[len(pctls) // 2]
        mean_pctl = sum(pctls) / len(pctls)
        best_r = max(scored, key=lambda r: r["percentile"])

        console.print("\n[bold]Key Statistics[/bold]")
        console.print(f"  Your Percentile (best model):  [bold]{best_r['percentile']:.1f}[/bold]  ({best_r['pgs_id']})")
        console.print(f"  Median Percentile:             [bold]{median_pctl:.1f}[/bold]  across {len(scored)} models")
        console.print(f"  Mean Percentile:               {mean_pctl:.1f}")
        if len(pctls) > 1:
            import statistics
            spread = statistics.pstdev(pctls)
            console.print(f"  Model Spread (SD):             {spread:.1f} pts")

        with_risk = [r for r in scored if "risk_ratio" in r]
        if with_risk:
            best_risk = max(with_risk, key=lambda r: r["percentile"])
            rr = best_risk["risk_ratio"]
            if rr >= 1.0:
                console.print(f"  Risk vs Average:               [bold red]{rr:.2f}x[/bold red]  (higher than population average)")
            else:
                console.print(f"  Risk vs Average:               [bold green]{rr:.2f}x[/bold green]  (lower than population average)")

        with_abs = [r for r in scored if "absolute_risk" in r]
        if with_abs:
            best_abs = max(with_abs, key=lambda r: r["percentile"])
            ar = best_abs["absolute_risk"]
            prev = best_abs.get("population_prevalence")
            method = best_abs.get("risk_method", "")
            risk_str = f"  Absolute Risk:                 [bold]{ar:.1%}[/bold]"
            if prev is not None:
                risk_str += f"  (pop. avg. {prev:.1%})"
            if method:
                risk_str += f"  [{method}]"
            console.print(risk_str)

    return result_dicts


@plot_app.command("bell-curve")
def plot_bell_curve_cmd(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file (.png, .svg, .html, .json)")],
    vcf: Annotated[
        Optional[str], typer.Option("--vcf", "-v", help="VCF file path or alias (e.g. 'anton'). Auto-computes PRS.")
    ] = None,
    ancestry: Annotated[
        str, typer.Option("--ancestry", "-a", help="Superpopulation code (AFR, AMR, EAS, EUR, SAS)")
    ] = "EUR",
    user_score: Annotated[
        Optional[float], typer.Option("--user-score", "-s", help="User's PRS score to mark on the curve")
    ] = None,
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (used with --vcf)")
    ] = "GRCh38",
    width: Annotated[int, typer.Option("--width", help="Chart width in pixels")] = 800,
    height: Annotated[int, typer.Option("--height", help="Chart height in pixels")] = 350,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass PRS result cache and recompute")
    ] = False,
    panel: Annotated[
        str, typer.Option("--panel", help="Reference panel (1000g or hgdp_1kg)")
    ] = "1000g",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Plot a single PGS bell curve for one ancestry with optional user score marker.

    Use --vcf (path or alias) to auto-compute PRS and mark it on the curve.
    Use --user-score to mark a pre-computed score directly.

    \b
    Examples:
      prs plot bell-curve PGS000001 -o bell.html
      prs plot bell-curve PGS000001 -o bell.html -a AFR --user-score 0.274
      prs plot bell-curve PGS000001 -o bell.html --vcf livia
      prs plot bell-curve PGS000001 -o bell.html --vcf anton -a EUR
    """
    from just_prs.viz import plot_prs_bell_curve, save_chart

    pgs_id = _validate_pgs_id(pgs_id)

    if vcf and user_score is not None:
        console.print("[red]Provide either --vcf or --user-score, not both.[/red]")
        raise typer.Exit(code=1)

    score: float | None = user_score
    if vcf:
        vcf_path = _resolve_vcf(vcf, cache_dir)
        cache = cache_dir or resolve_cache_dir()
        hit = None if no_cache else _get_cached_result(vcf_path, pgs_id, build, ancestry, cache)
        if hit is not None:
            score = hit["score"]
            console.print(f"[dim](cached)[/dim] Score: [green]{score:.6f}[/green]")
        else:
            console.print(f"Computing PRS for [cyan]{pgs_id}[/cyan] on {vcf_path}...")
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=build,
                cache_dir=cache,
                pgs_id=pgs_id,
            )
            score = result.score
            _put_cached_result(vcf_path, pgs_id, build, ancestry, {
                "pgs_id": pgs_id, "score": score, "match_rate": result.match_rate,
            }, cache)
            console.print(f"Score: [green]{score:.6f}[/green] (matched {result.variants_matched}/{result.variants_total})")

    dists, _ = _load_distributions(cache_dir, panel)

    console.print(f"Plotting bell curve for [cyan]{pgs_id}[/cyan] ({ancestry})...")
    chart = plot_prs_bell_curve(
        pgs_id=pgs_id,
        distributions_df=dists,
        user_score=score,
        ancestry=ancestry,
        width=width,
        height=height,
    )
    save_chart(chart, output)
    console.print(f"[green]Saved to {output.resolve()}[/green]")


@plot_app.command("multi-ancestry")
def plot_multi_ancestry_cmd(
    pgs_id: Annotated[str, typer.Argument(help="PGS score ID (e.g. PGS000001)")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file (.png, .svg, .html, .json)")],
    vcf: Annotated[
        Optional[str], typer.Option("--vcf", "-v", help="VCF file path or alias (e.g. 'anton'). Auto-computes PRS.")
    ] = None,
    user_score: Annotated[
        Optional[float], typer.Option("--user-score", "-s", help="User's PRS score to mark on the chart")
    ] = None,
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (used with --vcf)")
    ] = "GRCh38",
    ancestries: Annotated[
        Optional[str], typer.Option("--ancestries", help="Comma-separated superpopulation codes (default: all)")
    ] = None,
    width: Annotated[int, typer.Option("--width", help="Chart width in pixels")] = 800,
    height: Annotated[int, typer.Option("--height", help="Chart height in pixels")] = 350,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass PRS result cache and recompute")
    ] = False,
    panel: Annotated[
        str, typer.Option("--panel", help="Reference panel (1000g or hgdp_1kg)")
    ] = "1000g",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Plot overlaid bell curves for multiple ancestries on a single chart.

    \b
    Examples:
      prs plot multi-ancestry PGS000001 -o multi.html
      prs plot multi-ancestry PGS000001 -o multi.html --vcf anton
      prs plot multi-ancestry PGS000001 -o multi.html --ancestries EUR,AFR,EAS
      prs plot multi-ancestry PGS000001 -o multi.html --user-score 0.274
    """
    from just_prs.viz import plot_prs_multi_ancestry, save_chart

    pgs_id = _validate_pgs_id(pgs_id)

    if vcf and user_score is not None:
        console.print("[red]Provide either --vcf or --user-score, not both.[/red]")
        raise typer.Exit(code=1)

    score: float | None = user_score
    if vcf:
        vcf_path = _resolve_vcf(vcf, cache_dir)
        cache = cache_dir or resolve_cache_dir()
        hit = None if no_cache else _get_cached_result(vcf_path, pgs_id, build, "EUR", cache)
        if hit is not None:
            score = hit["score"]
            console.print(f"[dim](cached)[/dim] Score: [green]{score:.6f}[/green]")
        else:
            console.print(f"Computing PRS for [cyan]{pgs_id}[/cyan] on {vcf_path}...")
            result = compute_prs(
                vcf_path=vcf_path,
                scoring_file=pgs_id,
                genome_build=build,
                cache_dir=cache,
                pgs_id=pgs_id,
            )
            score = result.score
            _put_cached_result(vcf_path, pgs_id, build, "EUR", {
                "pgs_id": pgs_id, "score": score, "match_rate": result.match_rate,
            }, cache)
            console.print(f"Score: [green]{score:.6f}[/green] (matched {result.variants_matched}/{result.variants_total})")

    dists, _ = _load_distributions(cache_dir, panel)

    anc_list: list[str] | None = None
    if ancestries:
        anc_list = [a.strip().upper() for a in ancestries.split(",") if a.strip()]

    console.print(f"Plotting multi-ancestry curves for [cyan]{pgs_id}[/cyan]...")
    chart = plot_prs_multi_ancestry(
        pgs_id=pgs_id,
        distributions_df=dists,
        user_score=score,
        ancestries=anc_list,
        width=width,
        height=height,
    )
    save_chart(chart, output)
    console.print(f"[green]Saved to {output.resolve()}[/green]")


@plot_app.command("trait")
def plot_trait_cmd(
    trait: Annotated[str, typer.Argument(help="Trait name or substring (e.g. 'BMI', 'type 2 diabetes')")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file (.png, .svg, .html, .json)")],
    vcf: Annotated[
        Optional[str], typer.Option("--vcf", "-v", help="VCF file path or alias (e.g. 'anton', 'livia'). Auto-computes PRS for the trait.")
    ] = None,
    ancestry: Annotated[
        str, typer.Option("--ancestry", "-a", help="Superpopulation code (AFR, AMR, EAS, EUR, SAS)")
    ] = "EUR",
    results: Annotated[
        Optional[Path], typer.Option("--results", "-r", help="JSON file with user PRS results (list of {pgs_id, score, ...})")
    ] = None,
    build: Annotated[
        str, typer.Option("--build", "-b", help="Genome build (used with --vcf)")
    ] = "GRCh38",
    max_scores: Annotated[
        int, typer.Option("--max-scores", help="Max number of PGS models to show")
    ] = 25,
    show_table: Annotated[
        bool, typer.Option("--show-table/--no-table", help="Append a model summary table below the bell curve")
    ] = False,
    width: Annotated[int, typer.Option("--width", help="Chart width in pixels")] = 900,
    height: Annotated[int, typer.Option("--height", help="Bell curve height in pixels")] = 400,
    fuzzy: Annotated[
        bool, typer.Option("--fuzzy/--exact", help="Allow partial trait name matching (default: exact only)")
    ] = False,
    all_ancestries: Annotated[
        bool, typer.Option("--all-ancestries/--single-ancestry", help="Overlay all 5 population bell curves (default: single ancestry)")
    ] = False,
    ancestries: Annotated[
        Optional[str], typer.Option("--ancestries", help="Comma-separated population codes to overlay (e.g. EUR,AFR,EAS)")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass PRS result cache and recompute all scores")
    ] = False,
    panel: Annotated[
        str, typer.Option("--panel", help="Reference panel (1000g or hgdp_1kg)")
    ] = "1000g",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Override cache directory")
    ] = None,
) -> None:
    """Plot trait-grouped visualization: reference bell curve with per-model quality dots.

    Without --vcf or --results, shows model quality comparison as a dot strip.
    With --vcf, auto-computes PRS for all matching scores and plots the results.
    With --results, loads pre-computed results from a JSON file.

    Trait matching is exact by default (case-insensitive). If no exact match is
    found, the command lists partial matches and exits. Use --fuzzy to compute
    PRS for all partial matches.

    Use --all-ancestries to overlay all 5 population reference curves, or
    --ancestries EUR,AFR,EAS to select specific populations.

    PRS results are cached per (VCF, PGS ID, build, ancestry) so repeated
    plotting is instant. Use --no-cache to force recomputation.

    \b
    Examples:
      prs plot trait "Deep vein thrombosis" --vcf livia -o dvt.html --show-table
      prs plot trait thrombosis --vcf livia -o dvt.html --fuzzy --show-table
      prs plot trait BMI -o bmi.html --show-table --all-ancestries
      prs plot trait "type 2 diabetes" --vcf anton -o t2d.html --ancestries EUR,AFR,EAS
      prs plot trait "type 2 diabetes" -o t2d.html --results my_results.json
    """
    from just_prs.viz import plot_trait_scores, save_chart, save_trait_report

    if vcf and results:
        console.print("[red]Provide either --vcf or --results, not both.[/red]")
        raise typer.Exit(code=1)

    cache = cache_dir or resolve_cache_dir()
    dists, quality = _load_distributions(cache_dir, panel)

    user_results: list[dict] | None = None

    sample_name: str | None = None
    if vcf:
        vcf_path = _resolve_vcf(vcf, cache_dir)
        sample_name = vcf_path.name
        user_results = _compute_trait_results(
            trait, vcf_path, dists, ancestry, build, max_scores, cache,
            fuzzy=fuzzy, no_cache=no_cache,
        )
    elif results:
        sample_name = results.name
        user_results = _load_user_results(results)

    anc_list: list[str] | None = None
    default_visible: list[str] | None = None
    if all_ancestries:
        anc_list = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    elif ancestries:
        anc_list = [a.strip().upper() for a in ancestries.split(",")]

    html_report = user_results and output.suffix.lower() == ".html"

    if html_report and anc_list is None:
        anc_list = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        default_visible = [ancestry]

    pop_label = ", ".join(anc_list) if anc_list else ancestry
    console.print(f"Plotting trait chart for [cyan]{trait}[/cyan] ({pop_label})...")
    chart = plot_trait_scores(
        trait=trait,
        distributions_df=dists,
        quality_df=quality,
        user_results=user_results,
        ancestry=ancestry,
        ancestries=anc_list,
        default_visible_ancestries=default_visible,
        max_scores=max_scores,
        width=width,
        height=height,
        show_table=show_table and not html_report,
    )
    if html_report:
        save_trait_report(chart, output, trait, user_results, ancestry, sample_name=sample_name)
    else:
        save_chart(chart, output)
    console.print(f"[green]Saved to {output.resolve()}[/green]")


@plot_app.command("strip")
def plot_strip_cmd(
    results: Annotated[Path, typer.Argument(help="JSON file with PRS results (list of {pgs_id, percentile, ...})")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file (.png, .svg, .html, .json)")],
    title: Annotated[
        str, typer.Option("--title", "-t", help="Chart title")
    ] = "PRS Percentiles",
    width: Annotated[int, typer.Option("--width", help="Chart width in pixels")] = 800,
    height: Annotated[
        Optional[int], typer.Option("--height", help="Chart height in pixels (default: auto-sized)")
    ] = None,
) -> None:
    """Plot a horizontal percentile strip chart with risk-colored bands.

    Input JSON must be a list of objects with at least 'pgs_id' and 'percentile' keys.
    Optional keys: 'trait', 'label', 'quality_label'.

    \b
    Examples:
      prs plot strip results.json -o strip.png
      prs plot strip results.json -o strip.html --title "My PRS Report"
    """
    from just_prs.viz import plot_prs_percentile_strip, save_chart

    user_results = _load_user_results(results)

    missing = [r for r in user_results if "percentile" not in r or r["percentile"] is None]
    if missing:
        console.print(
            f"[yellow]Warning: {len(missing)} result(s) missing 'percentile' field — "
            f"they will be skipped.[/yellow]"
        )
        user_results = [r for r in user_results if r.get("percentile") is not None]

    if not user_results:
        console.print("[red]No results with valid percentiles to plot.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Plotting percentile strip for [cyan]{len(user_results)}[/cyan] scores...")
    chart = plot_prs_percentile_strip(
        results=user_results,
        title=title,
        width=width,
        height=height,
    )
    save_chart(chart, output)
    console.print(f"[green]Saved to {output.resolve()}[/green]")


def run() -> None:
    """Entry point for the CLI."""
    app()
