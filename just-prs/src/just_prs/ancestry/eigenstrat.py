"""Minimal reader for the AADR EIGENSTRAT / packed-`TGENO` (transpose_packed) format.

Used to build the `aadr_ho` ancestry model from the Allen Ancient DNA Resource Human
Origins present-day individuals (fine European / Slavic resolution). Read-only, numpy.

Layout (confirmed against AADR v66.p1 compatibility_HO):
- `.ind`  : one line per individual `IID  SEX  GROUP` (order matches the .geno records).
- `.snp`  : `SNPID  CHR  GPOS  POS  REF  VAR` — REF (col5) matches hg19 (GRCh37).
- `.anno` : tab-separated; col1 Genetic ID, col11 "Date mean in BP" (present-day = 0),
            col15 Group ID, col18/19 Lat/Long.
- `.geno` : packed **TGENO** — 48-byte ASCII header (`TGENO <nind> <nsnp> <hash> <hash>`),
            then `nind` **individual-major** records of `ceil(nsnp/4)` bytes; each byte packs
            4 SNPs at 2 bits, MSB-first; value 0/1/2 = genotype, 3 = missing. (The genotype
            counts the **reference** allele, .snp col5 — verified end-to-end at build.)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

TGENO_HEADER_BYTES = 48


def parse_ind(ind_path: Path) -> pl.DataFrame:
    """Parse a `.ind` file → DataFrame `iid, sex, group` in .geno record order (with `idx`)."""
    rows = []
    with open(ind_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return pl.DataFrame(
        {"iid": [r[0] for r in rows], "sex": [r[1] for r in rows], "group": [r[2] for r in rows]}
    ).with_row_index("idx")


def parse_snp(snp_path: Path) -> pl.DataFrame:
    """Parse a `.snp` file → DataFrame `chrom, pos, ref, alt` (ref = hg19, col5).

    `.snp` is whitespace-padded (variable spaces), so read robustly line by line.
    """
    chrom, pos, ref, alt = [], [], [], []
    with open(snp_path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 6:
                chrom.append(p[1]); pos.append(int(p[3])); ref.append(p[4]); alt.append(p[5])
    return pl.DataFrame({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt})


def parse_anno_present_day(anno_path: Path) -> dict[str, dict]:
    """Parse `.anno` → {Genetic ID: {group, date_bp, lat, lon}} for present-day filtering.

    Present-day individuals have Date-BP (col 11, 1-based) == 0.
    """
    out: dict[str, dict] = {}
    with open(anno_path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)  # header
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 19:
                continue
            try:
                date_bp = float(c[10])
            except ValueError:
                date_bp = float("nan")

            def _f(x):
                try:
                    return float(x)
                except ValueError:
                    return float("nan")

            out[c[0]] = {"group": c[14], "date_bp": date_bp, "lat": _f(c[17]), "lon": _f(c[18])}
    return out


def read_tgeno(
    geno_path: Path, n_ind: int, n_snp: int, ind_indices: np.ndarray
) -> np.ndarray:
    """Read selected individuals from a packed TGENO `.geno` → (n_selected x n_snp) int8.

    Values 0/1/2 = reference-allele count, -9 = missing. Individual-major: record ``i`` at
    byte offset ``48 + i*ceil(n_snp/4)``.
    """
    rlen = math.ceil(n_snp / 4)
    expected = TGENO_HEADER_BYTES + n_ind * rlen
    actual = geno_path.stat().st_size
    if actual != expected:
        raise ValueError(f"TGENO size {actual} != expected {expected} (n_ind={n_ind}, n_snp={n_snp})")

    out = np.empty((len(ind_indices), n_snp), dtype=np.int8)
    with open(geno_path, "rb") as fh:
        for row, i in enumerate(ind_indices):
            fh.seek(TGENO_HEADER_BYTES + int(i) * rlen)
            raw = np.frombuffer(fh.read(rlen), dtype=np.uint8)
            g = np.empty((rlen, 4), dtype=np.int8)
            g[:, 0] = (raw >> 6) & 3
            g[:, 1] = (raw >> 4) & 3
            g[:, 2] = (raw >> 2) & 3
            g[:, 3] = raw & 3
            g = g.reshape(-1)[:n_snp]
            out[row] = g
    out[out == 3] = -9  # missing sentinel
    return out  # (n_selected x n_snp)
