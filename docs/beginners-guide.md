# Beginner's guide: run just-prs on your computer

This page is for people who have never used this repo. It walks through
installing the toolbox, opening the web app, and connecting it to an AI
assistant (MCP). You do **not** need to rebuild reference panels or run the
Dagster pipeline.

**This is research software, not medical advice.** A polygenic risk score (PRS)
is a statistical predisposition signal. It is not a diagnosis.

## What you are installing

`just-prs` scores a genome file against thousands of published models from the
[PGS Catalog](https://www.pgscatalog.org/), then shows where that score sits
compared with a reference population.

Three ways to use it:

| Path | What you get | Who it is for |
|------|----------------|---------------|
| **Web UI** | Browser app at http://localhost:3000 | Most people |
| **CLI** | Commands like `prs compute` | Scripts and terminals |
| **MCP** | Tools inside Claude, Cursor, or Codex | People who want to ask in plain language |

You can use one, two, or all three. They share the same cache and the same
engine.

## What you need

- A computer with **8 GB RAM** (16 GB is more comfortable)
- About **2 GB** free disk to start. The cache grows as you score more models
  (roughly 10 MB per scoring file; a full one-build catalog is tens of GB)
- **Python 3.13 or newer**
- **[uv](https://docs.astral.sh/uv/)** — a package installer. Think of it as
  “the thing that installs this project correctly”
- A **VCF** (variant call file from sequencing) **or** a consumer array export
  (23andMe / AncestryDNA / MyHeritage) if you want to use the **web UI** — the
  UI only accepts a file you drop in. Public demo genomes from Zenodo (`anton`,
  `livia`) download from the **CLI** (and from **MCP**), not from the browser.

### Windows vs Linux / WSL

**Windows is fine** for the web UI, VCF scoring, plots, and MCP. Percentiles
still work: they download **already computed** tables from HuggingFace.

Use **Linux or WSL** only if you want to run the **testing / reference
pipeline** (`uv run pipeline run` and related `prs reference` jobs). That is
the path that scores the 1000 Genomes panel and rebuilds published tables. You
do not need it to try the app.

## 1. Install uv

Official instructions: [docs.astral.sh/uv — Installation](https://docs.astral.sh/uv/getting-started/installation/).

**Windows — easiest if you use winget** (App Installer, built into recent Windows):

```powershell
winget install --id=astral-sh.uv -e
```

**Windows — official PowerShell installer** (same idea as `curl | sh` on Linux):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`irm` is short for `Invoke-RestMethod` (download the script). `iex` is
`Invoke-Expression` (run it). That pair is Astral’s documented Windows command,
not a project-specific trick. If execution policy blocks it, the
`-ExecutionPolicy ByPass` line above is the form they publish.

Close and reopen the terminal, then check:

```powershell
uv --version
```

**macOS / Linux / WSL:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

If `uv` is “not recognized”, the installer put it on your PATH but this
terminal session has not picked it up. Open a new window.

## 2. Get the code

```bash
git clone https://github.com/dna-seq/just-prs.git
cd just-prs
```

If you already have a zip or a clone, `cd` into that folder instead. The
commands below assume you are in the **repo root** (the folder that contains
`pyproject.toml`, `prs-ui/`, and `just-prs/`).

Optional: copy the settings template. Everything has defaults; you only need
this file if you want to change ports or add a HuggingFace token.

```bash
# Windows
copy .env.template .env

# macOS / Linux / WSL
cp .env.template .env
```

## 3. Install the project

```bash
uv sync --all-packages
```

This downloads Python packages into a project-local environment. The first run
can take several minutes. On Windows it will **not** try to compile `pgenlib`
or `pysam`.

If this fails, see [Troubleshooting](#troubleshooting).

## 4. Start the web UI

```bash
uv run ui
```

(`uv run start` is the same command.)

The first lines of output print the URL. Default:

- App: **http://localhost:3000**
- Backend: port **8000**

Leave that terminal open. Open the URL in your browser.

### First click-through

1. Stay on **Compute PRS**.
2. Drop a VCF (or several, for a family comparison). The app detects the genome
   build and normalizes the file. That step can take a few minutes the first
   time; a spinner is expected. Re-uploading the same file is fast.
3. Pick **Select by PRS** (individual models) or **Select by Trait** (every
   model for e.g. type 2 diabetes).
4. Click compute. Results include match rate, quality, a bell curve, and
   (when data exists) absolute-risk context.

The web UI does **not** fetch the Zenodo demo genomes. Drop your own VCF, or
download `anton` / `livia` from the CLI (next section) and then upload that
file. `uv run preselect` only helps if `PRS_UI_PRESELECT_VCF` already points at
a local path.

## 5. Try the CLI without your own file

Built-in aliases `anton` and `livia` download public whole-genome VCFs from
Zenodo the first time you use them (~350–480 MB). This auto-download is a
**CLI** feature (`--vcf anton`). **MCP** can do the same
(`download_sample_genome`). The **web UI cannot** — it has no Zenodo button.

```bash
uv run prs compute --vcf anton --pgs-id PGS000001
uv run prs plot trait BMI --vcf anton -o bmi.html --show-table
```

Open `bmi.html` in a browser. Later CLI commands reuse the cache, so repeats
are much faster.

Your own file:

```bash
uv run prs compute --vcf C:\path\to\sample.vcf.gz --pgs-id PGS000001
```

On macOS/Linux use a normal `/path/to/sample.vcf.gz`.

## 6. Deploy MCP (talk to just-prs from an AI assistant)

**MCP** (Model Context Protocol) is a standard way for an AI chat app to call
local tools. [just-prs-mcp](https://github.com/dna-seq/just-prs-mcp) is a
**separate small server** that wraps this toolbox.

You do **not** need to clone `just-prs-mcp`. `uvx` downloads the published
package on first use.

What this does **not** do:

- It does not upload your VCF to a cloud API
- It does not need an API key
- Over stdio, file paths are on **your** machine

### One-time: make sure `uvx` works

```bash
uvx just-prs-mcp@latest --help
```

The first run downloads the package. If this command fails, fix uv before
editing any MCP config.

### Cursor

Create or edit `.cursor/mcp.json` in this project (or your user MCP config —
see [Cursor MCP docs](https://cursor.com/docs/mcp)):

```json
{
  "mcpServers": {
    "just-prs": {
      "command": "uvx",
      "args": ["just-prs-mcp@latest", "stdio"],
      "env": {
        "PRS_MCP_MODE": "essentials"
      }
    }
  }
}
```

Restart Cursor (or reload MCP servers). You should see a `just-prs` server
connected. Then ask something like:

> Download Anton's sample genome, normalize it, and compute PRS for type 2
> diabetes.

`essentials` is the beginner mode: catalog search, normalize, score, percentile,
quality, compare. Use `"PRS_MCP_MODE": "extended"` only if you need bulk
downloads or reference-panel scoring (Linux/WSL).

### Claude Code

```bash
claude mcp add just-prs -- uvx just-prs-mcp@latest stdio
claude mcp list
```

Pin a version in a shared lab setup:

```bash
claude mcp add just-prs -- uvx just-prs-mcp@0.2.0 stdio
```

### Codex

In `~/.codex/config.toml`:

```toml
[mcp_servers.just-prs]
command = "uvx"
args = ["just-prs-mcp@latest", "stdio"]
```

### Other clients (Claude Desktop, custom apps)

Same local process:

```json
{
  "command": "uvx",
  "args": ["just-prs-mcp@latest", "stdio"],
  "env": {
    "PRS_MCP_MODE": "essentials"
  }
}
```

HTTP instead of stdio (default port **3011**), if your client wants a URL:

```bash
uvx just-prs-mcp@latest http
```

There is also a Claude plugin and a `.mcpb` Desktop extension in the
[just-prs-mcp](https://github.com/dna-seq/just-prs-mcp) repo if you prefer a
click-to-install package.

### MCP without MCP: the `/prs` skill

If you only want the assistant to run CLI commands (no tool schemas), copy
[`docs/skills/prs/SKILL.md`](skills/prs/SKILL.md) into your agent's skills
folder. That path does not start the MCP server.

### Check that MCP is actually working

1. `uvx just-prs-mcp@latest --help` succeeds in a normal terminal.
2. The client shows the server as connected (green / listed).
3. Ask it to `search_scores` for `"BMI"` or to `download_sample_genome` with
   `sample="anton"`.
4. If the client says the server failed to start, look at that client's MCP
   log. The usual cause is `uvx` not on the PATH that the GUI app sees.

## 7. Optional settings (`.env`)

Copy `.env.template` to `.env` in the repo root. Useful knobs:

| Variable | Default | When to set it |
|----------|---------|----------------|
| `PRS_UI_PORT` | `3000` | Port 3000 is already taken |
| `PRS_UI_BACKEND_PORT` | `8000` | Port 8000 is already taken |
| `PRS_CACHE_DIR` | OS user cache | You want the cache on another drive |
| `HF_TOKEN` | unset | HuggingFace rate-limits you |

Windows cache default: `%LOCALAPPDATA%\just-prs\Cache\`  
Linux / WSL: `~/.cache/just-prs/`

Do not commit `.env`. It can hold tokens.

## What not to run yet

| Command | Why wait |
|---------|----------|
| `uv run pipeline run` | The testing / reference pipeline (Dagster). Linux/WSL. Hours, many GB. Skip unless that is what you came for. |
| `uv run prs reference download` | ~7 GB 1000 Genomes panel. Part of that same pipeline path. |
| `uv run pipeline ld-proxy` | Heavy memory job. Not for a first install. |

If you put the **web UI** on a shared server, do not accept private genome
uploads unless you intend to. For a public demo, prefer the built-in public
genomes and run private files only on the owner's machine.

## Troubleshooting

### `uv` is not recognized

The installer finished, but this terminal is old. Open a new PowerShell /
Terminal window. On Windows you can also try:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv --version
```

### `uv sync` fails on Python version

The project needs **Python 3.13+**. Let uv install it:

```bash
uv python install 3.13
uv sync --all-packages
```

### `uv sync` tries to compile `pgenlib` on Windows

You should not see this on current `main`. If you do, you are on an old
checkout or you overrode the Windows marker. Update the repo (`git pull`) and
sync again. The UI does not need `pgenlib`.

### `uv run ui` starts but the browser is blank / “connection refused”

- Read the first lines of the terminal. Use **that** URL, not a guess.
- Another app may own port 3000 or 8000. Set `PRS_UI_PORT` and
  `PRS_UI_BACKEND_PORT` in `.env` to free ports, then restart `uv run ui`.
- Wait until both frontend and backend have finished starting. The first
  Reflex compile can take a minute.

### UI says it is “normalizing” for a long time

Normalization is CPU work on a large VCF. A 300–500 MB file can take several
minutes. There is no trustworthy percent bar; the spinner is the real signal.
If it never finishes, check the terminal for a Python traceback (bad VCF,
disk full, out of memory).

### First compute is slow / “downloading”

The first score for a PGS ID downloads a scoring file and (if needed) catalog
metadata and percentile tables. Later runs use the cache. A flaky network
looks like a hang; retry once.

### `pgenlib is required` / `pysam is required`

You ran a Linux-only command on Windows (`prs pgen`, `prs reference score`,
pipeline scoring, FASTA universe build). Use the UI / `prs compute` instead,
or move that job to WSL.

### MCP server never connects in Cursor / Claude

1. Run `uvx just-prs-mcp@latest --help` in the same kind of terminal the app
   would use. GUI apps on Windows sometimes do not see `uv` if it was only
   added to your user PATH after the app started — fully quit and reopen the
   editor.
2. Use `just-prs-mcp@latest` or a pinned version. A bare `just-prs-mcp` can
   stick on the first cached version `uvx` ever downloaded.
3. Keep `"PRS_MCP_MODE": "essentials"` until the server stays up.
4. Extended-mode reference tools need Linux/WSL + `pgenlib`. They are unused
   in essentials.

### HuggingFace / catalog download errors

Create a free [HuggingFace](https://huggingface.co/) account, make a token,
and put `HF_TOKEN=...` in `.env`. Then retry. Public pulls often work without
a token; a token helps when you are rate-limited.

### Disk filled up

```bash
uv run prs cache report
```

That command is read-only. It tells you what is in the cache and what looks
like leftover junk. Do not delete files by hand unless you understand the
report. Scoring files live as parquet under the cache `scores/` folder.

### WSL is painfully slow

Do not run from `/mnt/c/...`. Clone the repo under your Linux home
(`~/just-prs`) so files live on the Linux filesystem.

## Next reading

- [README](../README.md) — feature overview
- [CLI reference](cli.md)
- [just-prs-mcp](https://github.com/dna-seq/just-prs-mcp) — full MCP tool list
- [Research-use FAQ](../README.md#research-use-only) — how to read a result
