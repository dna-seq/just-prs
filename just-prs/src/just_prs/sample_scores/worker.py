"""Subprocess entry for one public-sample scoring worker.

The parent owns the checkpoint queue. This process may score several
checkpoints to amortize reference-universe loading, then exit so RSS
returns to the OS.
"""

from __future__ import annotations

import sys
from pathlib import Path

from just_prs.sample_scores.engine import WorkerWorkOrder, run_worker


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        sys.stderr.write("usage: python -m just_prs.sample_scores.worker WORK.json REPORT.json\n")
        return 2
    work_path = Path(args[0])
    report_path = Path(args[1])
    order = WorkerWorkOrder.model_validate_json(work_path.read_text(encoding="utf-8"))

    def _emit(message: str) -> None:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    try:
        report = run_worker(order, log=_emit)
    except Exception as exc:
        from just_prs.sample_scores.engine import WorkerReport

        report = WorkerReport(profile_id=order.profile_id, error=str(exc))
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
