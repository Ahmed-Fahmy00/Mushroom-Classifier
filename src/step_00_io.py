from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	run_root = output_root / timestamp

	plots = run_root / "plots"
	tables = run_root / "tables"
	reports = run_root / "reports"

	plots.mkdir(parents=True, exist_ok=True)
	tables.mkdir(parents=True, exist_ok=True)
	reports.mkdir(parents=True, exist_ok=True)
	return {
		"run_root": run_root,
		"plots": plots,
		"tables": tables,
		"reports": reports,
	}


def save_plot(file_path: Path) -> None:
	plt.tight_layout()
	plt.savefig(file_path, dpi=300, bbox_inches="tight")
	plt.close()


__all__ = ["ensure_output_dirs", "save_plot"]
