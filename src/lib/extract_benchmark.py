"""Shared timing helpers for extract_png_from_fits (CPU) and extract_png_from_fits_experimental_gpu."""

from __future__ import annotations

import platform
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from paths import OUTPUT_DIR


@dataclass
class BenchmarkRow:
    phase: str
    seconds: float
    detail: str = ""


class ExtractBenchmark:
    """Collects named intervals (seconds) and writes a text report."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.rows: List[BenchmarkRow] = []
        self._t0 = time.perf_counter()

    def record(self, phase: str, seconds: float, detail: str = "") -> None:
        self.rows.append(BenchmarkRow(phase, seconds, detail))

    @contextmanager
    def span(self, phase: str, detail: str = "") -> Iterator[None]:
        t0 = time.perf_counter()
        yield
        self.record(phase, time.perf_counter() - t0, detail)

    def total_elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def write_report(self, path: str, *, input_json: str, argv: Optional[List[str]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        totals: Dict[str, float] = defaultdict(float)
        for r in self.rows:
            totals[r.phase] += r.seconds
        lines = [
            "extract benchmark report",
            "=" * 72,
            f"title: {self.title}",
            f"input_json: {input_json}",
            f"wall_clock_total_s: {self.total_elapsed():.6f}",
            f"python: {sys.version.split()[0]} ({sys.executable})",
            f"platform: {platform.platform()}",
            "",
            "argv:",
            "  " + " ".join(argv if argv is not None else sys.argv),
            "",
            "per_row (phase, seconds, detail):",
            "-" * 72,
        ]
        for r in self.rows:
            d = f" | {r.detail}" if r.detail else ""
            lines.append(f"  {r.phase:<36} {r.seconds:12.6f}{d}")
        lines.extend(["", "sum_by_phase:", "-" * 72])
        for phase in sorted(totals.keys()):
            lines.append(f"  {phase:<36} {totals[phase]:12.6f}")
        lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")


def default_benchmark_path(tag: str) -> str:
    d = OUTPUT_DIR / "benchmark"
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(d / f"extract_{tag}_{ts}.txt")
