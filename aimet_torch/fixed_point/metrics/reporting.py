# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Reusable renderers for per-layer metric tables.

The same row shape produced by :func:`per_layer_chained_cosine` and
:func:`per_layer_isolated_cosine` is rendered here as a markdown table that
fits both CI artifacts (``--md-out`` files) and console output. Other
models can reuse :func:`render_per_layer_table` directly without having to
re-implement column formatting.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List


def _fmt_sqnr(value: float) -> str:
    """Right-pad SQNR to 7 chars; render +inf as ``  +inf``."""

    if value == float("inf") or (isinstance(value, float) and math.isinf(value)):
        return "  +inf"
    return f"{value:7.2f}"


def _fmt_shape(shape: Iterable[int]) -> str:
    return "×".join(str(d) for d in shape)


def render_per_layer_table(
    rows: List[Dict[str, Any]],
    *,
    cosine_key: str = "cosine",
    printer: Callable[[str], None] = print,
) -> None:
    """Print a unified Tier-1 metric table from per-layer ``rows``.

    Columns: ``module | <cosine_key> | sqnr_dB | rmse | p99_abs_err | shape``.
    Missing metric fields fall back to ``nan`` formatting so the function
    stays robust against older row schemas.

    Parameters
    ----------
    rows
        Output of :func:`per_layer_chained_cosine` /
        :func:`per_layer_isolated_cosine` (or anything sharing the same
        row keys).
    cosine_key
        Column key for the cosine value (``"cosine"`` for chained,
        ``"isolated_cosine"`` for isolated).
    printer
        Sink for output lines; defaults to the built-in :func:`print`.
        Pass a list's ``append`` to capture into a buffer, or a file
        handle's ``write`` (wrap with ``lambda s: f.write(s + "\\n")``).
    """

    printer(f"| module | {cosine_key} | sqnr_dB | rmse | p99_abs_err | shape |")
    printer("|--------|--------|---------|------|-------------|-------|")
    for row in rows:
        shape = _fmt_shape(row.get("shape", ()))
        cos = row.get(cosine_key, float("nan"))
        sqnr = _fmt_sqnr(row.get("sqnr_db", float("nan")))
        rmse = row.get("rmse", float("nan"))
        p99 = row.get("p99_abs_err", float("nan"))
        printer(
            f"| `{row.get('module', '?')}` | {cos:.6f} | "
            f"{sqnr} | {rmse:.3e} | {p99:.3e} | {shape} |"
        )


def per_layer_table_to_markdown(
    rows: List[Dict[str, Any]],
    *,
    cosine_key: str = "cosine",
) -> str:
    """Return :func:`render_per_layer_table`'s output as a single string.

    Convenience for writing markdown report artifacts.
    """

    lines: List[str] = []
    render_per_layer_table(rows, cosine_key=cosine_key, printer=lines.append)
    return "\n".join(lines)
