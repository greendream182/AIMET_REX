"""Lightweight regression for ``examples/quick_start_int16_metric.py``
defaults that encode SYS-OPEN-Q-1 W6 findings (commit ``78fffef``).

Background
----------
SYS-OPEN-Q-1 W6 sweep showed that ``percentile=99.5`` mitigates the
``part A`` outlier-driven SQNR loss on three backbone nodes
(``fc0`` SQNR ``5.34 -> 13.02`` dB, ``freq_downs.2.conv2d`` cosine
``0.776 -> 0.918``, ``neck_seqs.1.conv_t`` cosine
``0.844 -> 0.911``). The metric script now overrides the upstream
educational default of ``99.99`` with ``99.5`` so users get the W6
mitigation by default.

Scope of this test
------------------
This test does **not** re-run the MRNN INT16 metric pipeline (~40 s
per scheme; that integration check belongs to a future SYS-FU-2
ticket) and intentionally avoids importing
``examples.quick_start_int16_metric`` (whose package layout pulls in
``common.torch_stft`` and other heavy deps).

It freezes the **default value contract** at the source-string level:

1. The module declares a constant ``SYSQ1_W6_PERCENTILE_VALUE = 99.5``.
2. The argparse declaration ``--percentile-value`` is wired to that
   constant rather than to the upstream ``PERCENTILE_VALUE`` import,
   so accidental future PRs that revert to ``99.99`` will trip this
   test.

If a future change wants to move the default again it should:

- update ``SYSQ1_W6_PERCENTILE_VALUE`` and document the new sweep, or
- explicitly remove the override and revert this test.
"""

# pylint: disable=missing-function-docstring
from __future__ import annotations

import re
from pathlib import Path

import pytest


_QSI_SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "quick_start_int16_metric.py"
)


@pytest.fixture(scope="module")
def qsi_source() -> str:
    assert _QSI_SOURCE_PATH.is_file(), (
        f"quick_start_int16_metric.py not found at {_QSI_SOURCE_PATH}"
    )
    return _QSI_SOURCE_PATH.read_text(encoding="utf-8")


def test_sysq1_w6_percentile_constant_pinned(qsi_source: str):
    pattern = re.compile(
        r'^SYSQ1_W6_PERCENTILE_VALUE\s*=\s*(?P<value>[0-9.]+)\s*$',
        re.MULTILINE,
    )
    match = pattern.search(qsi_source)
    assert match is not None, (
        "SYS-OPEN-Q-1 W6 mitigation constant missing; see commit 78fffef "
        "and doc/precision_validation.md SYS-OPEN-Q-1 W6 section."
    )
    assert float(match.group("value")) == pytest.approx(99.5), (
        f"W6 sweep identified 99.5 as local optimum across 8 backbone "
        f"nodes (99.0 regresses some layers, 99.99 misses part A). "
        f"Got SYSQ1_W6_PERCENTILE_VALUE={match.group('value')}"
    )


def test_argparse_percentile_default_uses_w6_constant(qsi_source: str):
    pattern = re.compile(
        r'parser\.add_argument\(\s*\n'
        r'\s*"--percentile-value"\s*,\s*\n'
        r'\s*type=float\s*,\s*\n'
        r'\s*default=(?P<default>[A-Za-z0-9_.]+)',
    )
    match = pattern.search(qsi_source)
    assert match is not None, (
        "Failed to locate --percentile-value argparse declaration; "
        "the test pattern must be updated alongside any structural "
        "change to main()."
    )
    assert match.group("default") == "SYSQ1_W6_PERCENTILE_VALUE", (
        f"--percentile-value default is now `{match.group('default')}`. "
        "Expected SYSQ1_W6_PERCENTILE_VALUE so the SYS-OPEN-Q-1 W6 "
        "mitigation is on by default. If this is intentional, update "
        "the constant + this test together."
    )
