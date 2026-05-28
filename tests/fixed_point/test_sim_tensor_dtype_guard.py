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
"""Static guard: forbid PR-3 regressions in the int32 sim-tensor container path.

This module performs source-level (grep-style) checks on the
``aimet_torch/fixed_point/kernels/`` package to make sure none of the patterns
that PR-3 explicitly removed creep back in via a future change:

* ``F.unfold(...)``           - pulls an integer tensor through a CPU/CUDA kernel
                                that ``torch`` does not provide for ``int32``;
                                callers must use :func:`im2col_int` instead.
* ``saturate_int16(...)`` calls - replaced by :func:`saturate_sim_tensor` so
                                  the carrier dtype is decided in one place
                                  (``SIM_TENSOR_DTYPE``).
* ``int_repr=...dtype=torch.int16``  in the kernels package - the carrier
                                                              dtype is
                                                              ``SIM_TENSOR_DTYPE``;
                                                              callers must
                                                              upcast before
                                                              constructing the
                                                              tensor.
* ``.to(torch.int16)`` on an ``int_repr`` slot - same reason as above.

We deliberately keep two intentional exceptions:

* ``q_b`` / ``lut`` table entries inside ``kernels/lut.py`` are
  ``torch.int16`` - that is the *LUT entry* width (hardware contract,
  ADR-014), not a carrier slot.
* The string ``F.unfold`` appearing in *docstrings* of ``kernels/_im2col.py``
  is allowed; those are references to the API we replaced.

If a future change actually needs to break one of these rules, update the
ADR (013/014) first, then this guard.
"""

from __future__ import annotations

import re
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parents[2] / "aimet_torch" / "fixed_point" / "kernels"
REQUANTIZE_PY = (
    Path(__file__).resolve().parents[2]
    / "aimet_torch"
    / "fixed_point"
    / "requantize.py"
)


def _iter_kernel_sources():
    for path in sorted(KERNELS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path, path.read_text(encoding="utf-8")


def _strip_strings_and_comments(source: str) -> str:
    """Return ``source`` with string literals and ``#`` comments blanked out.

    Sufficient to avoid false positives from docstrings / inline notes that
    only *mention* the forbidden pattern. We do not need a real Python lexer
    here; ``ast`` would be overkill and slower.
    """

    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "#":
            # Drop until end of line.
            j = source.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        if ch in ('"', "'"):
            # Detect triple-quoted strings; otherwise single-line strings.
            triple = source[i : i + 3]
            if triple == ch * 3:
                end = source.find(triple, i + 3)
                if end == -1:
                    break
                i = end + 3
                continue
            j = i + 1
            while j < n and source[j] != ch:
                if source[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    j += 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def test_no_F_unfold_calls_in_kernels():
    """``F.unfold(...)`` has no integer kernel; PR-3 banned it for sim path."""

    pattern = re.compile(r"\bF\s*\.\s*unfold\s*\(")
    offenders = []
    for path, source in _iter_kernel_sources():
        code = _strip_strings_and_comments(source)
        if pattern.search(code):
            offenders.append(path.name)
    assert not offenders, (
        "F.unfold call(s) found in kernels: "
        f"{offenders}. Use im2col_int from kernels._im2col instead."
    )


def test_no_saturate_int16_calls_in_kernels():
    """PR-3 standardised on ``saturate_sim_tensor`` for the carrier dtype."""

    # Match a call ``saturate_int16(`` but NOT the alias *definition* line.
    pattern = re.compile(r"(?<!def\s)\bsaturate_int16\s*\(")
    offenders = []
    for path, source in _iter_kernel_sources():
        code = _strip_strings_and_comments(source)
        if pattern.search(code):
            offenders.append(path.name)
    assert not offenders, (
        "saturate_int16(...) call(s) found in kernels: "
        f"{offenders}. Use saturate_sim_tensor from requantize.py instead."
    )


def test_no_int_repr_downcast_to_int16_in_kernels():
    """``int_repr.to(torch.int16)`` would silently truncate the int32 carrier."""

    pattern = re.compile(r"int_repr\s*\.\s*to\s*\(\s*torch\s*\.\s*int16")
    offenders = []
    for path, source in _iter_kernel_sources():
        code = _strip_strings_and_comments(source)
        if pattern.search(code):
            offenders.append(path.name)
    assert not offenders, (
        "int_repr.to(torch.int16) found in kernels: "
        f"{offenders}. Carrier dtype must stay SIM_TENSOR_DTYPE."
    )


def test_no_carrier_constructed_with_int16_in_kernels():
    """Constructing a carrier with ``int_repr=...dtype=torch.int16`` bypasses ADR-013."""

    # Same-line ``int_repr=...dtype=torch.int16`` (kernels build their carriers
    # inline; multi-line offenders would be flagged by the runtime guard in
    # ``FixedPointSimTensor.__post_init__``).
    pattern = re.compile(
        r"int_repr\s*=.*dtype\s*=\s*torch\s*\.\s*int16",
        re.MULTILINE,
    )
    offenders = []
    for path, source in _iter_kernel_sources():
        code = _strip_strings_and_comments(source)
        if pattern.search(code):
            offenders.append(path.name)
    assert not offenders, (
        "Carrier built with int16 int_repr in kernels: "
        f"{offenders}. Use SIM_TENSOR_DTYPE."
    )


def test_saturate_int16_alias_still_exists_for_back_compat():
    """The alias is kept for one release so downstream consumers can migrate."""

    source = REQUANTIZE_PY.read_text(encoding="utf-8")
    assert "def saturate_int16(" in source, (
        "saturate_int16 alias removed; downstream callers may break. Stage "
        "removal through ADR-013 deprecation window first."
    )
