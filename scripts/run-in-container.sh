#!/usr/bin/env bash
# =============================================================================
# Run a command inside the quant-gru-cuda128 container with the host
# workspace + quant_gru module mounted on PYTHONPATH and the host user's
# pip --user site-packages exposed (pytest, etc.).
#
# Why this exists:
#   * MRNN tests / examples / quick_start_int16_metric.py need
#     ``torchaudio + quant_gru + CUDA``.
#   * The host venv ``/home/llq/Project/AI_X/AIX/.venv`` has neither
#     torchaudio nor quant_gru and no GPU access; the
#     ``quant-gru-cuda128`` container is the canonical environment
#     (matched torch nightly, CUDA, GPU drivers, torchaudio).
#   * The container's default user is uid 1005 which does not own the
#     workspace files, so we ``-u root`` and let the container see the
#     host home via the ``/home → /home`` bind mount.
#
# Usage:
#   scripts/run-in-container.sh pytest tests/fixed_point/test_mrnn_int16_e2e.py -v
#   scripts/run-in-container.sh python examples/quick_start_int16_metric.py --max-eval-batches 4
#   scripts/run-in-container.sh -- bash                    # interactive
#   CONTAINER=other-cuda-container scripts/run-in-container.sh pytest ...
#
# Two callable shortcuts (positional verb):
#   pytest <args>                   → /home/llq/.local/bin/pytest <args>
#   python <args>                   → /usr/bin/python3 <args>
#
# Env knobs:
#   CONTAINER       container name (default: quant-gru-cuda128)
#   WORKSPACE       host workspace path (default: /home/llq/workspace/aimet_rx-main)
#   QUANT_GRU_PATH  quant_gru module path on host (default:
#                   /home/llq/workspace/quant-gru-pytorch/pytorch)
#   USER_BASE       host user pip --user base (default: /home/llq/.local)
#   EXTRA_PYPATH    extra paths appended to PYTHONPATH (colon-separated)
#   DOCKER_FLAGS    extra flags passed to docker exec (e.g. ``-it`` for tty)
# =============================================================================
set -euo pipefail

CONTAINER="${CONTAINER:-quant-gru-cuda128}"
WORKSPACE="${WORKSPACE:-/home/llq/workspace/aimet_rx-main}"
QUANT_GRU_PATH="${QUANT_GRU_PATH:-/home/llq/workspace/quant-gru-pytorch/pytorch}"
USER_BASE="${USER_BASE:-/home/llq/.local}"
EXTRA_PYPATH="${EXTRA_PYPATH:-}"

# Build PYTHONPATH = quant_gru + workspace + EXTRA. Avoid leading ":"
# when EXTRA_PYPATH is empty.
PYPATH="${QUANT_GRU_PATH}:${WORKSPACE}"
if [[ -n "${EXTRA_PYPATH}" ]]; then
    PYPATH="${PYPATH}:${EXTRA_PYPATH}"
fi

# Make sure the container is up before we exec into it. Fail loudly with
# the suggestion to ``docker start`` if it isn't.
if ! docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -q true; then
    echo "ERROR: container '${CONTAINER}' is not running." >&2
    echo "       Start it with: docker start ${CONTAINER}" >&2
    exit 1
fi

# Verb shortcut: ``pytest`` / ``python`` resolve to the well-known
# binaries inside the container; anything else is passed through.
verb="${1:-bash}"
case "${verb}" in
    pytest)
        shift
        set -- /home/llq/.local/bin/pytest "$@"
        ;;
    python)
        shift
        set -- /usr/bin/python3 "$@"
        ;;
    --)
        shift
        ;;
esac

# shellcheck disable=SC2086  # DOCKER_FLAGS deliberately splits on whitespace
exec docker exec ${DOCKER_FLAGS:-} -u root -w "${WORKSPACE}" \
    -e PYTHONPATH="${PYPATH}" \
    -e PYTHONUSERBASE="${USER_BASE}" \
    "${CONTAINER}" "$@"
