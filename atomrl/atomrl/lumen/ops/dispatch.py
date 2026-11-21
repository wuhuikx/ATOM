# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Backend dispatcher with automatic ASM → Triton fallback.

All backends are AITER implementations. No torch.nn.functional fallbacks.
Each operator registers its available backends via :func:`try_backends`.
On each call the dispatcher walks the priority chain and returns the
first successful result, logging fallbacks as warnings.
"""

import logging
import os
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import torch

_SKIP_BACKEND_SYNC = os.environ.get("LUMEN_SKIP_BACKEND_SYNC", "0") == "1"

# op_name -> winning backend label; also op_name + ":hits" / ":prev" / ":warned".
_backend_cache: dict[str, Any] = {}
_BACKEND_WARMUP_CALLS = 3

_IN_GRAPH_CAPTURE = False


def set_graph_capture_mode(active: bool):
    """Toggle graph-capture flag for dispatch safety."""
    global _IN_GRAPH_CAPTURE
    _IN_GRAPH_CAPTURE = active


try:
    from triton.compiler.errors import CompilationError as _TritonCompilationError
except ImportError:
    _TritonCompilationError = None

try:
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources
except ImportError:
    _TritonOutOfResources = None

logger = logging.getLogger(__name__)


class Backend(Enum):
    ASM = "asm"
    TRITON = "triton"
    HIPBLAS = "hipblas"


FALLBACK_ORDER = [Backend.ASM, Backend.TRITON]


# ---------------------------------------------------------------------------
# AITER is pinned by the release image. These symbols are part of the
# runtime contract; call sites invoke the kernels directly instead of
# importing and probing every optional submodule on every process.

AITER_TRITON_NORM = True
AITER_TRITON_RMSNORM = True
AITER_TRITON_GEMM = True
AITER_TRITON_GEMM_BF16 = True
AITER_TUNED_GEMM_BF16 = True
AITER_ASM_NORM = True
AITER_QUANT = True
AITER_TRITON_QUANT = True
AITER_HIPBLAS = True
AITER_FUSED_QUANT = True
AITER_FUSED_SILU_MUL_FP8 = True
AITER_GMM = True
AITER_DEEPGEMM = True
AITER_TRITON_GEMM_MXFP8 = True
AITER_TRITON_GEMM_MXFP4 = True
AITER_TRITON_GEMM_MXFP4_PRESHUFFLE = True
AITER_GEMM_MXFP4_ASM = True
AITER_TRITON_ROPE_CACHED = True
AITER_TRITON_ROPE_CACHED_2C = True
AITER_TRITON_ROPE_2D = True
AITER_TRITON_ROPE_3D = True
AITER_MOE_TOPK_SOFTMAX = True
AITER_SOFTMAX_TOPK = True
AITER_TRITON_MOE_AUX_LOSS = True
AITER_MOE_SORTING = True
AITER_MOE_SUM = True
AITER_MOE_GEMM_PER_TOKEN = True
AITER_MOE_GEMM_MXFP8 = True
AITER_TRITON_MOE_ALIGN = True
AITER_TRITON_FUSED_MOE = True
AITER_FUSED_GATED = True
AITER_FUSED_UNGATED = True
AITER_SWIGLU = True
AITER_FAST_TRANSPOSE = True
AITER_FUSED_ADD_RMSNORM_PAD = True
AITER_FUSED_GEMM_BLOCKSCALE_MUL_ADD = True


# Core fallback dispatcher
# ---------------------------------------------------------------------------


def _entry_label(entry: tuple) -> str:
    """Stable identity for one fallback-chain entry.

    ``Backend`` on its own is not unique: ``gemm_mxfp4`` offers three
    ``Backend.TRITON`` entries (preshuffled, row-major, and the dequant→BF16
    fallback), so a chain that reuses a ``Backend`` passes an explicit label as
    a third tuple element.
    """
    if len(entry) >= 3:
        return entry[2]
    return entry[0].value


def try_backends(
    backends: list[tuple[Backend, Callable]],
    *args,
    op_name: str = "op",
    slow_labels: Sequence[str] = (),
    **kwargs,
) -> Any:
    """Try each ``(backend, fn)`` pair in order; return first success.

    On ``RuntimeError``, ``NotImplementedError``, ``TypeError``,
    ``ValueError``, ``IndexError``, ``KeyError``, or Triton
    ``CompilationError`` from a backend, logs a warning and falls
    through to the next.  ``IndexError`` / ``KeyError`` are included
    because AITER JIT wrappers raise ``map::at`` when a kernel config
    lookup fails.  If all fail, raises the last exception.

    After a backend succeeds ``_BACKEND_WARMUP_CALLS`` consecutive times
    for a given ``op_name``, the winning backend is cached by *label* and
    subsequent calls skip the fallback chain entirely.  Caching the label
    rather than the list position matters for any op that rebuilds its chain
    per call: a position means a different backend once the order or length
    changes, so an index cached from one shape can hand a later shape the
    fallback -- or a kernel that shape is not allowed to use.  A label that is
    no longer on offer simply misses and re-runs the chain.

    Labels named in ``slow_labels`` are reported at ``warning`` rather than
    ``debug`` when they win, because locking onto a degraded path is not
    something a run should have to read debug logs to discover.

    ``torch.cuda.synchronize()`` is issued during warmup for error
    detection.  After warmup (or when ``LUMEN_SKIP_BACKEND_SYNC=1``),
    sync is skipped to reduce host-device round-trip overhead.
    """
    _catchable = (
        RuntimeError,
        NotImplementedError,
        TypeError,
        ValueError,
        IndexError,
        KeyError,
    )
    if _TritonCompilationError is not None:
        _catchable = _catchable + (_TritonCompilationError,)
    if _TritonOutOfResources is not None:
        _catchable = _catchable + (_TritonOutOfResources,)

    cached_label = _backend_cache.get(op_name)
    if cached_label is not None:
        for entry in backends:
            if _entry_label(entry) == cached_label:
                try:
                    return entry[1](*args, **kwargs)
                except _catchable as exc:
                    # The lock is a warmup shortcut, not a promise that these
                    # operands still suit the winner. Forget it and re-run the
                    # chain rather than propagating, so the degradation the
                    # chain exists to provide survives warmup.
                    logger.warning(
                        "%s: cached %s backend failed (%s), re-running the fallback chain",
                        op_name,
                        cached_label,
                        exc,
                    )
                    for suffix in ("", ":hits", ":prev"):
                        _backend_cache.pop(op_name + suffix, None)
                break

    if _IN_GRAPH_CAPTURE:
        raise RuntimeError(
            f"{op_name}: no cached backend during CUDA graph capture. "
            "Run warmup before graph capture."
        )

    last_exc = None
    for entry in backends:
        backend, fn = entry[0], entry[1]
        label = _entry_label(entry)
        try:
            result = fn(*args, **kwargs)
            if (
                torch.cuda.is_available()
                and not _SKIP_BACKEND_SYNC
                and not _IN_GRAPH_CAPTURE
            ):
                torch.cuda.synchronize()

            # Count consecutive wins for *this* backend. Reading the running
            # count before comparing would let a switch of backend inherit the
            # previous one's streak and lock on its first success.
            if _backend_cache.get(op_name + ":prev") == label:
                hit_count = _backend_cache.get(op_name + ":hits", 0) + 1
            else:
                hit_count = 1
            _backend_cache[op_name + ":hits"] = hit_count
            _backend_cache[op_name + ":prev"] = label

            if hit_count >= _BACKEND_WARMUP_CALLS:
                _backend_cache[op_name] = label
                if label in slow_labels and not _backend_cache.get(op_name + ":warned"):
                    _backend_cache[op_name + ":warned"] = True
                    logger.warning(
                        "%s: locked to the %s fallback after %d successes -- every later "
                        "call with this operand shape takes it. The faster backends were "
                        "tried first and failed; check the warnings above.",
                        op_name,
                        label,
                        hit_count,
                    )
                else:
                    logger.debug(
                        "%s: locked to %s backend (%s) after %d successes",
                        op_name,
                        backend.value,
                        label,
                        hit_count,
                    )

            return result
        except _catchable as exc:
            logger.warning(
                "%s: %s backend failed (%s), trying next...",
                op_name,
                backend.value,
                exc,
            )
            # Deliberately *not* resetting the streak here. The winner's count
            # has to survive a failure earlier in the chain, or a backend that
            # is never first can never lock -- which left every call paying the
            # cost of a kernel known to reject these operands, plus a warning.
            last_exc = exc
    raise RuntimeError(
        f"{op_name}: all AITER backends exhausted. Last error: {last_exc}"
    ) from last_exc


def build_fallback_chain(
    candidates: dict[Backend, Callable | None],
    order: list[Backend] = FALLBACK_ORDER,
) -> list[tuple[Backend, Callable]]:
    """Build an ordered list of (backend, callable) from candidates.

    ``None`` values (unavailable backends) are skipped.
    """
    chain = []
    for b in order:
        fn = candidates.get(b)
        if fn is not None:
            chain.append((b, fn))
    return chain
