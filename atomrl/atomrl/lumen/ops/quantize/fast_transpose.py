# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Fast FP8 matrix transpose dispatch.

Dispatches to AITER's Triton tiled transpose kernel which replaces
``tensor.t().contiguous()`` (a full ``aten::copy_`` kernel) with a single
fused kernel launch.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


AITER_FAST_TRANSPOSE = True


def fast_transpose_fp8(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D FP8 tensor via AITER Triton kernel.

    Returns a contiguous (N, M) tensor from a (M, N) input.
    Falls back to ``x.t().contiguous()`` if AITER is unavailable.
    """
    if AITER_FAST_TRANSPOSE:
        from aiter.ops.triton.quant.fast_transpose import fast_transpose_2d

        return fast_transpose_2d(x)

    logger.warning(
        "AITER fast_transpose_2d unavailable, falling back to .t().contiguous()"
    )
    return x.t().contiguous()
