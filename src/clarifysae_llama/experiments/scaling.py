from __future__ import annotations

import math


def compute_relative_residual_scale(*, residual_l2: float, decoder_l2: float) -> float:
    """Return the latent multiplier that makes alpha=1 match residual_l2 in L2 norm."""
    residual_l2 = float(residual_l2)
    decoder_l2 = float(decoder_l2)
    if not math.isfinite(residual_l2) or residual_l2 <= 0:
        raise ValueError(f"residual_l2 must be finite and positive, got {residual_l2}")
    if not math.isfinite(decoder_l2) or decoder_l2 <= 0:
        raise ValueError(f"decoder_l2 must be finite and positive, got {decoder_l2}")
    return residual_l2 / decoder_l2
