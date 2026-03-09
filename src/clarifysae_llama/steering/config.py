from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SteeringConfig:
    sae_repo: str
    hookpoint: str
    feature_indices: list[int]
    strength: float
    mode: str = 'additive'
    apply_to: str = 'all_positions'
    steer_generated_tokens_only: bool = False
    normalize_reconstruction: bool = False
    preserve_unsteered_residual: bool = False
    clamp_latents: Optional[float] = None
    log_feature_acts: bool = False
