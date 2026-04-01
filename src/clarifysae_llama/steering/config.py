from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SteeringConfig:
    sae_repo: str
    hookpoint: str
    feature_indices: list[int]
    strength: float

    # New fields for loader / repo compatibility.
    loader: str = "sparsify"  # "sparsify" or "dictionary_learning"
    sae_file: Optional[str] = None
    module_path: Optional[str] = None

    # Existing steering controls.
    mode: str = "additive"
    apply_to: str = "all_positions"
    steer_generated_tokens_only: bool = False
    normalize_reconstruction: bool = False
    preserve_unsteered_residual: bool = False
    clamp_latents: Optional[float] = None
    log_feature_acts: bool = False