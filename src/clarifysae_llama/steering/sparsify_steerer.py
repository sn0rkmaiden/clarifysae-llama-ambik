from __future__ import annotations

from typing import Any

import torch
from sparsify import Sae

from clarifysae_llama.discovery.sae_utils import SparseLatents, encode_sparse
from clarifysae_llama.steering.config import SteeringConfig
from clarifysae_llama.steering.hook_utils import get_submodule_by_path, map_sae_hookpoint_to_hf_module_path


class SparsifySteerer:
    def __init__(self, model, model_device: torch.device, dtype: torch.dtype, config: SteeringConfig):
        self.model = model
        self.model_device = model_device
        self.dtype = dtype
        self.config = config
        self.handle = None
        self.last_feature_stats: dict[str, Any] | None = None

        self.sae = Sae.load_from_hub(config.sae_repo, hookpoint=config.hookpoint)
        self.sae = self.sae.to(device=self.model_device, dtype=self.dtype)
        self.sae.eval()

        module_path = map_sae_hookpoint_to_hf_module_path(config.hookpoint)
        self.target_module = get_submodule_by_path(self.model, module_path)

    def attach(self) -> None:
        if self.handle is None:
            self.handle = self.target_module.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self) -> None:
        self.last_feature_stats = None

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return output
        if hidden.ndim != 3:
            raise ValueError(f'Expected hidden states with shape [batch, seq, d_model], got {tuple(hidden.shape)}')

        original_device = hidden.device
        original_dtype = hidden.dtype
        original_shape = hidden.shape

        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(
            device=self.model_device,
            dtype=self.dtype,
        )

        sparse_latents = encode_sparse(self.sae, hidden_2d)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(
                f"encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents"
            )

        top_acts = sparse_latents.top_acts.clone()
        top_indices = sparse_latents.top_indices.clone()

        if self.config.log_feature_acts:
            stats_mask = torch.zeros_like(top_acts, dtype=torch.bool)
            for feature_idx in self.config.feature_indices:
                stats_mask |= (top_indices == int(feature_idx))
            selected = top_acts[stats_mask]
            if selected.numel() == 0:
                self.last_feature_stats = {
                    'mean_abs_activation': 0.0,
                    'max_abs_activation': 0.0,
                }
            else:
                self.last_feature_stats = {
                    'mean_abs_activation': float(selected.abs().mean().item()),
                    'max_abs_activation': float(selected.abs().max().item()),
                }

        if self.config.mode != 'additive':
            raise ValueError(f'Unsupported steering mode: {self.config.mode}')

        if self.config.apply_to != 'all_positions':
            raise ValueError(f'Unsupported apply_to mode for current repo version: {self.config.apply_to}')

        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)

            hit_mask = top_indices == feature_idx
            if hit_mask.any():
                top_acts[hit_mask] += self.config.strength

            missing_rows = ~hit_mask.any(dim=1)
            if missing_rows.any():
                replacement_col = torch.argmin(top_acts[missing_rows], dim=1)
                row_idx = torch.arange(replacement_col.shape[0], device=top_acts.device)

                acts_missing = top_acts[missing_rows].clone()
                idx_missing = top_indices[missing_rows].clone()

                idx_missing[row_idx, replacement_col] = feature_idx
                acts_missing[row_idx, replacement_col] = self.config.strength

                top_acts[missing_rows] = acts_missing
                top_indices[missing_rows] = idx_missing

        if self.config.clamp_latents is not None:
            top_acts = top_acts.clamp(max=self.config.clamp_latents)

        recon = self.sae.decode(top_acts, top_indices)
        recon = recon.reshape(original_shape).to(
            device=original_device,
            dtype=original_dtype,
        )

        if self.config.normalize_reconstruction:
            in_norm = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out_norm = recon.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            recon = recon * (in_norm / out_norm)

        if self.config.preserve_unsteered_residual:
            delta = recon - hidden
            recon = hidden + delta

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon