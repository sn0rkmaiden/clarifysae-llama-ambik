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

    def _selected_position_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        if self.config.apply_to == 'all_positions':
            mask = torch.ones((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
        elif self.config.apply_to == 'last_position':
            mask = torch.zeros((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
            mask[:, -1] = True
        else:
            raise ValueError(f'Unsupported apply_to mode for current repo version: {self.config.apply_to}')

        if self.config.steer_generated_tokens_only and seq_len > 1:
            generated_mask = torch.zeros((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
            generated_mask[:, -1] = True
            mask &= generated_mask

        return mask.reshape(-1)

    @staticmethod
    def _normalize_reconstruction(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        recon_norm = reconstruction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return reconstruction * (target_norm / recon_norm)

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

        selected_mask = self._selected_position_mask(hidden)
        if not bool(selected_mask.any()):
            return output

        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=self.model_device, dtype=self.dtype)
        selected_hidden = hidden_2d[selected_mask.to(device=self.model_device)]

        sparse_latents = encode_sparse(self.sae, selected_hidden)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(f'encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents')

        base_top_acts = sparse_latents.top_acts.clone()
        base_top_indices = sparse_latents.top_indices.clone()
        steered_top_acts = base_top_acts.clone()
        steered_top_indices = base_top_indices.clone()

        if self.config.log_feature_acts:
            stats_mask = torch.zeros_like(base_top_acts, dtype=torch.bool)
            for feature_idx in self.config.feature_indices:
                stats_mask |= base_top_indices == int(feature_idx)
            selected = base_top_acts[stats_mask]
            if selected.numel() == 0:
                self.last_feature_stats = {'mean_abs_activation': 0.0, 'max_abs_activation': 0.0}
            else:
                self.last_feature_stats = {
                    'mean_abs_activation': float(selected.abs().mean().item()),
                    'max_abs_activation': float(selected.abs().max().item()),
                }

        if self.config.mode != 'additive':
            raise ValueError(f'Unsupported steering mode: {self.config.mode}')

        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)

            hit_mask = steered_top_indices == feature_idx
            if hit_mask.any():
                steered_top_acts[hit_mask] += self.config.strength

            missing_rows = ~hit_mask.any(dim=1)
            if missing_rows.any():
                replacement_col = torch.argmin(steered_top_acts[missing_rows], dim=1)
                row_idx = torch.arange(replacement_col.shape[0], device=steered_top_acts.device)
                acts_missing = steered_top_acts[missing_rows].clone()
                idx_missing = steered_top_indices[missing_rows].clone()
                idx_missing[row_idx, replacement_col] = feature_idx
                acts_missing[row_idx, replacement_col] = self.config.strength
                steered_top_acts[missing_rows] = acts_missing
                steered_top_indices[missing_rows] = idx_missing

        if self.config.clamp_latents is not None:
            clamp_value = float(self.config.clamp_latents)
            steered_top_acts = steered_top_acts.clamp(min=-clamp_value, max=clamp_value)

        steered_recon = self.sae.decode(steered_top_acts, steered_top_indices)
        if self.config.preserve_unsteered_residual:
            base_recon = self.sae.decode(base_top_acts, base_top_indices)
        else:
            base_recon = None

        if self.config.normalize_reconstruction:
            norm_target = base_recon if base_recon is not None else selected_hidden
            steered_recon = self._normalize_reconstruction(steered_recon, norm_target)
            if base_recon is not None:
                base_recon = self._normalize_reconstruction(base_recon, selected_hidden)

        if self.config.preserve_unsteered_residual:
            assert base_recon is not None
            steered_selected = selected_hidden + (steered_recon - base_recon)
        else:
            steered_selected = steered_recon

        updated_hidden = hidden_2d.clone()
        updated_hidden[selected_mask.to(device=self.model_device)] = steered_selected
        recon = updated_hidden.reshape(original_shape).to(device=original_device, dtype=original_dtype)

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon
