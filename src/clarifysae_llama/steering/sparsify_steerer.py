from __future__ import annotations

from typing import Any

import torch
from sparsify import Sae

from clarifysae_llama.discovery.sae_utils import encode_dense
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

        original_dtype = hidden.dtype
        original_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=self.model_device, dtype=self.dtype)

        # Normalize any sparse / tuple SAE encode output into a dense tensor.
        latents = encode_dense(self.sae, hidden_2d).to(device=self.model_device, dtype=self.dtype)

        if self.config.log_feature_acts:
            feature_slice = latents[:, self.config.feature_indices]
            self.last_feature_stats = {
                'mean_abs_activation': float(feature_slice.abs().mean().item()),
                'max_abs_activation': float(feature_slice.abs().max().item()),
            }

        if self.config.mode != 'additive':
            raise ValueError(f'Unsupported steering mode: {self.config.mode}')

        if self.config.apply_to != 'all_positions':
            raise ValueError(f'Unsupported apply_to mode for current repo version: {self.config.apply_to}')

        latents[:, self.config.feature_indices] += self.config.strength

        if self.config.clamp_latents is not None:
            latents = latents.clamp(max=self.config.clamp_latents)

        recon = self.sae.decode(latents).reshape(original_shape)

        if self.config.normalize_reconstruction:
            in_norm = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out_norm = recon.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            recon = recon * (in_norm / out_norm)

        recon = recon.to(original_dtype)

        if self.config.preserve_unsteered_residual:
            delta = recon - hidden
            recon = hidden + delta

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon