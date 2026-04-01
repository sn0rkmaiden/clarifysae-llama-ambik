from __future__ import annotations

from typing import Any

import torch
from huggingface_hub import hf_hub_download
from sparsify import Sae

from clarifysae_llama.discovery.sae_utils import (
    SparseLatents,
    encode_sparse,
    get_num_latents,
    sparse_to_dense,
)
from clarifysae_llama.steering.config import SteeringConfig


def get_submodule_by_path(root_module, path: str):
    current = root_module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def normalize_hookpoint_to_module_path(hookpoint: str) -> str:
    hp = hookpoint.strip()

    if hp == "embed_tokens":
        return "model.embed_tokens"

    if hp == "model.embed_tokens":
        return hp

    # Apart-style / already fully qualified.
    if hp.startswith("model."):
        return hp

    # EleutherAI sparsify-style, e.g. "layers.23.mlp"
    if hp.startswith("layers."):
        return f"model.{hp}"

    raise ValueError(f"Unsupported hookpoint for Llama-style model: {hookpoint}")


def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
    return module_path or normalize_hookpoint_to_module_path(hookpoint)


def infer_module_device(module, fallback: torch.device) -> torch.device:
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return fallback


def move_sae_to_device_dtype(sae: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if hasattr(sae, "to"):
        try:
            sae = sae.to(device=device, dtype=dtype)
        except TypeError:
            sae = sae.to(device=device)
    if hasattr(sae, "eval"):
        sae.eval()
    return sae


def load_sae(
    *,
    loader: str,
    sae_repo: str,
    hookpoint: str,
    sae_file: str | None,
    device: torch.device,
    dtype: torch.dtype,
):
    loader_name = str(loader).strip().lower().replace("-", "_")

    if loader_name == "sparsify":
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        return move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    if loader_name == "dictionary_learning":
        if not sae_file:
            raise ValueError(
                "steering.sae_file is required when steering.loader='dictionary_learning'."
            )
        try:
            from dictionary_learning import utils as dl_utils
        except ImportError as exc:
            raise ImportError(
                "dictionary_learning is required for steering.loader='dictionary_learning'. "
                "Install it with: pip install dictionary-learning"
            ) from exc

        local_path = hf_hub_download(repo_id=sae_repo, filename=sae_file)
        sae, _config = dl_utils.load_dictionary(local_path, device=str(device))
        return move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    raise ValueError(
        f"Unsupported SAE loader '{loader}'. Expected 'sparsify' or 'dictionary_learning'."
    )


class SparsifySteerer:
    def __init__(self, model, model_device: torch.device, dtype: torch.dtype, config: SteeringConfig):
        self.model = model
        self.model_device = model_device
        self.dtype = dtype
        self.config = config
        self.handle = None
        self.last_feature_stats: dict[str, Any] | None = None

        module_path = resolve_module_path(config.hookpoint, config.module_path)
        self.target_module = get_submodule_by_path(self.model, module_path)
        self.sae_device = infer_module_device(self.target_module, fallback=self.model_device)

        self.sae = load_sae(
            loader=config.loader,
            sae_repo=config.sae_repo,
            hookpoint=config.hookpoint,
            sae_file=config.sae_file,
            device=self.sae_device,
            dtype=self.dtype,
        )

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

        if self.config.apply_to == "all_positions":
            mask = torch.ones((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
        elif self.config.apply_to == "last_position":
            mask = torch.zeros((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
            mask[:, -1] = True
        else:
            raise ValueError(f"Unsupported apply_to mode: {self.config.apply_to}")

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

    def _decode_from_sparse(self, sparse: SparseLatents, *, dtype: torch.dtype) -> torch.Tensor:
        try:
            # sparsify-style SAEs
            return self.sae.decode(sparse.top_acts, sparse.top_indices)
        except TypeError:
            # dictionary_learning-style SAEs
            dense = sparse_to_dense(
                sparse,
                num_latents=get_num_latents(self.sae),
                dtype=dtype,
            )
            return self.sae.decode(dense)

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return output

        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape [batch, seq, d_model], got {tuple(hidden.shape)}"
            )

        original_device = hidden.device
        original_dtype = hidden.dtype
        original_shape = hidden.shape

        selected_mask = self._selected_position_mask(hidden)
        if not bool(selected_mask.any()):
            return output

        selected_mask_device = selected_mask.to(device=self.sae_device)
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=self.sae_device, dtype=self.dtype)
        selected_hidden = hidden_2d[selected_mask_device]

        sparse_latents = encode_sparse(self.sae, selected_hidden)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(f"encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents")

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
                self.last_feature_stats = {
                    "mean_abs_activation": 0.0,
                    "max_abs_activation": 0.0,
                }
            else:
                self.last_feature_stats = {
                    "mean_abs_activation": float(selected.abs().mean().item()),
                    "max_abs_activation": float(selected.abs().max().item()),
                }

        if self.config.mode != "additive":
            raise ValueError(f"Unsupported steering mode: {self.config.mode}")

        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)

            hit_mask = steered_top_indices == feature_idx
            if hit_mask.any():
                steered_top_acts[hit_mask] += self.config.strength

            missing_rows = ~hit_mask.any(dim=1)
            if missing_rows.any():
                # Replace the weakest currently-active feature, not the most negative one.
                replacement_col = torch.argmin(steered_top_acts[missing_rows].abs(), dim=1)
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

        steered_recon = self._decode_from_sparse(
            SparseLatents(top_acts=steered_top_acts, top_indices=steered_top_indices),
            dtype=selected_hidden.dtype,
        )

        if self.config.preserve_unsteered_residual:
            base_recon = self._decode_from_sparse(
                SparseLatents(top_acts=base_top_acts, top_indices=base_top_indices),
                dtype=selected_hidden.dtype,
            )
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
        updated_hidden[selected_mask_device] = steered_selected
        recon = updated_hidden.reshape(original_shape).to(device=original_device, dtype=original_dtype)

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon