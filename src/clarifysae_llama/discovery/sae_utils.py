from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SparseLatents:
    top_acts: torch.Tensor
    top_indices: torch.Tensor


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value)!r}")
    return value


def get_num_latents(sae: Any) -> int:
    for attr in ('num_latents', 'd_sae', 'num_features'):
        if hasattr(sae, attr):
            return int(getattr(sae, attr))
    if hasattr(sae, 'W_dec') and torch.is_tensor(sae.W_dec):
        return int(sae.W_dec.shape[0])
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight') and torch.is_tensor(sae.decoder.weight):
        return int(sae.decoder.weight.shape[0])
    raise AttributeError('Could not determine SAE latent dimension.')


def get_decoder_matrix(sae: Any) -> torch.Tensor:
    if hasattr(sae, 'W_dec') and torch.is_tensor(sae.W_dec):
        return sae.W_dec
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight') and torch.is_tensor(sae.decoder.weight):
        return sae.decoder.weight
    raise AttributeError('Could not find SAE decoder matrix (expected W_dec or decoder.weight).')


def _from_tuple(encoded: tuple[Any, ...]) -> SparseLatents | torch.Tensor:
    if len(encoded) >= 2 and torch.is_tensor(encoded[0]) and torch.is_tensor(encoded[1]):
        first, second = encoded[0], encoded[1]
        if second.dtype in (torch.int16, torch.int32, torch.int64, torch.long):
            return SparseLatents(top_acts=first, top_indices=second)
        if first.shape == second.shape:
            return first
    if len(encoded) >= 1 and torch.is_tensor(encoded[0]):
        return encoded[0]
    raise TypeError('Unsupported tuple output from sae.encode(...).')


def normalize_encoded(encoded: Any) -> SparseLatents | torch.Tensor:
    if torch.is_tensor(encoded):
        return encoded
    if isinstance(encoded, tuple):
        return _from_tuple(encoded)
    if hasattr(encoded, 'top_acts') and hasattr(encoded, 'top_indices'):
        return SparseLatents(
            top_acts=_as_tensor(encoded.top_acts, 'encoded.top_acts'),
            top_indices=_as_tensor(encoded.top_indices, 'encoded.top_indices'),
        )
    if hasattr(encoded, 'latents') and torch.is_tensor(encoded.latents):
        return encoded.latents
    raise TypeError(f'Unsupported output type from sae.encode(...): {type(encoded)!r}')


def sparse_to_dense(sparse: SparseLatents, num_latents: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    acts = sparse.top_acts
    indices = sparse.top_indices
    if acts.shape != indices.shape:
        raise ValueError('Sparse SAE activations must have matching shapes for acts and indices.')
    dense = torch.zeros(*acts.shape[:-1], num_latents, device=acts.device, dtype=dtype or acts.dtype)
    dense.scatter_add_(-1, indices.long(), acts.to(dtype or acts.dtype))
    return dense


def encode_dense(sae: Any, hidden: torch.Tensor) -> torch.Tensor:
    normalized = normalize_encoded(sae.encode(hidden))
    if torch.is_tensor(normalized):
        return normalized
    return sparse_to_dense(normalized, num_latents=get_num_latents(sae), dtype=hidden.dtype)


def encode_sparse(sae: Any, hidden: torch.Tensor) -> SparseLatents:
    normalized = normalize_encoded(sae.encode(hidden))
    if isinstance(normalized, SparseLatents):
        return normalized
    dense = normalized
    if dense.ndim < 2:
        raise ValueError('Dense SAE latents must have at least 2 dimensions to derive sparse representation.')
    nonzero = dense != 0
    k = int(nonzero.sum(dim=-1).max().item())
    if k <= 0:
        k = min(1, dense.shape[-1])
    top_acts, top_indices = torch.topk(dense, k=k, dim=-1)
    return SparseLatents(top_acts=top_acts, top_indices=top_indices)


def compute_a_max_from_sparse(
    sparse: SparseLatents,
    num_latents: int,
    current: torch.Tensor,
) -> torch.Tensor:
    acts = sparse.top_acts.reshape(-1).to(current.device, dtype=current.dtype)
    indices = sparse.top_indices.reshape(-1).to(current.device)
    if acts.numel() == 0:
        return current
    if hasattr(current, 'scatter_reduce_'):
        current.scatter_reduce_(0, indices.long(), acts, reduce='amax', include_self=True)
        return current

    # Fallback for older torch versions.
    for idx, value in zip(indices.tolist(), acts.tolist()):
        if value > current[idx].item():
            current[idx] = value
    return current


def decode_from_sparse(sae: Any, sparse: SparseLatents, dtype: torch.dtype | None = None) -> torch.Tensor:
    try:
        # sparsify-style decode(top_acts, top_indices)
        return sae.decode(sparse.top_acts, sparse.top_indices)
    except TypeError:
        # dictionary_learning-style decode(dense_latents)
        dense = sparse_to_dense(
            sparse,
            num_latents=get_num_latents(sae),
            dtype=dtype or sparse.top_acts.dtype,
        )
        return sae.decode(dense)
