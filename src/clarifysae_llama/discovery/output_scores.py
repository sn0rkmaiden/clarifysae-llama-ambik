from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from clarifysae_llama.discovery.dataset import load_token_chunks
from clarifysae_llama.discovery.sae_utils import (
    SparseLatents,
    compute_a_max_from_sparse,
    encode_sparse,
    get_decoder_matrix,
    get_num_latents,
    sparse_to_dense,
)
from clarifysae_llama.utils.io import ensure_dir


@dataclass
class OutputScoreResult:
    feature_idx: int
    top_token_ids: list[int]
    top_tokens: list[str]
    prob: float
    rank: int
    output_score: float
    prompt: str
    steering_scale: float


def _decode_from_sparse(sae, sparse: SparseLatents, *, dtype: torch.dtype) -> torch.Tensor:
    try:
        # sparsify-style
        return sae.decode(sparse.top_acts, sparse.top_indices)
    except TypeError:
        # dictionary_learning-style
        dense = sparse_to_dense(
            sparse,
            num_latents=get_num_latents(sae),
            dtype=dtype,
        )
        return sae.decode(dense)


class SingleFeatureIntervention:
    def __init__(
        self,
        target_module,
        sae,
        feature_idx: int,
        steering_delta: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.target_module = target_module
        self.sae = sae
        self.feature_idx = int(feature_idx)
        self.steering_delta = float(steering_delta)
        self.dtype = dtype
        self.device = device
        self._handle = None

    def __enter__(self):
        self._handle = self.target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

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

        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(
            device=self.device,
            dtype=self.dtype,
        )

        sparse_latents = encode_sparse(self.sae, hidden_2d)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(
                f"encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents"
            )

        top_acts = sparse_latents.top_acts.clone()
        top_indices = sparse_latents.top_indices.clone()

        hit_mask = top_indices == self.feature_idx
        if hit_mask.any():
            top_acts[hit_mask] += self.steering_delta

        missing_rows = ~hit_mask.any(dim=1)
        if missing_rows.any():
            replacement_col = torch.argmin(top_acts[missing_rows].abs(), dim=1)
            row_idx = torch.arange(replacement_col.shape[0], device=top_acts.device)

            acts_missing = top_acts[missing_rows].clone()
            idx_missing = top_indices[missing_rows].clone()

            idx_missing[row_idx, replacement_col] = self.feature_idx
            acts_missing[row_idx, replacement_col] = self.steering_delta

            top_indices[missing_rows] = idx_missing
            top_acts[missing_rows] = acts_missing

        recon = _decode_from_sparse(
            self.sae,
            SparseLatents(top_acts=top_acts, top_indices=top_indices),
            dtype=hidden_2d.dtype,
        )
        recon = recon.reshape(original_shape).to(
            device=original_device,
            dtype=original_dtype,
        )

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon


def compute_a_max(
    model,
    tokenizer,
    sae,
    target_module,
    dataset_cfg: dict[str, Any],
    tokenization_cfg: dict[str, Any],
    batch_size: int,
    dtype: torch.dtype,
    sae_device: torch.device,
    model_input_device: torch.device,
) -> torch.Tensor:
    from clarifysae_llama.runners.discover_features import HiddenActivationExtractor

    token_chunks = load_token_chunks(
        dataset_cfg=dataset_cfg,
        tokenizer=tokenizer,
        tokenization_cfg=tokenization_cfg,
    )

    a_max = torch.zeros(get_num_latents(sae), device=sae_device, dtype=torch.float32)
    extractor = HiddenActivationExtractor(model, target_module=target_module)

    with extractor:
        for start in tqdm(range(0, len(token_chunks), batch_size), desc="Computing a_max"):
            batch_tokens = token_chunks[start : start + batch_size]
            padded = pad_sequence(
                batch_tokens,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            attention_mask = (padded != tokenizer.pad_token_id).long()

            model_inputs = {
                "input_ids": padded.to(model_input_device),
                "attention_mask": attention_mask.to(model_input_device),
            }

            with torch.inference_mode():
                _ = model(**model_inputs, use_cache=False)
                hidden = extractor.pop()
                hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(
                    device=sae_device,
                    dtype=dtype,
                )
                sparse_latents = encode_sparse(sae, hidden_2d)
                a_max = compute_a_max_from_sparse(
                    sparse_latents,
                    get_num_latents(sae),
                    a_max,
                )

    return a_max.detach().cpu()


def compute_top_tokens_for_features(
    model,
    sae,
    tokenizer,
    feature_ids: list[int],
    top_k_tokens: int,
) -> dict[int, tuple[list[int], list[str]]]:
    decoder = get_decoder_matrix(sae)
    if decoder.ndim != 2:
        raise ValueError(
            f"Expected SAE decoder matrix to have shape [n_features, d_model], got {tuple(decoder.shape)}"
        )

    lm_head = model.lm_head.weight.detach()
    lm_head_device = lm_head.device
    lm_head_dtype = lm_head.dtype

    final_norm = getattr(model.model, "norm", None)
    results: dict[int, tuple[list[int], list[str]]] = {}

    with torch.inference_mode():
        for feature_idx in feature_ids:
            vec = decoder[int(feature_idx)].detach()
            vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)

            if final_norm is not None:
                try:
                    norm_param = next(final_norm.parameters())
                    norm_device = norm_param.device
                    norm_dtype = norm_param.dtype

                    vec = vec.to(device=norm_device, dtype=norm_dtype)
                    vec = final_norm(vec.unsqueeze(0)).squeeze(0)
                    vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)
                except Exception:
                    vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)

            scores = lm_head @ vec
            top_ids = torch.topk(scores, k=int(top_k_tokens)).indices.tolist()
            top_tokens = [tokenizer.decode([token_id]) for token_id in top_ids]
            results[int(feature_idx)] = (top_ids, top_tokens)

    return results


def compute_output_scores(
    model,
    tokenizer,
    sae,
    target_module,
    feature_ids: list[int],
    a_max: torch.Tensor,
    prompt: str,
    steering_strength: float,
    top_k_tokens: int,
    dtype: torch.dtype,
    sae_device: torch.device,
    model_input_device: torch.device,
) -> list[OutputScoreResult]:
    top_tokens_map = compute_top_tokens_for_features(
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        feature_ids=feature_ids,
        top_k_tokens=top_k_tokens,
    )

    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_inputs = {k: v.to(model_input_device) for k, v in prompt_inputs.items()}

    results: list[OutputScoreResult] = []
    vocab_size = int(model.lm_head.weight.shape[0])

    for feature_idx in tqdm(feature_ids, desc="Computing output scores"):
        steering_delta = float(steering_strength) * float(a_max[int(feature_idx)].item())

        with SingleFeatureIntervention(
            target_module=target_module,
            sae=sae,
            feature_idx=int(feature_idx),
            steering_delta=steering_delta,
            dtype=dtype,
            device=sae_device,
        ):
            with torch.inference_mode():
                logits = model(**prompt_inputs, use_cache=False).logits[0, -1]
                probs = torch.softmax(logits.float(), dim=-1).detach().cpu()

        top_ids, top_tokens = top_tokens_map[int(feature_idx)]
        target_token = int(top_ids[0])

        prob = float(probs[target_token].item())
        rank = int((probs > prob).sum().item()) + 1
        output_score = prob * (1.0 - (rank / vocab_size))

        results.append(
            OutputScoreResult(
                feature_idx=int(feature_idx),
                top_token_ids=top_ids,
                top_tokens=top_tokens,
                prob=prob,
                rank=rank,
                output_score=float(output_score),
                prompt=prompt,
                steering_scale=steering_delta,
            )
        )

    return results


def save_output_score_results(
    output_dir: str | Path,
    feature_scores_path: str | Path,
    a_max: torch.Tensor,
    results: list[OutputScoreResult],
    config: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)
    a_max_path = output_dir / "a_max.pt"
    torch.save(a_max, a_max_path)

    rows = [
        {
            "feature_idx": result.feature_idx,
            "output_score": result.output_score,
            "prob": result.prob,
            "rank": result.rank,
            "steering_scale": result.steering_scale,
            "top_token_ids": result.top_token_ids,
            "top_tokens": result.top_tokens,
            "prompt": result.prompt,
        }
        for result in results
    ]

    df = pd.DataFrame(rows).sort_values("output_score", ascending=False)
    df.to_csv(output_dir / "output_scores.csv", index=False)
    (output_dir / "output_scores.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    torch.save({"rows": rows}, output_dir / "output_scores.pt")

    config_payload = {
        "feature_scores_path": str(feature_scores_path),
        "a_max_path": str(a_max_path),
        "n_features_used": len(results),
        "top_features": [int(r.feature_idx) for r in results],
        "config": config,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )