from __future__ import annotations

import argparse
import ast
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

"""
python visualization/visualize_outputscore.py --config visualization/outputscore_8b_layer19_C.yaml
"""

from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.output_scores import SingleFeatureIntervention
from clarifysae_llama.utils.io import ensure_dir, write_csv, write_json
from clarifysae_llama.utils.seed import set_seed

try:
    from clarifysae_llama.runners.compute_output_scores import (
        _get_model_input_device,
        _get_module_device,
        _load_model_and_tokenizer,
        _load_sae,
    )
    from clarifysae_llama.steering.hook_utils import get_submodule_by_path, resolve_module_path
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import output-score helpers needed for visualization.") from exc


@dataclass
class FeatureRun:
    feature_idx: int
    prompt: str
    steering_delta: float | None
    amp_factor: float | None
    best_token_id: int | None
    best_token: str | None
    saved_output_score: float | None
    saved_prob: float | None
    saved_rank: int | None
    candidate_token_ids: list[int]
    candidate_tokens: list[str]


class FixedDeltaSingleFeatureIntervention(SingleFeatureIntervention):
    def __init__(
        self,
        target_module,
        sae,
        feature_idx: int,
        steering_delta: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            target_module=target_module,
            sae=sae,
            feature_idx=feature_idx,
            amp_factor=1.0,
            dtype=dtype,
            device=device,
        )
        self.fixed_delta = float(steering_delta)

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None or hidden.ndim != 3:
            return output

        original_device = hidden.device
        original_dtype = hidden.dtype
        updated_hidden = hidden.clone()
        last_hidden = hidden[:, -1, :].to(device=self.device, dtype=self.dtype)

        from clarifysae_llama.discovery.sae_utils import SparseLatents, encode_sparse
        from clarifysae_llama.discovery.output_scores import _decode_from_sparse

        sparse_latents = encode_sparse(self.sae, last_hidden)
        base_top_acts = sparse_latents.top_acts.clone()
        base_top_indices = sparse_latents.top_indices.clone()
        base_recon = _decode_from_sparse(
            self.sae,
            SparseLatents(top_acts=base_top_acts, top_indices=base_top_indices),
            dtype=last_hidden.dtype,
            device=self.device,
        )
        sae_error = (last_hidden.to(torch.float64) - base_recon.to(torch.float64)).to(last_hidden.dtype)

        steered_top_acts = base_top_acts.clone()
        steered_top_indices = base_top_indices.clone()
        steering_delta = float(self.fixed_delta)
        local_max_act = float(torch.max(base_top_acts).item()) if base_top_acts.numel() > 0 else 0.0
        self.last_local_max_act = local_max_act
        self.last_delta = steering_delta

        hit_mask = steered_top_indices == self.feature_idx
        if hit_mask.any():
            steered_top_acts[hit_mask] += steering_delta

        missing_rows = ~hit_mask.any(dim=1)
        if missing_rows.any():
            replacement_col = torch.argmin(steered_top_acts[missing_rows].abs(), dim=1)
            row_idx = torch.arange(replacement_col.shape[0], device=steered_top_acts.device)
            acts_missing = steered_top_acts[missing_rows].clone()
            idx_missing = steered_top_indices[missing_rows].clone()
            idx_missing[row_idx, replacement_col] = self.feature_idx
            acts_missing[row_idx, replacement_col] = steering_delta
            steered_top_indices[missing_rows] = idx_missing
            steered_top_acts[missing_rows] = acts_missing

        steered_recon = _decode_from_sparse(
            self.sae,
            SparseLatents(top_acts=steered_top_acts, top_indices=steered_top_indices),
            dtype=last_hidden.dtype,
            device=self.device,
        )
        steered_last_hidden = (steered_recon + sae_error).to(device=original_device, dtype=original_dtype)
        updated_hidden[:, -1, :] = steered_last_hidden
        if isinstance(output, tuple):
            return (updated_hidden,) + output[1:]
        return updated_hidden


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_maybe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _parse_int_list(value: Any) -> list[int]:
    items = _parse_maybe_list(value)
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _parse_str_list(value: Any) -> list[str]:
    items = _parse_maybe_list(value)
    return [str(item) for item in items]


def _first_present(row: pd.Series, columns: list[str]) -> Any | None:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return None


def _build_feature_runs(feature_cfg: dict[str, Any], prompt_cfg: dict[str, Any], steering_cfg: dict[str, Any]) -> list[FeatureRun]:
    mode = str(feature_cfg.get("mode", "manual")).strip().lower()
    if mode == "manual":
        feature_ids = feature_cfg.get("feature_ids") or []
        prompt = str(prompt_cfg.get("prompt", "In my experience,"))
        steering_delta = steering_cfg.get("steering_delta")
        amp_factor = steering_cfg.get("amp_factor")
        return [
            FeatureRun(
                feature_idx=int(feature_id),
                prompt=prompt,
                steering_delta=float(steering_delta) if steering_delta is not None else None,
                amp_factor=float(amp_factor) if amp_factor is not None else None,
                best_token_id=None,
                best_token=None,
                saved_output_score=None,
                saved_prob=None,
                saved_rank=None,
                candidate_token_ids=[],
                candidate_tokens=[],
            )
            for feature_id in feature_ids
        ]

    if mode not in {"from_csv", "from_table"}:
        raise ValueError(f"Unsupported feature selection mode: {mode}")

    path = feature_cfg.get("path")
    if not path:
        raise ValueError("feature_source.path is required when selecting features from a table.")
    df = _load_table(path)
    score_column = str(feature_cfg.get("score_column", "output_score"))
    feature_column = str(feature_cfg.get("feature_column", "feature_idx"))
    if score_column not in df.columns:
        raise KeyError(
            f"Score column {score_column!r} not found in {path}. Available columns: {sorted(df.columns)}"
        )
    if feature_column not in df.columns:
        raise KeyError(
            f"Feature column {feature_column!r} not found in {path}. Available columns: {sorted(df.columns)}"
        )

    ranked = df.sort_values(score_column, ascending=False).head(int(feature_cfg.get("top_n", 3)))
    prompt_column = str(prompt_cfg.get("prompt_column", "prompt"))
    delta_columns = [str(x) for x in steering_cfg.get("delta_columns", ["steering_delta", "steering_scale", "output_steering_delta"]) ]
    amp_columns = [str(x) for x in steering_cfg.get("amp_columns", ["amp_factor", "output_amp_factor"]) ]
    best_token_id_columns = [str(x) for x in feature_cfg.get("best_token_id_columns", ["best_token_id", "output_best_token_id"]) ]
    best_token_columns = [str(x) for x in feature_cfg.get("best_token_columns", ["best_token", "output_best_token"]) ]

    runs: list[FeatureRun] = []
    for _, row in ranked.iterrows():
        prompt = str(row[prompt_column]) if prompt_cfg.get("use_saved_prompt", True) and prompt_column in row else str(prompt_cfg.get("prompt", "In my experience,"))

        steering_delta = None
        for column in delta_columns:
            if column in row and pd.notna(row[column]):
                steering_delta = float(row[column])
                break

        amp_factor = None
        for column in amp_columns:
            if column in row and pd.notna(row[column]):
                amp_factor = float(row[column])
                break

        best_token_id = None
        for column in best_token_id_columns:
            if column in row and pd.notna(row[column]):
                best_token_id = int(row[column])
                break

        best_token = None
        for column in best_token_columns:
            if column in row and pd.notna(row[column]):
                best_token = str(row[column])
                break

        saved_prob_value = _first_present(row, ["top_token_score", "prob", "output_top_token_score"])
        saved_rank_value = _first_present(row, ["best_rank", "rank", "output_best_rank"])

        candidate_token_ids = _parse_int_list(row["top_token_ids"]) if "top_token_ids" in row else []
        candidate_tokens = _parse_str_list(row["top_tokens"]) if "top_tokens" in row else []

        runs.append(
            FeatureRun(
                feature_idx=int(row[feature_column]),
                prompt=prompt,
                steering_delta=steering_delta,
                amp_factor=amp_factor,
                best_token_id=best_token_id,
                best_token=best_token,
                saved_output_score=float(row[score_column]) if pd.notna(row[score_column]) else None,
                saved_prob=float(saved_prob_value) if saved_prob_value is not None else None,
                saved_rank=int(saved_rank_value) if saved_rank_value is not None else None,
                candidate_token_ids=candidate_token_ids,
                candidate_tokens=candidate_tokens,
            )
        )
    return runs


def _format_token_for_display(token: str, token_id: int | None = None) -> str:
    """Return a slide-friendly token label.

    Tokenizers often encode a leading space as a special marker such as ▁ or Ġ.
    Earlier versions displayed this as ␠, but many Matplotlib fonts render that
    symbol as a square box. For presentation plots we simply remove the marker
    and show the human-readable token.
    """
    label = str(token)
    label = label.replace("\n", "\\n").replace("\t", "\\t")
    label = label.replace("▁", " ").replace("Ġ", " ")
    label = label.strip()
    if not label:
        return f"id={token_id}" if token_id is not None else "∅"
    return label


def _is_readable_plot_label(label: str) -> bool:
    """Filter tokens that usually render as tofu/square boxes in default fonts."""
    if not label or "�" in label or label.startswith("id="):
        return False
    for ch in label:
        category = unicodedata.category(ch)
        if category.startswith("C"):
            return False
        name = unicodedata.name(ch, "")
        if any(block in name for block in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            return False
    return True


def _clean_token_label(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    return _format_token_for_display(token, int(token_id))


def _next_token_probs(model, tokenizer, prompt: str, model_input_device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model_input_device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1]
    return torch.softmax(logits.float(), dim=0).detach().cpu()


def _topk_dataframe(tokenizer, probs: torch.Tensor, top_k: int) -> pd.DataFrame:
    values, ids = torch.topk(probs, k=int(top_k))
    return pd.DataFrame(
        {
            "token_id": [int(x) for x in ids.tolist()],
            "token": [_clean_token_label(tokenizer, int(x)) for x in ids.tolist()],
            "probability": [float(x) for x in values.tolist()],
        }
    )


def _plot_topk_before_after(
    out_path: Path,
    tokenizer,
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    feature_idx: int,
    top_k: int,
) -> pd.DataFrame:
    """Plot ordinary top-k tokens before/after steering.

    This is useful as a sanity check, but it is not the main OutputScore plot:
    OutputScore is defined on feature-associated candidate tokens.
    """
    before = _topk_dataframe(tokenizer, probs_before, top_k)
    after = _topk_dataframe(tokenizer, probs_after, top_k)

    union_ids = list(dict.fromkeys(before["token_id"].tolist() + after["token_id"].tolist()))
    rows: list[dict[str, Any]] = []
    for token_id in union_ids:
        rows.append(
            {
                "token_id": int(token_id),
                "token": _clean_token_label(tokenizer, int(token_id)),
                "before": float(probs_before[int(token_id)].item()),
                "after": float(probs_after[int(token_id)].item()),
                "delta": float((probs_after[int(token_id)] - probs_before[int(token_id)]).item()),
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("after", ascending=False).head(max(top_k, len(before)))

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.45 * len(plot_df))))
    before_sorted = plot_df.sort_values("before", ascending=True)
    after_sorted = plot_df.sort_values("after", ascending=True)
    axes[0].barh(before_sorted["token"], before_sorted["before"])
    axes[0].set_title("До усиления")
    axes[0].set_xlabel("вероятность")
    axes[1].barh(after_sorted["token"], after_sorted["after"])
    axes[1].set_title("После усиления")
    axes[1].set_xlabel("вероятность")

    xmax = max(float(plot_df["before"].max()), float(plot_df["after"].max()), 1e-9) * 1.05
    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)

    fig.suptitle(f"Признак {feature_idx}: глобальные top-k токены до/после")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _rank_tensor_descending(probs: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(probs, dim=0, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), dtype=order.dtype)
    return ranks


def _candidate_token_dataframe(
    tokenizer,
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    candidate_token_ids: list[int],
    candidate_tokens: list[str] | None = None,
) -> pd.DataFrame:
    """Build the table that actually corresponds to OutputScore.

    candidate_token_ids are the feature-associated tokens from the SAE decoder/logit lens
    stored in output_scores.csv as top_token_ids/top_tokens.
    """
    unique_ids = list(dict.fromkeys(int(x) for x in candidate_token_ids))
    ranks_after = _rank_tensor_descending(probs_after)

    token_from_csv: dict[int, str] = {}
    if candidate_tokens:
        for token_id, token in zip(candidate_token_ids, candidate_tokens):
            token_from_csv[int(token_id)] = str(token)

    rows: list[dict[str, Any]] = []
    for token_id in unique_ids:
        before = float(probs_before[token_id].item())
        after = float(probs_after[token_id].item())
        raw_token = token_from_csv.get(int(token_id), tokenizer.convert_ids_to_tokens([int(token_id)])[0])
        display_token = _format_token_for_display(raw_token, int(token_id))
        rows.append(
            {
                "token_id": int(token_id),
                "token": str(raw_token).replace("\n", "\\n"),
                "display_token": display_token,
                "readable_for_plot": _is_readable_plot_label(display_token),
                "prob_before": before,
                "prob_after": after,
                "delta": after - before,
                "rank_after": int(ranks_after[token_id].item()) + 1,
            }
        )
    return pd.DataFrame(rows)


def _plot_candidate_tokens(
    out_path: Path,
    candidate_df: pd.DataFrame,
    feature_idx: int,
    saved_output_score: float | None,
    saved_prob: float | None,
    saved_rank: int | None,
    max_tokens_for_plot: int = 8,
    drop_unreadable_tokens: bool = True,
    show_saved_metrics_in_title: bool = False,
    show_rank_in_title: bool = False,
) -> None:
    if candidate_df.empty:
        return

    # The CSV metrics may come from an older OutputScore runner. The bars below
    # are recomputed by this visualization run, so the plot title should describe
    # the plotted probabilities, not silently mix them with stored CSV values.
    current_best = candidate_df.sort_values("prob_after", ascending=False).iloc[0]

    plot_df = candidate_df.copy()
    if drop_unreadable_tokens and "readable_for_plot" in plot_df.columns:
        plot_df = plot_df[plot_df["readable_for_plot"].astype(bool)]
    if plot_df.empty:
        plot_df = candidate_df.copy()
    if max_tokens_for_plot > 0:
        plot_df = plot_df.sort_values("prob_after", ascending=False).head(int(max_tokens_for_plot))
    plot_df = plot_df.sort_values("prob_after", ascending=True)

    token_col = "display_token" if "display_token" in plot_df.columns else "token"
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.48 * len(plot_df))))

    axes[0].barh(plot_df[token_col], plot_df["prob_before"])
    axes[0].set_title("До усиления")
    axes[0].set_xlabel("вероятность")

    axes[1].barh(plot_df[token_col], plot_df["prob_after"])
    axes[1].set_title("После усиления")
    axes[1].set_xlabel("вероятность")

    xmax = max(float(plot_df["prob_before"].max()), float(plot_df["prob_after"].max()), 1e-9) * 1.05
    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)

    best_label = str(current_best.get("display_token", current_best.get("token", "")))
    prob_after = float(current_best["prob_after"])
    title = (
        f"Признак {feature_idx}: токены, связанные с признаком"
        f"\nЛучший токен-кандидат после усиления: «{best_label}», "
        f"вероятность = {prob_after:.2f}"
    )
    if show_rank_in_title:
        title += f", место в распределении = {int(current_best['rank_after'])}"
    if show_saved_metrics_in_title:
        stored_parts = []
        if saved_output_score is not None:
            stored_parts.append(f"CSV OutputScore={saved_output_score:.4g}")
        if saved_prob is not None:
            stored_parts.append(f"CSV p={saved_prob:.4g}")
        if saved_rank is not None:
            stored_parts.append(f"CSV rank={saved_rank}")
        if stored_parts:
            title += "\n" + " | ".join(stored_parts)
    fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _select_token_id(
    mode: str,
    run: FeatureRun,
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
) -> tuple[int, str]:
    mode = mode.strip().lower()

    if mode == "saved_best" and run.best_token_id is not None:
        return int(run.best_token_id), "saved_best"

    candidate_ids = list(dict.fromkeys(int(x) for x in run.candidate_token_ids))
    if candidate_ids:
        if mode in {"saved_best", "candidate_best_after", "candidate"}:
            values = probs_after[candidate_ids]
            return int(candidate_ids[int(torch.argmax(values).item())]), "candidate_best_after"
        if mode == "candidate_max_delta":
            values = probs_after[candidate_ids] - probs_before[candidate_ids]
            return int(candidate_ids[int(torch.argmax(values).item())]), "candidate_max_delta"

    if mode in {"saved_best", "candidate_best_after", "candidate_max_delta", "candidate"}:
        delta = probs_after - probs_before
        return int(torch.argmax(delta).item()), "global_max_delta_fallback"

    if mode == "max_delta":
        delta = probs_after - probs_before
        return int(torch.argmax(delta).item()), "global_max_delta"

    raise ValueError(
        f"Unknown selected_token_mode={mode!r}. Expected one of: "
        "saved_best, candidate_best_after, candidate_max_delta, max_delta."
    )


def _plot_selected_token_delta(
    out_path: Path,
    tokenizer,
    probs_before: torch.Tensor,
    probs_after: torch.Tensor,
    token_id: int,
    feature_idx: int,
) -> dict[str, Any]:
    token = _clean_token_label(tokenizer, token_id)
    before = float(probs_before[token_id].item())
    after = float(probs_after[token_id].item())
    delta = after - before

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(["до усиления", "после усиления"], [before, after])
    ax.set_ylabel("вероятность")
    ax.set_title(f"Признак {feature_idx}: вероятность токена «{token}»")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "token_id": int(token_id),
        "token": token,
        "prob_before": before,
        "prob_after": after,
        "delta": delta,
    }


def run_outputscore_visualization(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    viz_cfg = config["output_visualization"]
    output_dir = ensure_dir(viz_cfg.get("output_dir", "outputs/visualizations/outputscore"))

    feature_runs = _build_feature_runs(
        viz_cfg["feature_source"],
        viz_cfg.get("prompt_source", {}),
        viz_cfg.get("steering", {}),
    )

    model_cfg = config["model"]
    model, tokenizer, dtype = _load_model_and_tokenizer(model_cfg)
    module_path = resolve_module_path(viz_cfg["hookpoint"], viz_cfg.get("module_path"))
    target_module = get_submodule_by_path(model, module_path)
    sae_device = _get_module_device(target_module)
    model_input_device = _get_model_input_device(model)
    sae = _load_sae(
        discovery_cfg={
            "loader": viz_cfg.get("loader", "dictionary_learning"),
            "sae_repo": viz_cfg["sae_repo"],
            "sae_file": viz_cfg.get("sae_file"),
            "hookpoint": viz_cfg["hookpoint"],
        },
        device=sae_device,
        dtype=dtype,
    )

    plots_cfg = viz_cfg.get("plots", {})
    top_k = int(plots_cfg.get("top_k_tokens", 10))
    selected_token_mode = str(plots_cfg.get("selected_token_mode", "max_delta")).strip().lower()
    max_candidate_tokens_for_plot = int(plots_cfg.get("max_candidate_tokens_for_plot", 8))
    drop_unreadable_candidate_tokens = bool(plots_cfg.get("drop_unreadable_candidate_tokens", True))
    show_saved_metrics_in_title = bool(plots_cfg.get("show_saved_metrics_in_title", False))
    show_rank_in_title = bool(plots_cfg.get("show_rank_in_title", False))

    summary_rows: list[dict[str, Any]] = []

    for run_idx, run in enumerate(feature_runs):
        probs_before = _next_token_probs(model, tokenizer, run.prompt, model_input_device)

        if run.steering_delta is not None:
            context = FixedDeltaSingleFeatureIntervention(
                target_module=target_module,
                sae=sae,
                feature_idx=int(run.feature_idx),
                steering_delta=float(run.steering_delta),
                dtype=dtype,
                device=sae_device,
            )
        elif run.amp_factor is not None:
            context = SingleFeatureIntervention(
                target_module=target_module,
                sae=sae,
                feature_idx=int(run.feature_idx),
                amp_factor=float(run.amp_factor),
                dtype=dtype,
                device=sae_device,
            )
        else:
            raise ValueError(
                f"Feature {run.feature_idx} has neither steering_delta nor amp_factor available."
            )

        with context:
            probs_after = _next_token_probs(model, tokenizer, run.prompt, model_input_device)

        topk_df = _plot_topk_before_after(
            output_dir / f"feature_{run.feature_idx}__run_{run_idx}__before_after_topk.png",
            tokenizer,
            probs_before,
            probs_after,
            int(run.feature_idx),
            top_k,
        )

        candidate_df = _candidate_token_dataframe(
            tokenizer=tokenizer,
            probs_before=probs_before,
            probs_after=probs_after,
            candidate_token_ids=run.candidate_token_ids,
            candidate_tokens=run.candidate_tokens,
        )
        candidate_df.to_csv(
            output_dir / f"feature_{run.feature_idx}__run_{run_idx}__candidate_token_table.csv",
            index=False,
        )
        _plot_candidate_tokens(
            output_dir / f"feature_{run.feature_idx}__run_{run_idx}__candidate_tokens_before_after.png",
            candidate_df=candidate_df,
            feature_idx=int(run.feature_idx),
            saved_output_score=run.saved_output_score,
            saved_prob=run.saved_prob,
            saved_rank=run.saved_rank,
            max_tokens_for_plot=max_candidate_tokens_for_plot,
            drop_unreadable_tokens=drop_unreadable_candidate_tokens,
            show_saved_metrics_in_title=show_saved_metrics_in_title,
            show_rank_in_title=show_rank_in_title,
        )

        selected_token_id, selected_mode_used = _select_token_id(
            selected_token_mode,
            run,
            probs_before,
            probs_after,
        )

        token_stats = _plot_selected_token_delta(
            output_dir / f"feature_{run.feature_idx}__run_{run_idx}__selected_token_delta.png",
            tokenizer,
            probs_before,
            probs_after,
            int(selected_token_id),
            int(run.feature_idx),
        )

        topk_df.to_csv(
            output_dir / f"feature_{run.feature_idx}__run_{run_idx}__topk_table.csv",
            index=False,
        )

        summary_rows.append(
            {
                "feature_idx": int(run.feature_idx),
                "prompt": run.prompt,
                "saved_output_score": run.saved_output_score,
                "saved_prob": run.saved_prob,
                "saved_rank": run.saved_rank,
                "candidate_token_ids": run.candidate_token_ids,
                "candidate_tokens": run.candidate_tokens,
                "selected_token_mode_requested": selected_token_mode,
                "selected_token_mode_used": selected_mode_used,
                "used_steering_delta": context.last_delta,
                "used_local_max_act": context.last_local_max_act,
                **token_stats,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    write_csv(output_dir / "outputscore_visualization_summary.csv", summary_df)
    write_json(
        output_dir / "run_config.json",
        {
            "experiment_name": config.get("experiment_name", "outputscore_visualization"),
            "config": config,
            "n_runs": len(feature_runs),
        },
    )
    print(f"Saved OutputScore visualizations to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_outputscore_visualization(load_yaml(args.config))
