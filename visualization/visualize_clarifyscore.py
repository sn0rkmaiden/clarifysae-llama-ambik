from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import patches

from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.sae_utils import encode_dense
from clarifysae_llama.discovery.vocab import load_vocab_groups
from clarifysae_llama.utils.io import ensure_dir, write_json, write_csv
from clarifysae_llama.utils.seed import set_seed

try:
    from clarifysae_llama.runners.discover_features import (
        HiddenActivationExtractor,
        _get_model_input_device,
        _get_module_device,
        _load_model_and_tokenizer,
        _load_sae,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import discovery helpers needed for ClarifyScore visualization.") from exc


@dataclass
class TextExample:
    example_id: str
    field: str
    text: str


def _clean_token_label(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    token = token.replace("\n", "\\n")
    token = token.replace("▁", "␠")
    token = token.replace("Ġ", "␠")
    if token == "":
        token = "∅"
    return token


def _load_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _select_features(feature_cfg: dict[str, Any]) -> list[int]:
    mode = str(feature_cfg.get("mode", "manual")).strip().lower()
    if mode == "manual":
        ids = feature_cfg.get("feature_ids") or []
        if not ids:
            raise ValueError("feature_source.mode='manual' requires feature_ids.")
        return [int(x) for x in ids]

    if mode in {"from_csv", "from_table"}:
        path = feature_cfg.get("path")
        if not path:
            raise ValueError("feature_source.path is required when selecting features from a table.")
        df = _load_feature_table(path)
        score_column = str(feature_cfg.get("score_column", "score"))
        if score_column not in df.columns:
            raise KeyError(
                f"Score column {score_column!r} not found in {path}. Available columns: {sorted(df.columns)}"
            )
        top_n = int(feature_cfg.get("top_n", 3))
        feature_column = str(feature_cfg.get("feature_column", "feature_idx"))
        if feature_column not in df.columns:
            raise KeyError(
                f"Feature column {feature_column!r} not found in {path}. Available columns: {sorted(df.columns)}"
            )
        ranked = df.sort_values(score_column, ascending=False)
        return [int(x) for x in ranked.head(top_n)[feature_column].tolist()]

    raise ValueError(f"Unsupported feature selection mode: {mode}")


def _load_prediction_examples(text_cfg: dict[str, Any]) -> list[TextExample]:
    path = Path(text_cfg["path"])
    fields = list(text_cfg.get("fields", ["gold_question", "ambiguous_instruction"]))
    max_per_field = int(text_cfg.get("max_per_field", 5))

    per_field_counts = {field: 0 for field in fields}
    examples: list[TextExample] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            example_id = str(row.get("id", len(examples)))
            for field in fields:
                if per_field_counts[field] >= max_per_field:
                    continue
                value = row.get(field)
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        examples.append(TextExample(example_id=example_id, field=field, text=text))
                        per_field_counts[field] += 1
                elif field == "model_questions" and isinstance(value, list):
                    question_limit = int(text_cfg.get("model_question_limit", 1))
                    for idx, question in enumerate(value[:question_limit]):
                        if per_field_counts[field] >= max_per_field:
                            break
                        question_text = str(question).strip()
                        if question_text:
                            examples.append(
                                TextExample(
                                    example_id=f"{example_id}_{idx}",
                                    field=field,
                                    text=question_text,
                                )
                            )
                            per_field_counts[field] += 1
            if all(per_field_counts[field] >= max_per_field for field in fields):
                break

    if not examples:
        raise ValueError(f"No usable examples were loaded from {path}")
    return examples


def _match_vocab_positions(
    token_ids_1d: torch.Tensor,
    token_groups: list[list[torch.Tensor]],
    *,
    expand_range: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    seq_len = int(token_ids_1d.numel())
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for group in token_groups:
        for ids_of_interest in group:
            ids = ids_of_interest.to(device=token_ids_1d.device)
            ids_len = int(ids.numel())
            if ids_len == 0 or ids_len > seq_len:
                continue
            for start in range(0, seq_len - ids_len + 1):
                if torch.equal(token_ids_1d[start : start + ids_len], ids):
                    left, right = expand_range
                    span_start = max(0, start - int(left))
                    span_end = min(seq_len, start + ids_len + int(right))
                    mask[span_start:span_end] = True
    return mask


def _plot_heatmap(
    out_path: Path,
    token_labels: list[str],
    activations: torch.Tensor,
    title: str,
    marker_mask: torch.Tensor | None = None,
) -> None:
    values = activations.detach().cpu().float().unsqueeze(0).numpy()
    width = max(10, min(0.45 * len(token_labels), 24))
    fig, ax = plt.subplots(figsize=(width, 2.8))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_yticklabels(["активация"])
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("токены")

    if marker_mask is not None:
        marker_mask = marker_mask.cpu().bool()
        for idx, is_marker in enumerate(marker_mask.tolist()):
            if is_marker:
                rect = patches.Rectangle((idx - 0.5, -0.5), 1.0, 1.0, fill=False, linewidth=1.5)
                ax.add_patch(rect)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_marker_background_boxplot(
    out_path: Path,
    rows: list[dict[str, Any]],
    feature_idx: int,
) -> None:
    if not rows:
        return

    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        field = str(row["field"])
        grouped.setdefault(field, {"marker": [], "background": []})
        grouped[field]["marker"].extend(row["marker_values"])
        grouped[field]["background"].extend(row["background_values"])

    labels: list[str] = []
    series: list[list[float]] = []
    for field, values in grouped.items():
        if values["marker"]:
            labels.append(f"{field}\nмаркеры")
            series.append(values["marker"])
        if values["background"]:
            labels.append(f"{field}\nфон")
            series.append(values["background"])

    if not series:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(series)), 4.5))
    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_title(f"Признак {feature_idx}: активации на целевых и фоновых позициях")
    ax.set_ylabel("активация")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_clarifyscore_visualization(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    viz_cfg = config["clarify_visualization"]
    output_dir = ensure_dir(viz_cfg.get("output_dir", "outputs/visualizations/clarifyscore"))

    feature_ids = _select_features(viz_cfg["feature_source"])
    examples = _load_prediction_examples(viz_cfg["text_source"])

    model_cfg = config["model"]
    model, tokenizer, dtype = _load_model_and_tokenizer(model_cfg)
    sae = _load_sae(
        discovery_cfg={
            "loader": viz_cfg.get("loader", "dictionary_learning"),
            "sae_repo": viz_cfg["sae_repo"],
            "sae_file": viz_cfg.get("sae_file"),
            "hookpoint": viz_cfg["hookpoint"],
        },
        device=torch.device(viz_cfg.get("sae_device", "cpu")) if viz_cfg.get("sae_device") else _get_module_device(getattr(model, 'model', model)),
        dtype=dtype,
    )

    extractor = HiddenActivationExtractor(
        model,
        hookpoint=viz_cfg["hookpoint"],
        module_path=viz_cfg.get("module_path"),
    )
    sae_device = _get_module_device(extractor.target_module)
    # Re-load SAE onto the actual module device if no explicit override was given.
    if not viz_cfg.get("sae_device"):
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

    model_input_device = _get_model_input_device(model)

    scoring_cfg = viz_cfg.get("scoring", {})
    expand_range = tuple(scoring_cfg.get("expand_range", [0, 0]))
    token_groups = load_vocab_groups(viz_cfg["vocab_path"], tokenizer)
    ignore_token_ids = set()
    if scoring_cfg.get("ignore_special_tokens", True):
        ignore_token_ids.update(tokenizer.all_special_ids)
    ignore_token_ids.update(int(x) for x in scoring_cfg.get("ignore_token_ids", []))

    records: list[dict[str, Any]] = []

    with extractor:
        for example_idx, example in enumerate(examples):
            encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(model_input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_input_device)

            model_inputs = {"input_ids": input_ids, "use_cache": False}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask

            with torch.inference_mode():
                _ = model(**model_inputs)
                hidden = extractor.pop()
                hidden = hidden.to(device=sae_device, dtype=dtype)
                latents = encode_dense(sae, hidden)

            token_ids = encoded["input_ids"][0].cpu()
            token_labels = [_clean_token_label(tokenizer, int(token_id)) for token_id in token_ids.tolist()]
            marker_mask = _match_vocab_positions(token_ids, token_groups, expand_range=(int(expand_range[0]), int(expand_range[1])))
            if ignore_token_ids:
                ignore_mask = torch.tensor([int(tok) in ignore_token_ids for tok in token_ids.tolist()], dtype=torch.bool)
                marker_mask &= ~ignore_mask
                background_mask = (~marker_mask) & (~ignore_mask)
            else:
                background_mask = ~marker_mask

            for feature_idx in feature_ids:
                acts = latents[0, :, int(feature_idx)].detach().cpu().float()
                heatmap_name = f"feature_{feature_idx}__{example.field}__example_{example_idx}.png"
                _plot_heatmap(
                    output_dir / heatmap_name,
                    token_labels=token_labels,
                    activations=acts,
                    title=f"Признак {feature_idx} — {example.field}",
                    marker_mask=marker_mask,
                )

                marker_values = acts[marker_mask].tolist()
                background_values = acts[background_mask].tolist()
                records.append(
                    {
                        "feature_idx": int(feature_idx),
                        "example_id": example.example_id,
                        "field": example.field,
                        "text": example.text,
                        "n_tokens": int(token_ids.numel()),
                        "n_marker_tokens": int(marker_mask.sum().item()),
                        "n_background_tokens": int(background_mask.sum().item()),
                        "mean_marker_activation": float(torch.tensor(marker_values).mean().item()) if marker_values else None,
                        "mean_background_activation": float(torch.tensor(background_values).mean().item()) if background_values else None,
                        "marker_values": marker_values,
                        "background_values": background_values,
                    }
                )

    summary_df = pd.DataFrame(records)
    summary_csv = summary_df.drop(columns=["marker_values", "background_values"], errors="ignore")
    write_csv(output_dir / "clarify_activation_summary.csv", summary_csv)

    for feature_idx in feature_ids:
        feature_rows = [row for row in records if int(row["feature_idx"]) == int(feature_idx)]
        _plot_marker_background_boxplot(
            output_dir / f"feature_{feature_idx}__marker_vs_background.png",
            feature_rows,
            int(feature_idx),
        )

    config_to_save = {
        "experiment_name": config.get("experiment_name", "clarifyscore_visualization"),
        "selected_features": [int(x) for x in feature_ids],
        "n_examples": len(examples),
        "config": config,
    }
    write_json(output_dir / "run_config.json", config_to_save)
    print(f"Saved ClarifyScore visualizations to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clarifyscore_visualization(load_yaml(args.config))
