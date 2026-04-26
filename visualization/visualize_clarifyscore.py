from __future__ import annotations

import argparse
import ast
import json
import re
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
    token_ids: list[int] | None = None


@dataclass
class VocabEntry:
    text: str
    token_group: list[torch.Tensor]


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


def _normalize_token_ids(value: Any) -> list[int]:
    if value is None:
        return []

    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.flatten().tolist()]

    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for x in value:
            try:
                out.append(int(x))
            except (TypeError, ValueError, OverflowError):
                continue
        return out

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        else:
            if converted is not value:
                return _normalize_token_ids(converted)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                parsed = None
            if parsed is not None:
                return _normalize_token_ids(parsed)
            pieces = [piece.strip() for piece in stripped[1:-1].split(",")]
            out: list[int] = []
            for piece in pieces:
                if not piece:
                    continue
                try:
                    out.append(int(piece))
                except (TypeError, ValueError, OverflowError):
                    continue
            return out
        return []

    if isinstance(value, dict):
        return []

    try:
        iterator = iter(value)
    except TypeError:
        iterator = None
    if iterator is not None:
        out: list[int] = []
        for x in iterator:
            try:
                out.append(int(x))
            except (TypeError, ValueError, OverflowError):
                continue
        return out

    try:
        return [int(value)]
    except (TypeError, ValueError, OverflowError):
        return []


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _infer_text_source_mode(text_cfg: dict[str, Any]) -> str:
    mode = str(text_cfg.get("mode", "")).strip().lower()
    if mode:
        return mode
    suffix = Path(text_cfg["path"]).suffix.lower()
    if suffix == ".parquet":
        return "discovery_parquet"
    return "prediction_jsonl"


def _display_field_name(field: str) -> str:
    mapping = {
        "discovery_marker": "marker example",
        "discovery_background": "background example",
        "gold_question": "gold question",
        "ambiguous_instruction": "ambiguous instruction",
        "model_questions": "model question",
    }
    return mapping.get(str(field), str(field))


def _compile_regexes(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if not str(pattern).strip():
            continue
        compiled.append(re.compile(str(pattern), flags=re.IGNORECASE | re.UNICODE))
    return compiled


def _passes_clean_text_filters(text: str, text_cfg: dict[str, Any]) -> bool:
    cleaning_cfg = text_cfg.get("cleaning", {}) or {}
    if not bool(cleaning_cfg.get("enabled", False)):
        return True

    stripped = text.strip()
    min_chars = int(cleaning_cfg.get("min_chars", 20))
    max_chars = int(cleaning_cfg.get("max_chars", 3000))
    if len(stripped) < min_chars or len(stripped) > max_chars:
        return False

    require_patterns = [str(x) for x in cleaning_cfg.get("require_patterns", [])]
    if require_patterns and not any(pattern.search(stripped) for pattern in _compile_regexes(require_patterns)):
        return False

    reject_patterns = [str(x) for x in cleaning_cfg.get("reject_patterns", [])]
    if any(pattern.search(stripped) for pattern in _compile_regexes(reject_patterns)):
        return False

    return True


def _normalize_example_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        item = str(value).strip()
        return [item] if item else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _normalize_selected_examples(value: Any) -> dict[int, set[str]]:
    """Normalize plotting.selected_examples from YAML.

    Expected YAML shape:
      selected_examples:
        63916: ["123", "456"]

    Keys are feature ids; values are example_id values copied from
    clarify_example_candidates_for_review.csv.
    """
    if not isinstance(value, dict):
        return {}

    selected: dict[int, set[str]] = {}
    for raw_feature_idx, raw_ids in value.items():
        try:
            feature_idx = int(raw_feature_idx)
        except (TypeError, ValueError):
            continue
        ids = set(_normalize_example_id_list(raw_ids))
        if ids:
            selected[feature_idx] = ids
    return selected


def _selected_ids_for_loading(selected_by_feature: dict[int, set[str]]) -> list[str]:
    ids: set[str] = set()
    for per_feature_ids in selected_by_feature.values():
        ids.update(per_feature_ids)
    return sorted(ids)


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
                        examples.append(TextExample(example_id=example_id, field=field, text=text, token_ids=None))
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
                                    token_ids=None,
                                )
                            )
                            per_field_counts[field] += 1
            if all(per_field_counts[field] >= max_per_field for field in fields):
                break

    if not examples:
        raise ValueError(f"No usable examples were loaded from {path}")
    return examples


def _load_discovery_examples(
    text_cfg: dict[str, Any],
    tokenizer,
    vocab_entries: list[VocabEntry],
    *,
    expand_range: tuple[int, int] = (0, 0),
) -> list[TextExample]:
    path = Path(text_cfg["path"])
    token_column = str(text_cfg.get("token_column", "tokens"))
    text_column = str(text_cfg.get("text_column", "text"))
    max_rows = text_cfg.get("max_rows")
    max_marker_examples = int(
        text_cfg.get(
            "max_marker_candidate_examples",
            text_cfg.get("max_marker_examples", text_cfg.get("max_per_field", 5)),
        )
    )
    max_background_examples = int(
        text_cfg.get(
            "max_background_candidate_examples",
            text_cfg.get("max_background_examples", 0),
        )
    )
    selected_example_ids = set(_normalize_example_id_list(text_cfg.get("selected_example_ids")))
    use_selected_examples = bool(selected_example_ids)

    df = pd.read_parquet(path)
    if token_column not in df.columns:
        raise KeyError(
            f"Token column {token_column!r} not found in {path}. Available columns: {list(df.columns)}"
        )
    if max_rows is not None:
        df = df.head(int(max_rows))

    marker_examples: list[TextExample] = []
    background_examples: list[TextExample] = []
    rows_scanned = 0
    rows_with_nonempty_tokens = 0
    first_nonempty_token_value: Any | None = None
    first_nonempty_token_type: str | None = None

    for row_idx, row in enumerate(df.to_dict(orient="records")):
        rows_scanned += 1
        raw_value = row.get(token_column)
        if first_nonempty_token_value is None and raw_value is not None:
            first_nonempty_token_value = raw_value
            first_nonempty_token_type = type(raw_value).__name__
        token_ids = _normalize_token_ids(raw_value)
        if not token_ids:
            continue
        rows_with_nonempty_tokens += 1
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        marker_mask, _, _ = _match_vocab_positions(
            token_tensor,
            vocab_entries,
            expand_range=expand_range,
        )
        has_marker = bool(marker_mask.any().item())
        example_id = str(row.get("id", row.get("conversation_id", row_idx)))

        if use_selected_examples:
            if example_id not in selected_example_ids:
                continue
        else:
            if not has_marker and len(background_examples) >= max_background_examples:
                continue
            if has_marker and len(marker_examples) >= max_marker_examples:
                continue

        text_value = row.get(text_column) if text_column in row else None
        if isinstance(text_value, str) and text_value.strip():
            text = text_value.strip()
        else:
            text = _decode_token_ids(tokenizer, token_ids)

        if not _passes_clean_text_filters(text, text_cfg):
            continue

        example = TextExample(
            example_id=example_id,
            field="discovery_marker" if has_marker else "discovery_background",
            text=text,
            token_ids=token_ids,
        )

        if has_marker:
            marker_examples.append(example)
        else:
            background_examples.append(example)

        if use_selected_examples:
            loaded_ids = {example.example_id for example in marker_examples + background_examples}
            if selected_example_ids.issubset(loaded_ids):
                break
        elif len(marker_examples) >= max_marker_examples and len(background_examples) >= max_background_examples:
            break

    examples = marker_examples + background_examples
    if use_selected_examples:
        loaded_ids = {example.example_id for example in examples}
        missing_ids = sorted(selected_example_ids - loaded_ids)
        if missing_ids:
            print(
                "Warning: selected example_id values were not found or did not pass filters: "
                + ", ".join(missing_ids[:20])
                + (" ..." if len(missing_ids) > 20 else "")
            )
    if not examples:
        preview = repr(first_nonempty_token_value)
        if preview and len(preview) > 240:
            preview = preview[:240] + "..."
        raise ValueError(
            f"No usable examples were loaded from {path}. "
            f"Rows scanned: {rows_scanned}. Rows with non-empty parsed token ids: {rows_with_nonempty_tokens}. "
            f"Token column: {token_column!r}. First observed token value type: {first_nonempty_token_type!r}. "
            f"First observed token value preview: {preview}"
        )
    if not marker_examples:
        raise ValueError(
            f"No chunks with vocabulary matches were found in {path}. "
            "Check that the parquet corpus, tokenizer, and vocabulary correspond to the discovery run."
        )
    return examples


def _load_text_examples(
    text_cfg: dict[str, Any],
    tokenizer,
    vocab_entries: list[VocabEntry],
    *,
    expand_range: tuple[int, int] = (0, 0),
) -> list[TextExample]:
    mode = _infer_text_source_mode(text_cfg)
    if mode == "prediction_jsonl":
        return _load_prediction_examples(text_cfg)
    if mode == "discovery_parquet":
        return _load_discovery_examples(text_cfg, tokenizer, vocab_entries, expand_range=expand_range)
    raise ValueError(f"Unsupported text source mode: {mode}")


def _iter_vocab_strings(payload: Any) -> list[str]:
    phrases: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                phrases.append(item)
    elif isinstance(payload, dict):
        # For grouped vocabularies, display the semantic group name rather than
        # flattening all variants. load_vocab_groups(...) preserves the same
        # key order, so these names align with the returned token groups.
        for key in payload.keys():
            phrases.append(str(key))
    else:
        raise ValueError("Vocabulary file must contain either a JSON list or a JSON dictionary.")
    return phrases


def _load_vocab_entries(path: str | Path, tokenizer) -> list[VocabEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_phrases = _iter_vocab_strings(payload)
    token_groups = load_vocab_groups(path, tokenizer)
    if len(raw_phrases) != len(token_groups):
        raise ValueError(
            f"Vocabulary phrase count ({len(raw_phrases)}) does not match token group count ({len(token_groups)})."
        )
    entries: list[VocabEntry] = []
    for phrase, group in zip(raw_phrases, token_groups):
        cleaned = str(phrase).strip()
        if not cleaned:
            continue
        entries.append(VocabEntry(text=cleaned, token_group=group))
    if not entries:
        raise ValueError(f"No usable vocabulary entries were loaded from {path}")
    return entries


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted((int(a), int(b)) for a, b in spans if int(b) > int(a))
    merged: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _match_vocab_positions(
    token_ids_1d: torch.Tensor,
    vocab_entries: list[VocabEntry],
    *,
    expand_range: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, list[str], list[tuple[int, int]]]:
    seq_len = int(token_ids_1d.numel())
    mask = torch.zeros(seq_len, dtype=torch.bool)
    matched_phrases: list[str] = []
    matched_spans: list[tuple[int, int]] = []

    left, right = int(expand_range[0]), int(expand_range[1])
    for entry in vocab_entries:
        entry_matched = False
        for ids_of_interest in entry.token_group:
            ids = ids_of_interest.to(device=token_ids_1d.device)
            ids_len = int(ids.numel())
            if ids_len == 0 or ids_len > seq_len:
                continue
            for start in range(0, seq_len - ids_len + 1):
                if torch.equal(token_ids_1d[start : start + ids_len], ids):
                    span_start = max(0, start - left)
                    span_end = min(seq_len, start + ids_len + right)
                    mask[span_start:span_end] = True
                    matched_spans.append((span_start, span_end))
                    entry_matched = True
        if entry_matched:
            matched_phrases.append(entry.text)
    return mask, matched_phrases, _merge_spans(matched_spans)


def _resolve_heatmap_vmax(values: torch.Tensor, clip_percentile: float = 95.0) -> float:
    values = values.detach().cpu().float()
    if values.numel() == 0:
        return 1.0
    max_value = float(values.max().item())
    if max_value <= 0.0:
        return 1.0

    # SAE activations are very sparse. Quantiling over all values often returns
    # zero, which makes the colorbar display ~1e-6 and saturates every non-zero
    # activation. Quantile only over positive activations instead.
    positive_values = values[values > 0]
    if positive_values.numel() == 0:
        return 1.0

    clip_percentile = float(clip_percentile)
    if 0.0 < clip_percentile < 100.0:
        vmax = float(torch.quantile(positive_values, clip_percentile / 100.0).item())
        return max(vmax, 1e-6)
    return max_value


def _make_local_masks(
    marker_mask: torch.Tensor,
    matched_spans: list[tuple[int, int]],
    seq_len: int,
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    marker_mask = marker_mask.bool().clone()
    if local_window <= 0 or not matched_spans:
        return marker_mask, (~marker_mask)
    local_support = torch.zeros(seq_len, dtype=torch.bool)
    for start, end in matched_spans:
        left = max(0, start - local_window)
        right = min(seq_len, end + local_window)
        local_support[left:right] = True
    local_background = local_support & (~marker_mask)
    return marker_mask, local_background


def _crop_to_focus(
    token_labels: list[str],
    activations: torch.Tensor,
    marker_mask: torch.Tensor,
    matched_spans: list[tuple[int, int]],
    *,
    window_left: int,
    window_right: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    if not matched_spans:
        return token_labels, activations, marker_mask
    first_start = min(start for start, _ in matched_spans)
    last_end = max(end for _, end in matched_spans)
    crop_start = max(0, first_start - int(window_left))
    crop_end = min(len(token_labels), last_end + int(window_right))
    return (
        token_labels[crop_start:crop_end],
        activations[crop_start:crop_end],
        marker_mask[crop_start:crop_end],
    )


def _plot_heatmap(
    out_path: Path,
    token_labels: list[str],
    activations: torch.Tensor,
    title: str,
    marker_mask: torch.Tensor | None = None,
    *,
    clip_percentile: float = 95.0,
) -> None:
    values = activations.detach().cpu().float()
    vmax = _resolve_heatmap_vmax(values, clip_percentile)
    matrix = values.unsqueeze(0).numpy()
    width = max(7, min(0.55 * len(token_labels), 18))
    fig, ax = plt.subplots(figsize=(width, 2.8))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_yticklabels(["activation"])
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("tokens")

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
    """Plot a slide-friendly summary of marker/background activations.

    The historical function name is kept for compatibility, but the figure is a
    mean-bar plot rather than a boxplot. For small, hand-picked visualization
    sets a boxplot tends to look more statistical than the evidence warrants.
    """
    if not rows:
        return

    marker_values: list[float] = []
    local_background_values: list[float] = []
    pure_background_values: list[float] = []

    for row in rows:
        n_marker = int(row.get("n_marker_tokens") or 0)
        if n_marker > 0:
            marker_values.extend(float(x) for x in row.get("marker_values", []))
            local_background_values.extend(float(x) for x in row.get("background_values", []))
        else:
            pure_background_values.extend(float(x) for x in row.get("background_values", []))

    categories: list[tuple[str, list[float]]] = [
        ("marker tokens", marker_values),
        ("nearby background", local_background_values),
        ("background examples", pure_background_values),
    ]
    categories = [(label, values) for label, values in categories if values]
    if not categories:
        return

    labels = [label for label, _ in categories]
    means = [float(torch.tensor(values).float().mean().item()) for _, values in categories]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), 4.2))
    ax.bar(labels, means)
    ax.set_title(f"Feature {feature_idx}: marker vs background activation")
    ax.set_ylabel("mean activation")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _example_quality(row: dict[str, Any], rank_mode: str) -> float:
    marker = row.get("mean_marker_activation")
    background = row.get("mean_background_activation")
    marker = float(marker) if marker is not None and pd.notna(marker) else float("-inf")
    background = float(background) if background is not None and pd.notna(background) else 0.0
    if rank_mode == "marker_over_background":
        return marker / max(background, 1e-6)
    return marker - background


def run_clarifyscore_visualization(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    viz_cfg = config["clarify_visualization"]
    output_dir = ensure_dir(viz_cfg.get("output_dir", "outputs/visualizations/clarifyscore"))

    model_cfg = config["model"]
    model, tokenizer, dtype = _load_model_and_tokenizer(model_cfg)

    scoring_cfg = viz_cfg.get("scoring", {})
    plotting_cfg = viz_cfg.get("plotting", {})
    expand_range = tuple(scoring_cfg.get("expand_range", [0, 0]))
    vocab_entries = _load_vocab_entries(viz_cfg["vocab_path"], tokenizer)
    feature_ids = _select_features(viz_cfg["feature_source"])

    selected_examples_by_feature = _normalize_selected_examples(plotting_cfg.get("selected_examples"))
    text_source_cfg = dict(viz_cfg["text_source"])
    selected_ids_from_plotting = _selected_ids_for_loading(selected_examples_by_feature)
    if selected_ids_from_plotting:
        explicit_ids = set(_normalize_example_id_list(text_source_cfg.get("selected_example_ids")))
        text_source_cfg["selected_example_ids"] = sorted(explicit_ids | set(selected_ids_from_plotting))
    examples = _load_text_examples(text_source_cfg, tokenizer, vocab_entries, expand_range=expand_range)

    sae = _load_sae(
        discovery_cfg={
            "loader": viz_cfg.get("loader", "dictionary_learning"),
            "sae_repo": viz_cfg["sae_repo"],
            "sae_file": viz_cfg.get("sae_file"),
            "hookpoint": viz_cfg["hookpoint"],
        },
        device=torch.device(viz_cfg.get("sae_device", "cpu")) if viz_cfg.get("sae_device") else _get_module_device(getattr(model, "model", model)),
        dtype=dtype,
    )

    extractor = HiddenActivationExtractor(
        model,
        hookpoint=viz_cfg["hookpoint"],
        module_path=viz_cfg.get("module_path"),
    )
    sae_device = _get_module_device(extractor.target_module)
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

    ignore_token_ids = set()
    if scoring_cfg.get("ignore_special_tokens", True):
        ignore_token_ids.update(tokenizer.all_special_ids)
    ignore_token_ids.update(int(x) for x in scoring_cfg.get("ignore_token_ids", []))
    heatmap_clip_percentile = float(scoring_cfg.get("heatmap_clip_percentile", 95.0))
    local_background_window = int(scoring_cfg.get("local_background_window", 12))
    crop_window_left = int(plotting_cfg.get("crop_window_left", 12))
    crop_window_right = int(plotting_cfg.get("crop_window_right", 12))
    max_examples_per_feature = int(plotting_cfg.get("max_examples_per_feature", 3))
    rank_examples_by = str(plotting_cfg.get("rank_examples_by", "marker_minus_background")).strip().lower()

    records: list[dict[str, Any]] = []
    plot_candidates: list[dict[str, Any]] = []

    with extractor:
        for example_idx, example in enumerate(examples):
            if example.token_ids is not None:
                token_ids_list = [int(x) for x in example.token_ids]
            else:
                token_ids_list = tokenizer.encode(example.text, add_special_tokens=False)
            if not token_ids_list:
                continue

            input_ids = torch.tensor([token_ids_list], dtype=torch.long, device=model_input_device)
            attention_mask = torch.ones_like(input_ids)

            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
            with torch.inference_mode():
                _ = model(**model_inputs)
                hidden = extractor.pop()
                hidden = hidden.to(device=sae_device, dtype=dtype)
                latents = encode_dense(sae, hidden)

            token_ids = torch.tensor(token_ids_list, dtype=torch.long)
            token_labels = [_clean_token_label(tokenizer, int(token_id)) for token_id in token_ids.tolist()]
            marker_mask, matched_phrases, matched_spans = _match_vocab_positions(
                token_ids,
                vocab_entries,
                expand_range=(int(expand_range[0]), int(expand_range[1])),
            )
            if ignore_token_ids:
                ignore_mask = torch.tensor([int(tok) in ignore_token_ids for tok in token_ids.tolist()], dtype=torch.bool)
                marker_mask &= ~ignore_mask
            marker_mask_for_stats, background_mask_for_stats = _make_local_masks(
                marker_mask,
                matched_spans,
                seq_len=int(token_ids.numel()),
                local_window=local_background_window,
            )
            if ignore_token_ids:
                background_mask_for_stats &= ~ignore_mask

            for feature_idx in feature_ids:
                acts = latents[0, :, int(feature_idx)].detach().cpu().float()
                marker_values = acts[marker_mask_for_stats].tolist()
                background_values = acts[background_mask_for_stats].tolist()
                mean_marker = float(torch.tensor(marker_values).mean().item()) if marker_values else None
                mean_background = float(torch.tensor(background_values).mean().item()) if background_values else None
                row = {
                    "feature_idx": int(feature_idx),
                    "example_id": example.example_id,
                    "example_index": int(example_idx),
                    "field": example.field,
                    "text": example.text,
                    "matched_phrases": matched_phrases,
                    "n_tokens": int(token_ids.numel()),
                    "n_marker_tokens": int(marker_mask_for_stats.sum().item()),
                    "n_background_tokens": int(background_mask_for_stats.sum().item()),
                    "mean_marker_activation": mean_marker,
                    "mean_background_activation": mean_background,
                    "marker_values": marker_values,
                    "background_values": background_values,
                }
                records.append(row)

                if matched_spans and marker_values:
                    cropped_labels, cropped_acts, cropped_mask = _crop_to_focus(
                        token_labels,
                        acts,
                        marker_mask,
                        matched_spans,
                        window_left=crop_window_left,
                        window_right=crop_window_right,
                    )
                    plot_candidates.append(
                        {
                            **row,
                            "quality": _example_quality(row, rank_examples_by),
                            "token_labels": cropped_labels,
                            "activations": cropped_acts,
                            "marker_mask": cropped_mask,
                        }
                    )

    summary_df = pd.DataFrame(records)
    summary_csv = summary_df.drop(columns=["marker_values", "background_values"], errors="ignore")
    write_csv(output_dir / "clarify_activation_summary.csv", summary_csv)

    review_rows: list[dict[str, Any]] = []
    for feature_idx in feature_ids:
        ranked_for_review = sorted(
            [row for row in plot_candidates if int(row["feature_idx"]) == int(feature_idx)],
            key=lambda row: float(row["quality"]),
            reverse=True,
        )
        for candidate_rank, row in enumerate(ranked_for_review):
            text = str(row.get("text", ""))
            review_rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "candidate_rank": int(candidate_rank),
                    "quality": float(row["quality"]),
                    "example_id": row.get("example_id"),
                    "field": row.get("field"),
                    "matched_phrases": row.get("matched_phrases"),
                    "mean_marker_activation": row.get("mean_marker_activation"),
                    "mean_background_activation": row.get("mean_background_activation"),
                    "n_tokens": row.get("n_tokens"),
                    "n_marker_tokens": row.get("n_marker_tokens"),
                    "text_preview": text[:800],
                }
            )
    if review_rows:
        write_csv(output_dir / "clarify_example_candidates_for_review.csv", pd.DataFrame(review_rows))

    for feature_idx in feature_ids:
        feature_rows = [row for row in records if int(row["feature_idx"]) == int(feature_idx)]
        _plot_marker_background_boxplot(
            output_dir / f"feature_{feature_idx}__marker_vs_background.png",
            feature_rows,
            int(feature_idx),
        )

        ranked = sorted(
            [row for row in plot_candidates if int(row["feature_idx"]) == int(feature_idx)],
            key=lambda row: float(row["quality"]),
            reverse=True,
        )
        selected_ids_for_feature = selected_examples_by_feature.get(int(feature_idx))
        if selected_ids_for_feature:
            ranked = [row for row in ranked if str(row.get("example_id")) in selected_ids_for_feature]
        for plot_rank, row in enumerate(ranked[:max_examples_per_feature]):
            heatmap_name = f"feature_{feature_idx}__{row['field']}__best_{plot_rank}.png"
            _plot_heatmap(
                output_dir / heatmap_name,
                token_labels=row["token_labels"],
                activations=row["activations"],
                title=f"Feature {feature_idx} — {row['field']}",
                marker_mask=row["marker_mask"],
                clip_percentile=heatmap_clip_percentile,
            )

    config_to_save = {
        "experiment_name": config.get("experiment_name", "clarifyscore_visualization"),
        "selected_features": feature_ids,
        "n_examples": len(examples),
        "config": config,
    }
    write_json(output_dir / "run_config.json", config_to_save)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clarifyscore_visualization(load_yaml(args.config))
