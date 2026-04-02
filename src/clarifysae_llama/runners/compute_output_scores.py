from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from clarifysae_llama.config import get_by_dotted_path, load_yaml
from clarifysae_llama.discovery.output_scores import (
    compute_a_max,
    compute_output_scores,
    save_output_score_results,
)
from clarifysae_llama.runners.discover_features import (
    _get_model_input_device,
    _get_module_device,
    _load_model_and_tokenizer,
    _load_sae,
)
from clarifysae_llama.steering.hook_utils import get_submodule_by_path, resolve_module_path
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed


def _load_feature_scores(path: str | Path, tensor_key: str | None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload.float()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported feature score payload type: {type(payload)!r}")
    if tensor_key is None:
        tensor_key = "scores" if "scores" in payload else None
    if tensor_key is None:
        raise KeyError(
            "Could not infer tensor key from feature score file. "
            "Set output_scoring.feature_score_key explicitly."
        )
    value = get_by_dotted_path(payload, tensor_key)
    if not torch.is_tensor(value):
        raise TypeError(f"Feature score entry {tensor_key!r} is not a tensor.")
    return value.float()


def _select_features(score_tensor: torch.Tensor, cfg: dict[str, Any]) -> list[int]:
    if "feature_indices" in cfg and cfg["feature_indices"] is not None:
        return [int(x) for x in cfg["feature_indices"]]
    top_k = int(cfg["top_k_features"])
    return score_tensor.topk(k=top_k).indices.tolist()


def run_output_score_pipeline(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    output_cfg = config["output_scoring"]
    experiment_name = config["experiment_name"]
    output_root = Path(output_cfg.get("root_dir", "outputs/discovery"))
    run_root = ensure_dir(output_root / experiment_name)
    output_dir = ensure_dir(run_root / "output_scores" / output_cfg.get("name", "default"))
    ensure_dir(output_root / "logs")

    feature_scores_path = Path(output_cfg["feature_scores_path"])
    score_tensor = _load_feature_scores(feature_scores_path, output_cfg.get("feature_score_key"))
    feature_ids = _select_features(score_tensor, output_cfg)

    model, tokenizer, dtype = _load_model_and_tokenizer(config["model"])

    module_path = resolve_module_path(
        output_cfg["hookpoint"],
        output_cfg.get("module_path"),
    )
    target_module = get_submodule_by_path(model, module_path)
    sae_device = _get_module_device(target_module)
    model_input_device = _get_model_input_device(model)

    sae = _load_sae(
        discovery_cfg={
            "loader": output_cfg.get("loader", "sparsify"),
            "sae_repo": output_cfg["sae_repo"],
            "sae_file": output_cfg.get("sae_file"),
            "hookpoint": output_cfg["hookpoint"],
        },
        device=sae_device,
        dtype=dtype,
    )

    a_max = compute_a_max(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        target_module=target_module,
        dataset_cfg=output_cfg["amax_dataset"],
        tokenization_cfg=output_cfg.get("tokenization", {}),
        batch_size=int(output_cfg.get("batching", {}).get("token_batch_size", 8)),
        dtype=dtype,
        sae_device=sae_device,
        model_input_device=model_input_device,
    )

    results = compute_output_scores(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        target_module=target_module,
        feature_ids=feature_ids,
        a_max=a_max,
        prompt=output_cfg.get("prompt", "In my experience,"),
        steering_strength=float(output_cfg.get("steering_strength", 6.0)),
        top_k_tokens=int(output_cfg.get("top_k_tokens", 10)),
        dtype=dtype,
        sae_device=sae_device,
        model_input_device=model_input_device,
    )

    save_output_score_results(
        output_dir=output_dir,
        feature_scores_path=feature_scores_path,
        a_max=a_max,
        results=results,
        config=config,
    )

    run_metadata = {
        "experiment_name": experiment_name,
        "feature_scores_path": str(feature_scores_path),
        "output_dir": str(output_dir),
        "n_features_used": len(feature_ids),
        "feature_ids": feature_ids,
        "sae_repo": output_cfg["sae_repo"],
        "hookpoint": output_cfg["hookpoint"],
        "module_path": output_cfg.get("module_path"),
        "loader": output_cfg.get("loader", "sparsify"),
        "sae_file": output_cfg.get("sae_file"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_run(output_root / "logs" / "runs.jsonl", run_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_output_score_pipeline(load_yaml(args.config))