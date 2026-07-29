from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from clarifysae_llama.clarq_legacy.utils import clarq_task_name
from clarifysae_llama.eval.clarq_metrics import task_type_summary_dataframes
from clarifysae_llama.runners.clarq_run_state import (
    append_dialogue_checkpoint,
    assert_run_config_compatible,
    load_dialogue_checkpoints,
)


def test_clarq_task_name_uses_zero_based_runner_indices() -> None:
    assert clarq_task_name(0) == "1. Gather Resources"
    assert clarq_task_name(5) == "6. Defense Mission"
    assert clarq_task_name(30) == "31. Magic Task"
    with pytest.raises(IndexError):
        clarq_task_name(31)


def test_task_type_summaries_include_names_and_macro_average() -> None:
    rows = pd.DataFrame(
        [
            {
                "task_type_index": 1,
                "task_type_name": "2. Escort Mission",
                "success": 1,
                "step_recall": 1.0,
                "ClarQ_count": 2,
                "ClarQ_rate": 0.5,
                "Goodbye": 1,
                "dialogue_present": 1,
            },
            {
                "task_type_index": 1,
                "task_type_name": "2. Escort Mission",
                "success": 0,
                "step_recall": 0.5,
                "ClarQ_count": 4,
                "ClarQ_rate": 1.0,
                "Goodbye": 0,
                "dialogue_present": 1,
            },
            {
                "task_type_index": 2,
                "task_type_name": "3. Stealth Mission",
                "success": 1,
                "step_recall": 0.25,
                "ClarQ_count": 1,
                "ClarQ_rate": 0.25,
                "Goodbye": 1,
                "dialogue_present": 1,
            },
        ]
    )

    task_df, macro_df = task_type_summary_dataframes(rows)

    assert list(task_df["task_type_index"]) == [1, 2]
    assert list(task_df["task_type_name"]) == [
        "2. Escort Mission",
        "3. Stealth Mission",
    ]
    assert task_df.loc[task_df["task_type_index"] == 1, "success_rate"].item() == 0.5
    assert macro_df["num_task_types"].item() == 2
    assert macro_df["macro_success_rate"].item() == pytest.approx(0.75)
    assert macro_df["macro_step_recall"].item() == pytest.approx(0.5)


def test_task_type_macro_can_exclude_selection_task() -> None:
    rows = pd.DataFrame(
        [
            {
                "task_type_index": 0,
                "task_type_name": "1. Gather Resources",
                "success": 1,
                "step_recall": 1.0,
                "ClarQ_count": 1,
                "ClarQ_rate": 1.0,
                "Goodbye": 1,
                "dialogue_present": 1,
            },
            {
                "task_type_index": 1,
                "task_type_name": "2. Escort Mission",
                "success": 0,
                "step_recall": 0.25,
                "ClarQ_count": 2,
                "ClarQ_rate": 0.5,
                "Goodbye": 0,
                "dialogue_present": 1,
            },
        ]
    )

    _, macro_df = task_type_summary_dataframes(rows, exclude_from_macro=[0])

    assert macro_df["excluded_task_type_indices"].item() == "0"
    assert macro_df["num_task_types"].item() == 1
    assert macro_df["macro_success_rate"].item() == 0.0
    assert macro_df["macro_step_recall"].item() == 0.25


def test_dialogue_checkpoints_resume_latest_unique_dialogues(tmp_path: Path) -> None:
    path = tmp_path / "dialogues.jsonl"
    append_dialogue_checkpoint(
        path,
        task_type_index=1,
        task_type_name="2. Escort Mission",
        dialogue_index=3,
        conversation=["provider one", "seeker one"],
    )
    append_dialogue_checkpoint(
        path,
        task_type_index=1,
        task_type_name="2. Escort Mission",
        dialogue_index=3,
        conversation=["provider replacement", "seeker replacement"],
    )
    append_dialogue_checkpoint(
        path,
        task_type_index=2,
        task_type_name="3. Stealth Mission",
        dialogue_index=0,
        conversation=["provider two", "seeker two"],
    )

    checkpoints = load_dialogue_checkpoints(path)

    assert set(checkpoints) == {(1, 3), (2, 0)}
    assert checkpoints[(1, 3)]["conversation"][0] == "provider replacement"
    assert checkpoints[(2, 0)]["task_type_name"] == "3. Stealth Mission"


def test_existing_run_directory_rejects_different_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_fingerprint.json").write_text(
        json.dumps({"config_fingerprint": "existing"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different resolved configuration"):
        assert_run_config_compatible(run_dir, {"experiment_name": "new"})


def test_existing_run_directory_accepts_matching_config(tmp_path: Path) -> None:
    from clarifysae_llama.experiments.fingerprint import fingerprint_payload

    config = {"experiment_name": "same", "clarq": {"evaluation_set": "0-5"}}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_fingerprint.json").write_text(
        json.dumps({"config_fingerprint": fingerprint_payload(config)}),
        encoding="utf-8",
    )

    assert_run_config_compatible(run_dir, config)


def _load_config(repo: Path, filename: str) -> dict:
    return yaml.safe_load(
        (repo / "configs" / "final" / "clarq" / filename).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("model_key", "feature", "strength", "hookpoint"),
    [
        ("gemma2b", 538, -5.0, "blocks.12.hook_resid_post"),
        ("gemma9b", 344, 3.0, "blocks.20.hook_resid_post"),
        ("llama1b", 6230, -5.0, "model.layers.12"),
        ("llama8b", 62124, 10.0, "model.layers.27"),
    ],
)
def test_final_clarq_config_pairs_are_matched(
    model_key: str,
    feature: int,
    strength: float,
    hookpoint: str,
) -> None:
    repo = Path(__file__).parents[1]
    baseline = _load_config(repo, f"{model_key}_baseline_eval0to5.yaml")
    steered = _load_config(repo, f"{model_key}_clarifysae_eval0to5.yaml")

    assert baseline["clarq"]["evaluation_set"] == "0-5"
    assert steered["clarq"]["evaluation_set"] == "0-5"
    assert baseline["clarq"]["exclude_task_types_from_macro"] == [0]
    assert steered["clarq"]["exclude_task_types_from_macro"] == [0]
    assert baseline["steering"]["enabled"] is False
    assert steered["steering"]["enabled"] is True
    assert steered["steering"]["feature_indices"] == [feature]
    assert steered["steering"]["strength"] == strength
    assert steered["steering"]["hookpoint"] == hookpoint

    for key in (
        "model",
        "generation",
        "prompting",
        "provider_model",
        "provider_generation",
        "provider_prompting",
        "judge_model",
        "judge_generation",
        "judge_prompting",
    ):
        assert baseline[key] == steered[key], f"{model_key}: mismatched {key}"


@pytest.mark.parametrize("strength", [3.0, 5.0])
def test_llama8b_strength_sensitivity_configs_only_change_declared_fields(
    strength: float,
) -> None:
    repo = Path(__file__).parents[1]
    primary = _load_config(repo, "llama8b_clarifysae_eval0to5.yaml")
    sensitivity = _load_config(
        repo,
        f"llama8b_clarifysae_f62124_a{int(strength)}_eval1to5.yaml",
    )

    assert sensitivity["experiment_name"] == (
        f"clarq_eval1to5_llama8b_f62124_a{int(strength)}_sensitivity_v1"
    )
    assert sensitivity["clarq"]["evaluation_set"] == "1-5"
    assert sensitivity["clarq"]["exclude_task_types_from_macro"] == []
    assert sensitivity["steering"]["feature_indices"] == [62124]
    assert sensitivity["steering"]["strength"] == strength
    assert sensitivity["output"]["root_dir"] == "outputs/clarq_reviewer_eval_v1"

    for config in (primary, sensitivity):
        config.pop("experiment_name")
        config["clarq"].pop("evaluation_set")
        config["clarq"].pop("exclude_task_types_from_macro")
        config["steering"].pop("strength")

    assert sensitivity == primary
