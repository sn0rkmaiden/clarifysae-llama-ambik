from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.experiments.fingerprint import sha256_file
from clarifysae_llama.utils.io import ensure_dir, write_json


def _row_hash(row: pd.Series) -> str:
    fields = [
        str(row.get("id", "")),
        str(row.get("environment_full", "")),
        str(row.get("ambiguous_task", "")),
        str(row.get("unambiguous_direct", "")),
        str(row.get("ambiguity_type", "")),
        str(row.get("question", "")),
        str(row.get("answer", "")),
    ]
    return hashlib.sha256("\u241f".join(fields).encode("utf-8")).hexdigest()


def _largest_remainder_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"Target {target} must be between 0 and {total}.")
    if total == 0:
        return {key: 0 for key in counts}

    exact = {key: target * count / total for key, count in counts.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - allocation[key], counts[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def _stratified_sample_ids(
    frame: pd.DataFrame,
    *,
    target: int,
    seed: int,
    category_column: str,
) -> set[str]:
    if target > len(frame):
        raise ValueError(f"Requested {target} rows from a frame with only {len(frame)} rows.")
    counts = frame[category_column].astype(str).value_counts().to_dict()
    allocation = _largest_remainder_allocation(counts, target)
    rng = random.Random(seed)
    selected: set[str] = set()
    for category in sorted(allocation):
        values = frame.loc[frame[category_column].astype(str) == category, "id"].astype(str).tolist()
        rng.shuffle(values)
        selected.update(values[: allocation[category]])
    if len(selected) != target:
        raise AssertionError(f"Expected {target} selected IDs, got {len(selected)}.")
    return selected


def create_ambik_split(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    explore_size: int = 400,
    confirm_size: int = 600,
    smoke_size: int = 20,
    pilot_size: int = 100,
    seed: int = 20260717,
    category_column: str = "ambiguity_type",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_path = Path(dataset_path)
    frame = pd.read_csv(dataset_path)
    if "id" not in frame.columns:
        if "Unnamed: 0" in frame.columns:
            frame = frame.rename(columns={"Unnamed: 0": "id"})
        else:
            frame.insert(0, "id", range(len(frame)))

    if len(frame) != explore_size + confirm_size:
        raise ValueError(
            f"Dataset has {len(frame)} rows, but explore_size + confirm_size = "
            f"{explore_size + confirm_size}. Set sizes explicitly for this dataset."
        )
    if frame["id"].astype(str).duplicated().any():
        duplicates = frame.loc[frame["id"].astype(str).duplicated(), "id"].tolist()[:10]
        raise ValueError(f"AmbiK IDs must be unique; duplicates include {duplicates}.")
    if category_column not in frame.columns:
        raise ValueError(f"Missing stratification column: {category_column}")

    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    explore_ids = _stratified_sample_ids(
        frame,
        target=explore_size,
        seed=seed,
        category_column=category_column,
    )
    explore_frame = frame[frame["id"].isin(explore_ids)].copy()
    smoke_ids = _stratified_sample_ids(
        explore_frame,
        target=smoke_size,
        seed=seed + 1,
        category_column=category_column,
    )
    pilot_ids = _stratified_sample_ids(
        explore_frame,
        target=pilot_size,
        seed=seed + 2,
        category_column=category_column,
    )
    pilot_ids.update(smoke_ids)
    if len(pilot_ids) > pilot_size:
        # smoke is intended to be nested inside pilot. Rebuild pilot with the
        # smoke rows fixed and stratify the remaining places approximately.
        non_smoke = explore_frame[~explore_frame["id"].isin(smoke_ids)]
        extra = _stratified_sample_ids(
            non_smoke,
            target=pilot_size - len(smoke_ids),
            seed=seed + 3,
            category_column=category_column,
        )
        pilot_ids = set(smoke_ids) | extra

    result = pd.DataFrame({
        "example_id": frame["id"],
        "scenario_id": frame["id"],
        "ambiguity_category": frame[category_column].astype(str),
        "split": frame["id"].map(lambda value: "explore400" if value in explore_ids else "confirm600"),
        "is_smoke20": frame["id"].isin(smoke_ids),
        "is_pilot100": frame["id"].isin(pilot_ids),
        "row_hash": frame.apply(_row_hash, axis=1),
    })
    result = result.sort_values(["split", "ambiguity_category", "example_id"]).reset_index(drop=True)

    split_counts = result.groupby(["split", "ambiguity_category"]).size().unstack(fill_value=0)
    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split_seed": seed,
        "explore_size": explore_size,
        "confirm_size": confirm_size,
        "smoke_size": smoke_size,
        "pilot_size": pilot_size,
        "category_column": category_column,
        "split_counts": split_counts.to_dict(orient="index"),
        "total_rows": len(result),
    }

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False)
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return result, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified AmbiK split manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="splits/ambik_v1.csv")
    parser.add_argument("--explore-size", type=int, default=400)
    parser.add_argument("--confirm-size", type=int, default=600)
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--category-column", default="ambiguity_type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = create_ambik_split(
        dataset_path=args.dataset,
        output_path=args.output,
        explore_size=args.explore_size,
        confirm_size=args.confirm_size,
        smoke_size=args.smoke_size,
        pilot_size=args.pilot_size,
        seed=args.seed,
        category_column=args.category_column,
    )
    print(frame.groupby(["split", "ambiguity_category"]).size())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
