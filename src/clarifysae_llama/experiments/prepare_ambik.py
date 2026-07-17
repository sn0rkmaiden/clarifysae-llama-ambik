from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.experiments.fingerprint import sha256_file
from clarifysae_llama.utils.io import ensure_dir, write_json


REQUIRED_COLUMNS = {
    "environment_full",
    "unambiguous_direct",
    "ambiguity_type",
    "ambiguous_task",
    "question",
    "answer",
    "plan_for_clear_task",
}

# Stable semantic fields used to identify a scenario across historical exports.
IDENTITY_COLUMNS = [
    "environment_full",
    "unambiguous_direct",
    "ambiguity_type",
    "ambiguous_task",
    "question",
    "answer",
    "plan_for_clear_task",
    "plan_for_amb_task",
]


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _scenario_hash(row: pd.Series) -> str:
    payload = "\u241f".join(_clean_text(row.get(column, "")) for column in IDENTITY_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _drop_export_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # Historical CSV exports contain one or two unnamed dataframe-index columns.
    return frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed:")], errors="ignore")


def _validate_source(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    frame = _drop_export_index_columns(frame.copy())
    if "id" in frame.columns:
        frame = frame.rename(columns={"id": "source_original_id"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}")
    frame["scenario_hash"] = frame.apply(_scenario_hash, axis=1)
    if frame["scenario_hash"].duplicated().any():
        duplicates = frame.loc[frame["scenario_hash"].duplicated(), "scenario_hash"].tolist()[:10]
        raise ValueError(f"{source_name} contains duplicate scenarios: {duplicates}")
    return frame


def prepare_ambik_full_dataset(
    *,
    calib100_path: str | Path,
    test900_path: str | Path,
    output_path: str | Path,
    expected_rows: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one canonical 1,000-row AmbiK file from the historical 100 + 900 exports.

    The supplied ``ambik_test_400.csv`` is intentionally not an input: it is a
    redundant historical copy of the beginning of the 900-row export and is not
    an independent split.
    """
    calib100_path = Path(calib100_path)
    test900_path = Path(test900_path)
    output_path = Path(output_path)

    calib = _validate_source(pd.read_csv(calib100_path), "calib100")
    test = _validate_source(pd.read_csv(test900_path), "test900")

    overlap = set(calib["scenario_hash"]).intersection(test["scenario_hash"])
    if overlap:
        raise ValueError(
            "The 100-row and 900-row sources are not disjoint; "
            f"found {len(overlap)} overlapping scenarios."
        )

    calib["source_export"] = calib100_path.name
    calib["source_row"] = range(len(calib))
    test["source_export"] = test900_path.name
    test["source_row"] = range(len(test))

    full = pd.concat([calib, test], ignore_index=True, sort=False)
    if len(full) != expected_rows:
        raise ValueError(f"Expected {expected_rows} total rows, found {len(full)}.")
    if full["scenario_hash"].duplicated().any():
        raise ValueError("The merged AmbiK file contains duplicate scenarios.")

    # Use immutable content-derived IDs instead of unstable dataframe indices.
    full.insert(0, "id", full["scenario_hash"].map(lambda value: f"ambik_{value[:20]}"))
    if full["id"].duplicated().any():
        raise ValueError("Content-derived AmbiK IDs unexpectedly collided.")

    preferred_columns = [
        "id",
        "scenario_hash",
        "source_export",
        "source_row",
        "environment_short",
        "environment_full",
        "unambiguous_direct",
        "unambiguous_indirect",
        "ambiguity_type",
        "amb_shortlist",
        "ambiguous_task",
        "question",
        "answer",
        "plan_for_clear_task",
        "plan_for_amb_task",
        "end_of_ambiguity",
        "user_intent",
        "variants",
    ]
    ordered = [column for column in preferred_columns if column in full.columns]
    extras = [column for column in full.columns if column not in ordered]
    full = full[ordered + extras]

    ensure_dir(output_path.parent)
    full.to_csv(output_path, index=False)

    category_counts = full["ambiguity_type"].astype(str).value_counts().sort_index().to_dict()
    metadata = {
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "total_rows": len(full),
        "category_counts": category_counts,
        "sources": {
            "calib100": {
                "path": str(calib100_path),
                "sha256": sha256_file(calib100_path),
                "rows": len(calib),
            },
            "test900": {
                "path": str(test900_path),
                "sha256": sha256_file(test900_path),
                "rows": len(test),
            },
        },
        "id_method": "ambik_ + first 20 hex characters of SHA256 over normalized scenario fields",
        "note": "ambik_test_400.csv is redundant and is not used to construct the canonical dataset.",
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return full, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the historical AmbiK calib100 and test900 CSVs into one canonical dataset."
    )
    parser.add_argument("--calib100", required=True)
    parser.add_argument("--test900", required=True)
    parser.add_argument("--output", default="data/processed/ambik/ambik_full_1000.csv")
    parser.add_argument("--expected-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = prepare_ambik_full_dataset(
        calib100_path=args.calib100,
        test900_path=args.test900,
        output_path=args.output,
        expected_rows=args.expected_rows,
    )
    print(frame["ambiguity_type"].value_counts().sort_index())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
