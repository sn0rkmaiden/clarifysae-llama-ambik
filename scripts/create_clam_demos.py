#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ("environment_full", "question", "answer")
REQUIRED_COLUMNS = {
    "environment_full",
    "ambiguous_task",
    "unambiguous_direct",
    "ambiguity_type",
    "question",
    "answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create balanced CLAM few-shot demonstrations from AmbiK splits "
            "without overlapping the calibration or reserved test examples."
        )
    )
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--test400", required=True, type=Path)
    parser.add_argument("--test900", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pairs-per-category",
        type=int,
        default=1,
        help="Number of ambiguous/clear source pairs per ambiguity type (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_columns(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def normalize(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    parts = [normalize(result[col]) for col in KEY_COLUMNS]
    result["_example_key"] = parts[0] + "\u241f" + parts[1] + "\u241f" + parts[2]
    return result


def source_label(row: pd.Series) -> str:
    for column in ("id", "Unnamed: 0", "Unnamed: 0.1"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return str(row.name)


def main() -> None:
    args = parse_args()
    if args.pairs_per_category < 1:
        raise ValueError("--pairs-per-category must be at least 1")

    calib = pd.read_csv(args.calib)
    test400 = pd.read_csv(args.test400)
    test900 = pd.read_csv(args.test900)

    for name, df in (
        ("calib", calib),
        ("test400", test400),
        ("test900", test900),
    ):
        check_columns(df, name)

    calib = add_key(calib)
    test400 = add_key(test400)
    test900 = add_key(test900)

    calib_keys = set(calib["_example_key"])
    test400_keys = set(test400["_example_key"])
    test900_keys = set(test900["_example_key"])

    calib_test400_overlap = calib_keys & test400_keys
    calib_test900_overlap = calib_keys & test900_keys
    missing_from_test900 = test400_keys - test900_keys

    print(f"calib rows: {len(calib)}")
    print(f"test400 rows: {len(test400)}")
    print(f"test900 rows: {len(test900)}")
    print(f"calib ∩ test400: {len(calib_test400_overlap)}")
    print(f"calib ∩ test900: {len(calib_test900_overlap)}")
    print(f"test400 missing from test900: {len(missing_from_test900)}")

    if calib_test400_overlap or calib_test900_overlap:
        raise ValueError("Calibration examples overlap another split.")
    if missing_from_test900:
        raise ValueError("test400 is not a subset of test900 under the stable key.")

    excluded_keys = calib_keys | test400_keys
    pool = test900[~test900["_example_key"].isin(excluded_keys)].copy()

    # A source row can only form a useful pair when the ambiguous and clear
    # instructions are both present and actually differ.
    for column in REQUIRED_COLUMNS:
        pool = pool[normalize(pool[column]) != ""]

    pool = pool[
        normalize(pool["ambiguous_task"]).str.casefold()
        != normalize(pool["unambiguous_direct"]).str.casefold()
    ].copy()

    print(f"Eligible demonstration pool: {len(pool)} rows")
    print("Eligible rows by ambiguity type:")
    print(pool["ambiguity_type"].value_counts().to_string())

    selected_parts = []
    for category in sorted(pool["ambiguity_type"].dropna().astype(str).unique()):
        group = pool[pool["ambiguity_type"].astype(str) == category]
        if len(group) < args.pairs_per_category:
            raise ValueError(
                f"Not enough eligible rows for category {category!r}: "
                f"need {args.pairs_per_category}, found {len(group)}"
            )
        selected_parts.append(
            group.sample(
                n=args.pairs_per_category,
                random_state=args.seed,
                replace=False,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=False)
    demonstrations: list[dict[str, str]] = []

    for _, row in selected.iterrows():
        environment = str(row["environment_full"]).strip()
        ambiguous_task = str(row["ambiguous_task"]).strip()
        clear_task = str(row["unambiguous_direct"]).strip()
        question = str(row["question"]).strip()

        demonstrations.append(
            {
                "environment": environment,
                "task": ambiguous_task,
                "label": "AMBIGUOUS",
                "question": question,
            }
        )
        demonstrations.append(
            {
                "environment": environment,
                "task": clear_task,
                "label": "CLEAR",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(demonstrations, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(demonstrations)} demonstrations to: {args.output}")
    print("Selected source rows:")
    for _, row in selected.iterrows():
        print(
            f"  source={source_label(row)}, "
            f"type={row['ambiguity_type']}, "
            f"question={row['question']}"
        )


if __name__ == "__main__":
    main()
