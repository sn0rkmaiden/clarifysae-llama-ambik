from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def normalize_messages(sample: dict[str, Any], conversation_column: str) -> list[dict[str, str]]:
    conv = sample[conversation_column]
    if isinstance(conv, str):
        conv = json.loads(conv)
    if not isinstance(conv, list):
        raise TypeError(f"Expected a list of messages in column '{conversation_column}', got {type(conv)!r}")

    messages: list[dict[str, str]] = []
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or msg.get("speaker") or "user"
        content = msg.get("content") or msg.get("value") or msg.get("text") or ""
        if not isinstance(content, str):
            content = str(content)
        role = "assistant" if str(role).lower() in {"assistant", "gpt", "model", "bot"} else "user"
        messages.append({"role": role, "content": content})
    return messages


def fallback_render(messages: list[dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m["role"].capitalize()
        parts.append(f"{role}: {m['content']}")
    return "\n\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--conversation-column", default="conversation")
    p.add_argument("--language", default="English")
    p.add_argument("--max-samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--push-to-hub", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.streaming:
        rows = []
        for row in ds:
            if args.language and str(row.get("language", "")).lower() != args.language.lower():
                continue
            rows.append(row)
            if len(rows) >= args.max_samples * 3:
                break
        ds = Dataset.from_list(rows)
    else:
        if args.language and "language" in ds.column_names:
            ds = ds.filter(lambda x: str(x.get("language", "")).lower() == args.language.lower(), num_proc=args.num_proc)
        ds = ds.shuffle(seed=args.seed)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples * 3, len(ds))))

    def transform(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out = {"tokens": [], "text": [], "conversation_id": [], "model": [], "language": []}
        n = len(batch[args.conversation_column])
        for i in range(n):
            sample = {k: batch[k][i] for k in batch.keys()}
            messages = normalize_messages(sample, args.conversation_column)
            if not messages:
                continue
            try:
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=args.max_length,
                )
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = fallback_render(messages)
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=True,
                )["input_ids"]
            if len(tokens) < args.min_tokens:
                continue
            out["tokens"].append(tokens)
            out["text"].append(text)
            out["conversation_id"].append(sample.get("conversation_id"))
            out["model"].append(sample.get("model"))
            out["language"].append(sample.get("language"))
        return out

    keep_cols = [c for c in [args.conversation_column, "conversation_id", "model", "language"] if c in ds.column_names]
    ds = ds.map(
        transform,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in keep_cols],
        num_proc=None if args.streaming else args.num_proc,
        desc="Tokenizing LMSYS with Llama tokenizer",
    )
    ds = ds.filter(lambda x: len(x["tokens"]) >= args.min_tokens, num_proc=None if args.streaming else args.num_proc)
    if len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))

    dsd = DatasetDict({"train": ds})
    dsd.save_to_disk(str(out_dir / "dataset"))
    ds.to_parquet(str(out_dir / "train.parquet"))

    meta = {
        "source_dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "conversation_column": args.conversation_column,
        "language": args.language,
        "max_samples": len(ds),
        "max_length": args.max_length,
        "min_tokens": args.min_tokens,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub, private=True)

    print(f"Saved {len(ds)} examples to {out_dir}")
    print("Use column_name: tokens in your discovery config.")


if __name__ == "__main__":
    main()
