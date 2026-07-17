from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from clarifysae_llama.config import dump_yaml
from clarifysae_llama.experiments.fingerprint import fingerprint_payload, sha256_file
from clarifysae_llama.utils.io import ensure_dir, write_json


def _run(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def git_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd())
    commit = _run(["git", "rev-parse", "HEAD"], root)
    status = _run(["git", "status", "--short"], root)
    diff = _run(["git", "diff", "--binary"], root)
    return {
        "commit": commit,
        "status": status,
        "diff": diff,
        "repo_root": str(root.resolve()),
    }


def environment_metadata() -> dict[str, Any]:
    cuda_device = None
    if torch.cuda.is_available():
        try:
            cuda_device = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "device_count": torch.cuda.device_count(),
            }
        except Exception:
            cuda_device = {"available": True}

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": cuda_device,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def write_run_provenance(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    command: list[str] | None = None,
    extra_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    dump_yaml(run_dir / "resolved_config.yaml", config)

    git = git_metadata(repo_root)
    if git.get("diff"):
        (run_dir / "git_diff.patch").write_text(str(git["diff"]), encoding="utf-8")
        git = {**git, "diff": "git_diff.patch"}

    metadata = {
        "config_fingerprint": fingerprint_payload(config),
        "git": git,
        "environment": environment_metadata(),
        "command": command or sys.argv,
    }
    write_json(run_dir / "run_fingerprint.json", metadata)
    (run_dir / "command.txt").write_text(" ".join(command or sys.argv) + "\n", encoding="utf-8")

    checksums: dict[str, str] = {}
    for path in extra_files or []:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            checksums[str(path_obj)] = sha256_file(path_obj)
    if checksums:
        write_json(run_dir / "input_checksums.json", checksums)
    return metadata
