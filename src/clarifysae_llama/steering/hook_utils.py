# src/clarifysae_llama/steering/hook_utils.py
from __future__ import annotations


def get_submodule_by_path(root_module, path: str):
    current = root_module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def normalize_hookpoint_to_module_path(hookpoint: str) -> str:
    hp = hookpoint.strip()

    if hp == "embed_tokens":
        return "model.embed_tokens"
    if hp == "model.embed_tokens":
        return hp

    # Apart-style repos already include the model prefix.
    if hp.startswith("model."):
        return hp

    # EleutherAI sparsify repos usually use layers.X or layers.X.mlp
    if hp.startswith("layers."):
        return f"model.{hp}"

    raise ValueError(f"Unsupported hookpoint for Llama-style model: {hookpoint}")


def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
    return module_path or normalize_hookpoint_to_module_path(hookpoint)