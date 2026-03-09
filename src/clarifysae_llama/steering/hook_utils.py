from __future__ import annotations


def get_submodule_by_path(root_module, path: str):
    current = root_module
    for part in path.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def map_sae_hookpoint_to_hf_module_path(hookpoint: str) -> str:
    if hookpoint == 'embed_tokens':
        return 'model.embed_tokens'
    if hookpoint.startswith('layers.'):
        layer_idx = hookpoint.split('.')[1]
        return f'model.layers.{layer_idx}'
    raise ValueError(f'Unsupported hookpoint mapping for Llama-style model: {hookpoint}')
