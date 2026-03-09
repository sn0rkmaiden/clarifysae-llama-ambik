from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f'Unsupported torch dtype: {dtype_name}')
    return mapping[dtype_name]


def normalize_generation_kwargs(generation_kwargs: dict[str, Any], tokenizer) -> dict[str, Any]:
    kwargs = dict(generation_kwargs)

    if 'max_tokens' in kwargs and 'max_new_tokens' not in kwargs:
        kwargs['max_new_tokens'] = kwargs.pop('max_tokens')

    kwargs.pop('logprobs', None)
    kwargs.pop('logit_bias', None)

    temperature = kwargs.get('temperature', None)
    if temperature is not None and float(temperature) == 0.0:
        kwargs['do_sample'] = False
        kwargs.pop('temperature', None)
        kwargs.pop('top_p', None)

    if kwargs.get('pad_token_id') is None:
        kwargs['pad_token_id'] = tokenizer.pad_token_id
    if kwargs.get('eos_token_id') is None:
        kwargs['eos_token_id'] = tokenizer.eos_token_id

    return kwargs


class HFCausalBackend:
    def __init__(self, config: dict[str, Any]):
        model_cfg = config['model']
        generation_cfg = config['generation']

        self.model_name = model_cfg['name']
        self.dtype = _resolve_torch_dtype(model_cfg.get('torch_dtype', 'bfloat16'))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {'torch_dtype': self.dtype}
        if model_cfg.get('device_map', None) is not None:
            model_kwargs['device_map'] = model_cfg['device_map']
        if model_cfg.get('attn_implementation', None) is not None:
            model_kwargs['attn_implementation'] = model_cfg['attn_implementation']

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        self.generation_kwargs = normalize_generation_kwargs(generation_cfg, self.tokenizer)

    def _inputs_to_model_device(self, tokenized):
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in tokenized.items()}

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = self._inputs_to_model_device(inputs)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = self._inputs_to_model_device(inputs)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
