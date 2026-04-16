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

    # Avoid repeated HF warning:
    # "Both `max_new_tokens` and `max_length` seem to have been set"
    if 'max_new_tokens' in kwargs and 'max_length' in kwargs:
        kwargs.pop('max_length', None)

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
        prompting_cfg = config.get('prompting', {})

        self.model_name = model_cfg['name']
        self.dtype = _resolve_torch_dtype(model_cfg.get('torch_dtype', 'bfloat16'))
        self.chat_template_mode = prompting_cfg.get('use_chat_template', 'auto')
        self.system_prompt = prompting_cfg.get(
            'system_prompt',
            'You are a careful assistant. Follow the user instruction exactly and return only the requested output format.',
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Llama-style causal generation should be left-padded in batch mode.
        self.tokenizer.padding_side = 'left'

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

    def _should_use_chat_template(self) -> bool:
        mode = self.chat_template_mode
        if isinstance(mode, bool):
            requested = mode
        else:
            lowered = str(mode).strip().lower()
            if lowered in {'true', '1', 'yes', 'on'}:
                requested = True
            elif lowered in {'false', '0', 'no', 'off'}:
                requested = False
            else:
                requested = bool(getattr(self.tokenizer, 'chat_template', None)) and any(
                    token in self.model_name.lower() for token in ('instruct', 'chat')
                )
        return requested and bool(getattr(self.tokenizer, 'chat_template', None))

    def _stringify_message_content(self, content: Any) -> str:
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        return str(content)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(f'Message must be a dict, got {type(message)!r}')
            role = str(message.get('role', 'user')).strip().lower() or 'user'
            if role not in {'system', 'user', 'assistant'}:
                role = 'user'
            normalized.append({
                'role': role,
                'content': self._stringify_message_content(message.get('content', '')),
            })
        return normalized

    def _flatten_messages(self, messages: list[dict[str, str]]) -> str:
        role_labels = {
            'system': 'System',
            'user': 'User',
            'assistant': 'Assistant',
        }
        parts: list[str] = []
        if self.system_prompt and not any(msg['role'] == 'system' for msg in messages):
            parts.append(f"System: {self.system_prompt}")
        for message in messages:
            role_label = role_labels.get(message['role'], 'User')
            parts.append(f"{role_label}: {message['content']}")
        parts.append('Assistant:')
        return '\n\n'.join(parts)

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        normalized = self._normalize_messages(messages)
        if not self._should_use_chat_template():
            return self._flatten_messages(normalized)

        if self.system_prompt and not any(msg['role'] == 'system' for msg in normalized):
            normalized = [{'role': 'system', 'content': self.system_prompt}, *normalized]

        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_prompt(self, prompt: str) -> str:
        return self._format_messages([{'role': 'user', 'content': prompt}])

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        return [self._format_prompt(prompt) for prompt in prompts]

    @staticmethod
    def _decode_new_tokens(tokenizer, sequence: torch.Tensor, prompt_width: int) -> str:
        continuation_ids = sequence[int(prompt_width):]
        return tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = self._inputs_to_model_device(inputs)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        prompt_width = int(inputs['input_ids'].shape[1])
        return self._decode_new_tokens(self.tokenizer, output[0], prompt_width)

    @torch.inference_mode()
    def generate_messages(self, messages: list[dict[str, Any]]) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = self._inputs_to_model_device(inputs)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        prompt_width = int(inputs['input_ids'].shape[1])
        return self._decode_new_tokens(self.tokenizer, output[0], prompt_width)

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        prompts = self._format_prompts(prompts)
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = self._inputs_to_model_device(inputs)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)

        # In batched generation with left padding, new tokens begin after the
        # full padded prompt width for every row, not after each row's
        # non-padding token count.
        prompt_width = int(inputs['input_ids'].shape[1])
        return [
            self._decode_new_tokens(self.tokenizer, outputs[row_idx], prompt_width)
            for row_idx in range(outputs.shape[0])
        ]
