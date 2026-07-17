from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any


class MetricBackendError(RuntimeError):
    """Raised when a requested semantic metric backend cannot be loaded or run."""


def normalize_text(text: str | None) -> str:
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def exact_contains_match(a: str | None, b: str | None) -> tuple[bool, float]:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return False, 0.0
    if a_norm == b_norm:
        return True, 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return True, 0.95
    return False, 0.0


@lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise MetricBackendError(
            'sentence-transformers is required for embedding metrics.'
        ) from exc

    try:
        device = os.environ.get('CLARIFYSAE_METRIC_DEVICE', 'cpu')
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise MetricBackendError(
            f'Failed to load sentence-transformer metric model {model_name!r}.'
        ) from exc


def embedding_similarity(
    a: str | None,
    b: str | None,
    *,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
) -> float:
    a_text = (a or '').strip()
    b_text = (b or '').strip()
    if not a_text or not b_text:
        return 0.0

    model = _get_sentence_transformer(model_name)
    try:
        embeddings = model.encode([a_text, b_text], normalize_embeddings=True)
        return float((embeddings[0] * embeddings[1]).sum())
    except Exception as exc:
        raise MetricBackendError(
            f'Embedding metric model {model_name!r} failed while scoring.'
        ) from exc


def best_match_score(
    model_out: str | None,
    gold: str | None,
    threshold: float = 0.75,
    return_pass: bool = False,
):
    matched, lexical_score = exact_contains_match(model_out, gold)
    if matched:
        return (lexical_score, lexical_score >= threshold) if return_pass else lexical_score

    sim = embedding_similarity(model_out, gold)
    return (sim, sim >= threshold) if return_pass else sim


@lru_cache(maxsize=4)
def _get_nli_components(model_name: str = 'roberta-large-mnli'):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise MetricBackendError('transformers and torch are required for NLI metrics.') from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        requested_device = os.environ.get('CLARIFYSAE_METRIC_DEVICE', 'cpu')
        device = torch.device(requested_device)
        model.to(device)
        label2id = model.config.label2id
        entail_id = label2id.get('ENTAILMENT') or label2id.get('entailment') or 2
        return tokenizer, model, device, entail_id
    except Exception as exc:
        raise MetricBackendError(f'Failed to load NLI metric model {model_name!r}.') from exc


def nli_question_similarity(a: str | None, b: str | None) -> float:
    matched, lexical_score = exact_contains_match(a, b)
    if matched:
        return lexical_score

    premise = (a or '').strip()
    hypothesis = (b or '').strip()
    if not premise or not hypothesis:
        return 0.0

    tokenizer, model, device, entail_id = _get_nli_components()
    try:
        import torch
        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = logits.softmax(dim=-1).squeeze(0)
        p1 = float(probs[entail_id].item())

        encoded_rev = tokenizer(
            hypothesis,
            premise,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            logits_rev = model(**encoded_rev).logits
            probs_rev = logits_rev.softmax(dim=-1).squeeze(0)
        p2 = float(probs_rev[entail_id].item())
        return 0.5 * (p1 + p2)
    except Exception as exc:
        raise MetricBackendError('NLI metric backend failed while scoring.') from exc


def initialize_metric_backends(*, enable_nli: bool = False) -> dict[str, Any]:
    """Eagerly load requested metric backends so a run fails before generation."""
    embedding_model = _get_sentence_transformer()
    metadata: dict[str, Any] = {
        'embedding_model': getattr(embedding_model, '_model_card_vars', None) or 'sentence-transformers/all-MiniLM-L6-v2',
        'embedding_loaded': True,
        'nli_enabled': bool(enable_nli),
    }
    if enable_nli:
        _get_nli_components()
        metadata['nli_model'] = 'roberta-large-mnli'
        metadata['nli_loaded'] = True
    return metadata
