#!/usr/bin/env bash
python -m clarifysae_llama.runners.sweep --config "${1:-configs/sweep_single_feature_strengths_chosen.yaml}"
