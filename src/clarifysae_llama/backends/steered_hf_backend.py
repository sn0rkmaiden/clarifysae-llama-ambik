from __future__ import annotations

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.steering.config import SteeringConfig
from clarifysae_llama.steering.sparsify_steerer import SparsifySteerer


class SteeredHFCausalBackend(HFCausalBackend):
    def __init__(self, config: dict):
        super().__init__(config)

        steering_cfg = config["steering"]
        runtime_cfg = steering_cfg.get("runtime", {})
        model_device = next(self.model.parameters()).device

        self.steering = SparsifySteerer(
            model=self.model,
            model_device=model_device,
            dtype=self.dtype,
            config=SteeringConfig(
                sae_repo=steering_cfg["sae_repo"],
                hookpoint=steering_cfg["hookpoint"],
                feature_indices=list(steering_cfg["feature_indices"]),
                strength=float(steering_cfg["strength"]),
                loader=steering_cfg.get("loader", "sparsify"),
                sae_file=steering_cfg.get("sae_file"),
                module_path=steering_cfg.get("module_path"),
                mode=steering_cfg.get("mode", "additive"),
                apply_to=steering_cfg.get("apply_to", "all_positions"),
                steer_generated_tokens_only=steering_cfg.get("steer_generated_tokens_only", False),
                normalize_reconstruction=runtime_cfg.get("normalize_reconstruction", False),
                preserve_unsteered_residual=runtime_cfg.get("preserve_unsteered_residual", False),
                clamp_latents=runtime_cfg.get("clamp_latents"),
                log_feature_acts=runtime_cfg.get("log_feature_acts", False),
            ),
        )

    def generate(self, prompt: str) -> str:
        self.steering.reset()
        self.steering.attach()
        try:
            return super().generate(prompt)
        finally:
            self.steering.detach()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.steering.reset()
        self.steering.attach()
        try:
            return super().generate_batch(prompts)
        finally:
            self.steering.detach()