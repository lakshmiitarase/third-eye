"""Generate distinct prompt-only identities from controlled witness attributes.

This mode intentionally does not use the confirmed reference or ControlNet.
It creates alternative identities that match the parsed description, not the
exact witness-confirmed geometry. Model loading remains lazy.
"""

import gc
import json
from datetime import datetime
from pathlib import Path

from photorealistic_generator import (
    BASE_CONFIG_ID,
    BASE_MODEL_FILE,
    BASE_MODEL_ID,
    VAE_MODEL_ID,
    build_prompts,
)


DEFAULT_SEEDS = (9101, 9102, 9103)
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

_PROMPT_PIPELINE = None


def _validate_seeds(seeds):
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds must be a non-empty list or tuple")

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("every seed must be an integer")
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError("every seed must be between 0 and 4294967295")


def _validate_settings(steps, guidance_scale, width, height):
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise TypeError("steps must be an integer")
    if not 1 <= steps <= 50:
        raise ValueError("steps must be between 1 and 50")

    if not isinstance(guidance_scale, (int, float)):
        raise TypeError("guidance_scale must be numeric")
    if not 1.0 <= guidance_scale <= 15.0:
        raise ValueError("guidance_scale must be between 1.0 and 15.0")

    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or value < 256 or value > 768:
            raise ValueError(
                f"{name} must be an integer between 256 and 768"
            )
        if value % 8 != 0:
            raise ValueError(f"{name} must be divisible by 8")


def load_prompt_pipeline(
    base_model_id=BASE_MODEL_ID,
    base_model_file=BASE_MODEL_FILE,
):
    """Lazily load a low-memory prompt-only Stable Diffusion pipeline."""
    global _PROMPT_PIPELINE

    if _PROMPT_PIPELINE is not None:
        return _PROMPT_PIPELINE

    import torch
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
    )
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for local prompt-only generation"
        )

    dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    checkpoint_path = hf_hub_download(
        repo_id=base_model_id,
        filename=base_model_file,
    )

    pipeline = StableDiffusionPipeline.from_single_file(
        checkpoint_path,
        config=BASE_CONFIG_ID,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
    )
    pipeline.enable_attention_slicing("max")
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()

    _PROMPT_PIPELINE = pipeline
    return _PROMPT_PIPELINE


def unload_prompt_pipeline():
    """Release the prompt-only pipeline and CUDA cache."""
    global _PROMPT_PIPELINE

    _PROMPT_PIPELINE = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def generate_prompt_variants(
    analysis,
    output_dir,
    seeds=DEFAULT_SEEDS,
    steps=DEFAULT_STEPS,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    pipeline=None,
):
    """Generate auditable, distinct identities from one attribute set."""
    seeds = tuple(seeds)
    _validate_seeds(seeds)
    _validate_settings(steps, guidance_scale, width, height)

    prompt, negative_prompt = build_prompts(analysis)
    output_directory = Path(output_dir)

    planned_paths = []
    for seed in seeds:
        portrait_path = output_directory / f"prompt_portrait_seed_{seed}.png"
        metadata_path = output_directory / f"prompt_generation_seed_{seed}.json"
        planned_paths.append((seed, portrait_path, metadata_path))

    for _, portrait_path, metadata_path in planned_paths:
        if portrait_path.exists() or metadata_path.exists():
            existing = portrait_path if portrait_path.exists() else metadata_path
            raise FileExistsError(
                f"Prompt-variant output already exists: {existing}"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    active_pipeline = pipeline or load_prompt_pipeline()

    import torch

    records = []

    for seed, portrait_path, metadata_path in planned_paths:
        random_generator = torch.Generator(device="cpu").manual_seed(seed)

        result = active_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=float(guidance_scale),
            width=width,
            height=height,
            generator=random_generator,
            num_images_per_prompt=1,
        )

        if not getattr(result, "images", None):
            raise RuntimeError(
                f"The prompt-only pipeline returned no image for seed {seed}"
            )

        portrait = result.images[0].convert("RGB")
        portrait.save(portrait_path)

        record = {
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "mode": "prompt_only_attribute_variant",
            "uses_confirmed_reference": False,
            "base_model": BASE_MODEL_ID,
            "base_model_file": BASE_MODEL_FILE,
            "vae_model": VAE_MODEL_ID,
            "seed": seed,
            "portrait": str(portrait_path),
            "settings": {
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": float(guidance_scale),
            },
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "notice": (
                "Alternative identity generated from controlled attributes; "
                "it does not preserve the confirmed reference geometry."
            ),
        }

        metadata_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        record["metadata_file"] = str(metadata_path)
        records.append(record)

    return records
