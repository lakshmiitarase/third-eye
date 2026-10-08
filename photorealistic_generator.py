"""Low-memory ControlNet/img2img backend for confirmed forensic references.

Model weights are loaded lazily. Importing this module does not initialize
PyTorch, Diffusers, CUDA, or download any model files.
"""

import gc
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


BASE_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
BASE_MODEL_FILE = "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
BASE_CONFIG_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"
DEFAULT_SIZE = 512
DEFAULT_SEED = 240926
DEFAULT_STEPS = 24
DEFAULT_STRENGTH = 0.60
DEFAULT_GUIDANCE_SCALE = 6.5
DEFAULT_CONTROL_SCALE = 0.75

_PIPELINE = None


FEATURE_PROMPT_LABELS = {
    "gender": "gender",
    "age_group": "age",
    "face_shape": "face shape",
    "face_width": "face width",
    "jaw_shape": "jaw shape",
    "chin_shape": "chin shape",
    "hair_length": "hair length",
    "hair_texture": "hair texture",
    "hair_colour": "hair colour",
    "eye_shape": "eye shape",
    "eye_size": "eye size",
    "eyebrow_shape": "eyebrow shape",
    "eyebrow_thickness": "eyebrow thickness",
    "nose_width": "nose width",
    "nose_length": "nose length",
    "lip_thickness": "lip thickness",
    "beard_style": "beard",
    "moustache_style": "moustache",
    "skin_tone": "skin tone",
}

ABSENCE_NEGATIVES = {
    "beard_style": ["beard", "facial beard", "stubble"],
    "moustache_style": ["moustache", "mustache"],
    "glasses": ["glasses", "spectacles", "eyeglasses"],
}

PRESENTATION_NEGATIVES = {
    "male": ["female", "woman", "feminine face", "makeup"],
    "female": ["male", "man", "masculine face"],
}

ATTRIBUTE_NEGATIVES = {
    ("hair_length", "short"): [
        "long hair",
        "shoulder length hair",
        "chin length bob",
    ],
    ("eyebrow_thickness", "thick"): [
        "thin eyebrows",
        "fine eyebrows",
    ],
}

BASE_NEGATIVE_PROMPT = [
    "sketch",
    "drawing",
    "illustration",
    "cartoon",
    "painting",
    "paper texture",
    "wax skin",
    "plastic skin",
    "mannequin",
    "doll",
    "cgi",
    "3d render",
    "airbrushed skin",
    "pencil lines",
    "multiple faces",
    "side profile",
    "cropped head",
    "deformed face",
    "distorted anatomy",
    "extra facial features",
    "blurry",
    "text",
    "watermark",
]

MAX_PROMPT_WORDS = 55
CLIP_MAX_TOKENS = 77


def _validate_number(name, value, minimum, maximum):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _validate_settings(
    seed,
    steps,
    strength,
    guidance_scale,
    control_scale,
):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")

    _validate_number("steps", steps, 1, 50)
    _validate_number("strength", strength, 0.10, 0.80)
    _validate_number("guidance_scale", guidance_scale, 1.0, 15.0)
    _validate_number("control_scale", control_scale, 0.0, 2.0)


def prepare_reference_image(reference_path, size=DEFAULT_SIZE):
    """Load a reference without cropping and center it on white."""
    path = Path(reference_path)

    if not path.is_file():
        raise FileNotFoundError(f"Confirmed reference not found: {path}")

    if not isinstance(size, int) or size < 256 or size > 1024:
        raise ValueError("size must be an integer between 256 and 1024")

    with Image.open(path) as source:
        image = source.convert("RGB")

    contained = ImageOps.contain(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), "white")
    position = (
        (size - contained.width) // 2,
        (size - contained.height) // 2,
    )
    canvas.paste(contained, position)
    return canvas


def create_canny_control(reference_image, low_threshold=100, high_threshold=200):
    """Create a three-channel Canny control image from a PIL image."""
    if not isinstance(reference_image, Image.Image):
        raise TypeError("reference_image must be a PIL image")

    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError(
            "Canny thresholds must satisfy 0 <= low < high <= 255"
        )

    rgb = np.asarray(reference_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Suppress fine pencil hatching while retaining component geometry.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(edges_rgb, mode="RGB")


def _feature_states(analysis):
    if not isinstance(analysis, dict):
        raise TypeError("analysis must be a dictionary")

    features = analysis.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("analysis.features must be a dictionary")

    return features


def _controlled_value(features, feature_name):
    state = features.get(feature_name)
    if not isinstance(state, dict):
        return None

    value = state.get("value")
    if value in (None, "", "unknown"):
        return None

    normalized = str(value).replace("_", " ").strip()
    if state.get("negated") or normalized in {
        "none",
        "no",
        "absent",
        "clean shaven",
    }:
        return None

    return normalized


def _combined_phrase(features, names, noun):
    values = [
        _controlled_value(features, name)
        for name in names
    ]
    values = [value for value in values if value]
    if not values:
        return None
    return f"{' '.join(values)} {noun}"


def _token_ids(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return token_ids


def _token_count(tokenizer, text):
    return len(_token_ids(tokenizer, text))


def _token_budget(tokenizer, default=CLIP_MAX_TOKENS):
    configured = getattr(tokenizer, "model_max_length", default)
    if not isinstance(configured, int) or configured <= 0 or configured > 10000:
        return default
    return min(configured, default)


def _bounded_join(
    parts,
    max_words=MAX_PROMPT_WORDS,
    tokenizer=None,
    required_count=0,
):
    accepted = []
    used_words = 0
    token_limit = _token_budget(tokenizer) if tokenizer is not None else None

    for index, part in enumerate(parts):
        if not part:
            continue
        words = len(part.split())
        candidate = ", ".join(accepted + [part])
        exceeds_budget = (
            _token_count(tokenizer, candidate) > token_limit
            if tokenizer is not None
            else used_words + words > max_words
        )
        if exceeds_budget:
            if index < required_count:
                raise ValueError(
                    "Witness-critical prompt constraints exceed the model "
                    "token budget; generation was stopped instead of "
                    "silently truncating them."
                )
            continue
        accepted.append(part)
        used_words += words

    return ", ".join(accepted)


def build_prompts(analysis, tokenizer=None):
    """Build concise feature-first prompts within the CLIP token budget."""
    features = _feature_states(analysis)

    feature_parts = []

    gender = _controlled_value(features, "gender")
    age = _controlled_value(features, "age_group")
    if gender == "male":
        feature_parts.append("adult man with masculine facial structure")
    elif gender == "female":
        feature_parts.append("adult woman with feminine facial structure")
    elif gender:
        feature_parts.append(gender)
    if age:
        feature_parts.append(age)

    face_shape = _controlled_value(features, "face_shape")
    face_width = _controlled_value(features, "face_width")
    jaw_shape = _controlled_value(features, "jaw_shape")
    chin_shape = _controlled_value(features, "chin_shape")
    if face_shape:
        feature_parts.append(f"{face_shape} face")
    if face_width:
        feature_parts.append(f"{face_width} face width")
    if jaw_shape:
        feature_parts.append(f"{jaw_shape} jaw")
    if chin_shape:
        feature_parts.append(f"{chin_shape} chin")

    feature_parts.extend([
        _combined_phrase(
            features,
            ["hair_length", "hair_texture", "hair_colour"],
            "hair",
        ),
        _combined_phrase(
            features,
            ["eye_size", "eye_shape"],
            "eyes",
        ),
        _combined_phrase(
            features,
            ["eyebrow_thickness", "eyebrow_shape"],
            "eyebrows",
        ),
        _combined_phrase(
            features,
            ["nose_width", "nose_length"],
            "nose",
        ),
    ])

    lips = _controlled_value(features, "lip_thickness")
    if lips:
        feature_parts.append(f"{lips} lips")

    beard = _controlled_value(features, "beard_style")
    moustache = _controlled_value(features, "moustache_style")
    skin_tone = _controlled_value(features, "skin_tone")
    if beard:
        feature_parts.append(f"{beard} beard")
    if moustache:
        feature_parts.append(f"{moustache} moustache")
    if skin_tone:
        feature_parts.append(f"{skin_tone} skin tone")

    if skin_tone:
        image_mode = "natural color forensic photograph"
    else:
        image_mode = "black and white forensic photograph"

    critical_positive = feature_parts + [
        image_mode,
        "photorealistic forensic visualization",
        "preserve reference facial geometry",
        "preserve reference face proportions",
        "front facing",
        "neutral expression",
    ]
    optional_positive = [
        "centered head and shoulders",
        "natural skin pores",
        "subtle skin imperfections",
        "natural facial anatomy",
        "even studio lighting",
        "plain background",
        "sharp focus",
    ]

    # Put witness-critical exclusions first so the prompt word cap cannot
    # discard them in favor of generic quality terms.
    negative = []
    if skin_tone:
        negative.append("grayscale")
    for feature_name, terms in ABSENCE_NEGATIVES.items():
        state = features.get(feature_name)
        if not isinstance(state, dict):
            continue
        value = state.get("value")
        if state.get("negated") or value == "none":
            negative.extend(terms)

    gender_state = features.get("gender")
    if isinstance(gender_state, dict):
        gender_value = str(gender_state.get("value", "")).lower()
        negative.extend(PRESENTATION_NEGATIVES.get(gender_value, []))

    for (feature_name, requested_value), terms in ATTRIBUTE_NEGATIVES.items():
        state = features.get(feature_name)
        if not isinstance(state, dict):
            continue
        value = str(state.get("value", "")).lower()
        if value == requested_value:
            negative.extend(terms)

    critical_negative_count = len(negative)
    negative.extend(BASE_NEGATIVE_PROMPT)
    critical_positive = list(
        dict.fromkeys(part for part in critical_positive if part)
    )
    optional_positive = list(
        dict.fromkeys(part for part in optional_positive if part)
    )
    negative = list(dict.fromkeys(negative))

    prompt = _bounded_join(
        critical_positive + optional_positive,
        tokenizer=tokenizer,
        required_count=len(critical_positive),
    )
    negative_prompt = _bounded_join(
        negative,
        tokenizer=tokenizer,
        required_count=critical_negative_count,
    )
    return prompt, negative_prompt

def load_pipeline(
    base_model_id=BASE_MODEL_ID,
    controlnet_model_id=CONTROLNET_MODEL_ID,
):
    """Load and cache the low-memory pipeline on first generation only."""
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from huggingface_hub import hf_hub_download
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for local photorealistic generation"
        )

    dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    checkpoint_path = hf_hub_download(
        repo_id=base_model_id,
        filename=BASE_MODEL_FILE,
    )

    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        checkpoint_path,
        config=BASE_CONFIG_ID,
        vae=vae,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipeline.enable_attention_slicing("max")
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()

    _PIPELINE = pipeline
    return _PIPELINE


def unload_pipeline():
    """Release cached model references and CUDA cache where possible."""
    global _PIPELINE

    _PIPELINE = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def generate_photorealistic(
    reference_path,
    analysis,
    output_dir,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    strength=DEFAULT_STRENGTH,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    control_scale=DEFAULT_CONTROL_SCALE,
    low_threshold=100,
    high_threshold=200,
    pipeline=None,
):
    """Generate one auditable portrait from a confirmed reference."""
    _validate_settings(
        seed,
        steps,
        strength,
        guidance_scale,
        control_scale,
    )

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    portrait_path = output_directory / f"portrait_seed_{seed}.png"
    control_path = output_directory / f"control_seed_{seed}.png"
    metadata_path = output_directory / f"generation_seed_{seed}.json"

    for path in (portrait_path, control_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Generation output already exists: {path}")

    reference = prepare_reference_image(reference_path)
    control = create_canny_control(
        reference,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    active_pipeline = pipeline or load_pipeline()
    tokenizer = getattr(active_pipeline, "tokenizer", None)
    prompt, negative_prompt = build_prompts(
        analysis,
        tokenizer=tokenizer,
    )
    prompt_budget = {
        "mode": (
            "clip_tokenizer"
            if tokenizer is not None
            else "fallback_word_budget"
        ),
        "maximum_tokens": (
            _token_budget(tokenizer)
            if tokenizer is not None
            else None
        ),
        "prompt_tokens": (
            _token_count(tokenizer, prompt)
            if tokenizer is not None
            else None
        ),
        "negative_prompt_tokens": (
            _token_count(tokenizer, negative_prompt)
            if tokenizer is not None
            else None
        ),
        "critical_constraints_protected": True,
    }

    import torch

    random_generator = torch.Generator(device="cpu").manual_seed(seed)

    result = active_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=reference,
        control_image=control,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(control_scale),
        generator=random_generator,
        num_images_per_prompt=1,
    )

    if not getattr(result, "images", None):
        raise RuntimeError("The diffusion pipeline returned no images")

    portrait = result.images[0].convert("RGB")
    portrait.save(portrait_path)
    control.save(control_path)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_reference": str(Path(reference_path)),
        "portrait": str(portrait_path),
        "control_image": str(control_path),
        "base_model": BASE_MODEL_ID,
        "base_model_file": BASE_MODEL_FILE,
        "vae_model": VAE_MODEL_ID,
        "controlnet_model": CONTROLNET_MODEL_ID,
        "seed": seed,
        "settings": {
            "width": DEFAULT_SIZE,
            "height": DEFAULT_SIZE,
            "steps": int(steps),
            "strength": float(strength),
            "guidance_scale": float(guidance_scale),
            "control_scale": float(control_scale),
            "canny_low_threshold": low_threshold,
            "canny_high_threshold": high_threshold,
        },
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_budget": prompt_budget,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata["metadata_file"] = str(metadata_path)
    return metadata
