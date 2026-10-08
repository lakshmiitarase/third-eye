import os
import cv2
import numpy as np
import torch

from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler


# ============================================================
# THIRD-EYE REALISTIC FACE GENERATOR
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "features"
)

PROCESSED_FEATURE_DIR = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "processed_features"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--SG161222--Realistic_Vision_V6.0_B1_noVAE",
    "snapshots",
    "9a857a696b9aabbf509073e0aa55ec8200b6ef7d"
)

IP_ADAPTER_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--h94--IP-Adapter",
    "snapshots",
    "018e402774aeeddd60609b4ecdb7e298259dc729",
    "models"
)

IP_ADAPTER_WEIGHT = "ip-adapter-plus_sd15.bin"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print("======================================")
print("Third-Eye Face Generation System")
print("======================================")
print("Device:", DEVICE)


# ============================================================
# LOAD STABLE DIFFUSION
# ============================================================

print()
print("Loading Stable Diffusion...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    safety_checker=None,
    local_files_only=True
)

pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)

pipe = pipe.to(DEVICE)

print("Stable Diffusion loaded.")


# ============================================================
# LOAD IP-ADAPTER
# ============================================================

print()
print("Loading IP-Adapter...")

pipe.load_ip_adapter(
    IP_ADAPTER_PATH,
    subfolder="",
    weight_name=IP_ADAPTER_WEIGHT
)

pipe.set_ip_adapter_scale(0.30)

print("IP-Adapter loaded.")


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_rgba(path):

    if not os.path.exists(path):
        print("FILE NOT FOUND:", path)
        return None

    image = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        print("COULD NOT LOAD:", path)
        return None

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGRA
        )

    elif image.shape[2] == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2BGRA
        )

    return image


def resize_keep_ratio(image, width):

    h, w = image.shape[:2]

    if w == 0:
        return image

    scale = width / w

    new_h = max(
        1,
        int(h * scale)
    )

    return cv2.resize(
        image,
        (width, new_h),
        interpolation=cv2.INTER_AREA
    )


def alpha_overlay(
    background,
    foreground,
    x,
    y
):

    bg = background.copy()

    if foreground is None:
        return bg

    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)

    if x1 >= x2 or y1 >= y2:
        return bg

    fx1 = x1 - x
    fy1 = y1 - y

    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    fg = foreground[
        fy1:fy2,
        fx1:fx2
    ]

    bg_part = bg[
        y1:y2,
        x1:x2
    ]

    alpha = (
        fg[:, :, 3]
        .astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, None]

    result = (
        fg[:, :, :3].astype(np.float32)
        * alpha
        +
        bg_part[:, :, :3].astype(np.float32)
        * (1.0 - alpha)
    )

    bg_part[:, :, :3] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    bg[
        y1:y2,
        x1:x2
    ] = bg_part

    return bg


# ============================================================
# FIND FEATURE FILE
# ============================================================

def find_feature(
    root,
    keywords
):

    if not os.path.exists(root):
        return None

    matches = []

    for current_root, dirs, files in os.walk(root):

        for filename in files:

            lower = filename.lower()

            if not lower.endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            score = 0

            for word in keywords:

                if word is None:
                    continue

                word = str(word).strip()

                if not word:
                    continue

                if word.lower() in lower:
                    score += 1

            if score > 0:

                matches.append(
                    (
                        score,
                        os.path.join(
                            current_root,
                            filename
                        )
                    )
                )

    if not matches:
        return None

    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return matches[0][1]


# ============================================================
# BASE FACE
# ============================================================

def get_base_face(
    face,
    gender
):

    # --------------------------------------------------------
    # CURRENT AVAILABLE BASE FACE
    # --------------------------------------------------------

    if face == "square" and gender == "male":

        path = os.path.join(
            FEATURE_DIR,
            "face shapes",
            "blank_square_male.png"
        )

        if os.path.exists(path):

            print(
                "Base face:",
                path
            )

            return load_rgba(path)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    # For other gender/face combinations, use a neutral
    # reference canvas and let the text prompt control
    # gender and face shape.
    # --------------------------------------------------------

    fallback_path = os.path.join(
        OUTPUT_DIR,
        "blank_reference.png"
    )

    blank = Image.new(
        "RGBA",
        (512, 512),
        (235, 235, 235, 255)
    )

    blank.save(
        fallback_path
    )

    print(
        "Using fallback reference:",
        fallback_path
    )

    return load_rgba(
        fallback_path
    )


# ============================================================
# BUILD REFERENCE IMAGE
# ============================================================

def build_reference(
    attributes
):

    face = attributes.get(
        "face"
    )

    gender = attributes.get(
        "gender"
    )

    eyes = attributes.get(
        "eyes"
    )

    eyebrows = attributes.get(
        "eyebrows"
    )

    nose = attributes.get(
        "nose"
    )

    lips = attributes.get(
        "lips"
    )

    print()
    print("======================================")
    print("REFERENCE FEATURES")
    print("======================================")
    print("Face      :", face)
    print("Gender    :", gender)
    print("Eyes      :", eyes)
    print("Eyebrows  :", eyebrows)
    print("Nose      :", nose)
    print("Lips      :", lips)

    canvas = get_base_face(
        face,
        gender
    )

    if canvas is None:
        return None

    h, w = canvas.shape[:2]


    # ========================================================
    # EYES
    # ========================================================

    if eyes:

        eye_path = find_feature(
            PROCESSED_FEATURE_DIR,
            [
                eyes,
                "eye"
            ]
        )

        if eye_path:

            print(
                "Eye reference:",
                eye_path
            )

            eye = load_rgba(
                eye_path
            )

            if eye is not None:

                eye = resize_keep_ratio(
                    eye,
                    120
                )

                canvas = alpha_overlay(
                    canvas,
                    eye,
                    145,
                    285
                )

                canvas = alpha_overlay(
                    canvas,
                    eye,
                    335,
                    285
                )

        else:

            print(
                "WARNING: Eye reference not found."
            )

    else:

        print(
            "No eye feature specified."
        )


    # ========================================================
    # EYEBROWS
    # ========================================================

    if eyebrows:

        brow_path = find_feature(
            PROCESSED_FEATURE_DIR,
            [
                eyebrows,
                "brow",
                "eyebrow"
            ]
        )

        if brow_path:

            print(
                "Eyebrow reference:",
                brow_path
            )

            brow = load_rgba(
                brow_path
            )

            if brow is not None:

                brow = resize_keep_ratio(
                    brow,
                    125
                )

                canvas = alpha_overlay(
                    canvas,
                    brow,
                    140,
                    245
                )

                canvas = alpha_overlay(
                    canvas,
                    brow,
                    335,
                    245
                )

        else:

            print(
                "WARNING: Eyebrow reference not found."
            )

    else:

        print(
            "No eyebrow feature specified."
        )


    # ========================================================
    # NOSE
    # ========================================================

    if nose:

        nose_path = find_feature(
            FEATURE_DIR,
            [
                nose,
                "nose"
            ]
        )

        if nose_path:

            print(
                "Nose reference:",
                nose_path
            )

            nose_img = load_rgba(
                nose_path
            )

            if nose_img is not None:

                nose_img = resize_keep_ratio(
                    nose_img,
                    100
                )

                nose_x = (
                    w // 2
                    -
                    nose_img.shape[1] // 2
                )

                nose_y = 330

                canvas = alpha_overlay(
                    canvas,
                    nose_img,
                    nose_x,
                    nose_y
                )

        else:

            print(
                "WARNING: Nose reference not found."
            )

    else:

        print(
            "No nose feature specified."
        )


    # ========================================================
    # LIPS
    # ========================================================

    if lips:

        lips_path = find_feature(
            FEATURE_DIR,
            [
                lips,
                "lip"
            ]
        )

        if lips_path:

            print(
                "Lips reference:",
                lips_path
            )

            lips_img = load_rgba(
                lips_path
            )

            if lips_img is not None:

                lips_img = resize_keep_ratio(
                    lips_img,
                    110
                )

                lips_x = (
                    w // 2
                    -
                    lips_img.shape[1] // 2
                )

                lips_y = 485

                canvas = alpha_overlay(
                    canvas,
                    lips_img,
                    lips_x,
                    lips_y
                )

        else:

            print(
                "WARNING: Lips reference not found."
            )

    else:

        print(
            "No lips feature specified."
        )


    # ========================================================
    # SAVE DEBUG REFERENCE
    # ========================================================

    debug_path = os.path.join(
        OUTPUT_DIR,
        "debug_reference.png"
    )

    debug_image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGRA2BGR
    )

    cv2.imwrite(
        debug_path,
        debug_image
    )

    print()
    print(
        "Debug reference:",
        debug_path
    )


    # ========================================================
    # CONVERT TO PIL
    # ========================================================

    reference = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGRA2RGB
    )

    reference = Image.fromarray(
        reference
    )

    return reference


# ============================================================
# GENERATE REALISTIC FACE
# ============================================================

def generate_face(
    attributes
):

    print()
    print("======================================")
    print("THIRD-EYE REALISTIC FACE GENERATION")
    print("======================================")


    # ========================================================
    # GET ALL FEATURES
    # ========================================================

    gender = attributes.get(
        "gender"
    )

    age_group = attributes.get(
        "age_group"
    )

    face = attributes.get(
        "face"
    )

    eyes = attributes.get(
        "eyes"
    )

    eyebrows = attributes.get(
        "eyebrows"
    )

    nose = attributes.get(
        "nose"
    )

    lips = attributes.get(
        "lips"
    )

    hair = attributes.get(
        "hair"
    )

    skin_tone = attributes.get(
        "skin_tone"
    )

    beard = attributes.get(
        "beard"
    )

    moustache = attributes.get(
        "moustache"
    )


    # ========================================================
    # PRINT FEATURES
    # ========================================================

    print()
    print("======================================")
    print("FEATURES USED FOR GENERATION")
    print("======================================")

    print("Gender     :", gender)
    print("Age        :", age_group)
    print("Face       :", face)
    print("Eyes       :", eyes)
    print("Eyebrows   :", eyebrows)
    print("Nose       :", nose)
    print("Lips       :", lips)
    print("Hair       :", hair)
    print("Skin tone  :", skin_tone)
    print("Beard      :", beard)
    print("Moustache  :", moustache)

    print("======================================")


    # ========================================================
    # BUILD REFERENCE
    # ========================================================

    reference = build_reference(
        attributes
    )

    if reference is None:

        raise RuntimeError(
            "Could not build reference image."
        )


    # ========================================================
    # GENDER PROMPT
    # ========================================================

    if gender == "male":

        gender_text = (
            "clearly adult male, "
            "masculine facial structure"
        )

        opposite_gender = (
            "female face, woman, feminine face"
        )

    elif gender == "female":

        gender_text = (
            "clearly adult female, "
            "feminine facial structure"
        )

        opposite_gender = (
            "male face, man, masculine face, beard, moustache"
        )

    else:

        gender_text = "adult person"

        opposite_gender = ""


    # ========================================================
    # BUILD DESCRIPTION-BASED PROMPT
    # ========================================================

    prompt_parts = [
        "photorealistic portrait",
        gender_text
    ]


    if age_group:

        prompt_parts.append(
            age_group
        )


    if face:

        prompt_parts.append(
            f"{face} face shape"
        )


    if eyes:

        prompt_parts.append(
            eyes
        )


    if eyebrows:

        prompt_parts.append(
            eyebrows
        )


    if nose:

        prompt_parts.append(
            nose
        )


    if lips:

        prompt_parts.append(
            lips
        )


    if hair:

        prompt_parts.append(
            f"hair: {hair}"
        )


    if skin_tone:

        prompt_parts.append(
            skin_tone
        )


    if beard:

        prompt_parts.append(
            f"beard: {beard}"
        )


    if moustache:

        prompt_parts.append(
            f"moustache: {moustache}"
        )


    # ========================================================
    # GENERAL FACE REQUIREMENTS
    # ========================================================

    prompt_parts.extend([
        "front facing",
        "looking directly at camera",
        "centered face",
        "neutral expression",
        "realistic human face",
        "realistic skin texture",
        "symmetrical facial features",
        "passport style portrait",
        "studio lighting",
        "plain background"
    ])


    prompt = ", ".join(
        prompt_parts
    )


    # ========================================================
    # NEGATIVE PROMPT
    # ========================================================

    negative_parts = [
        "cartoon",
        "anime",
        "illustration",
        "painting",
        "drawing",
        "deformed face",
        "distorted face",
        "extra eyes",
        "extra nose",
        "extra mouth",
        "duplicate face",
        "blurry",
        "low quality",
        "side profile",
        "profile view",
        "multiple people"
    ]


    if opposite_gender:

        negative_parts.append(
            opposite_gender
        )


    # If no beard was requested,
    # discourage beard generation.

    if not beard:

        negative_parts.append(
            "beard"
        )


    # If no moustache was requested,
    # discourage moustache generation.

    if not moustache:

        negative_parts.append(
            "moustache"
        )

        negative_parts.append(
            "mustache"
        )


    negative_prompt = ", ".join(
        negative_parts
    )


    # ========================================================
    # SHOW PROMPT
    # ========================================================

    print()
    print("======================================")
    print("PROMPT SENT TO STABLE DIFFUSION")
    print("======================================")

    print(prompt)

    print()
    print("======================================")
    print("NEGATIVE PROMPT")
    print("======================================")

    print(negative_prompt)

    print()
    print("Generating realistic face...")


    # ========================================================
    # GENERATE
    # ========================================================

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=reference,
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5
    )

    generated = result.images[0]


    # ========================================================
    # RESIZE OUTPUT
    # ========================================================

    generated = generated.resize(
        (600, 800),
        Image.Resampling.LANCZOS
    )


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "generated_face.png"
    )

    generated.save(
        output_path
    )


    print()
    print("======================================")
    print("GENERATED FACE")
    print("======================================")

    print(output_path)

    print("======================================")


    return output_path


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_attributes = {

        "gender": "male",

        "age_group": "young adult",

        "face": "square",

        "eyes": "narrow eyes",

        "eyebrows": "straight eyebrows",

        "nose": "long nose",

        "lips": "medium lips",

        "hair": "short black hair",

        "skin_tone": "medium brown skin",

        "beard": "goatee",

        "moustache": None
    }


    output = generate_face(
        test_attributes
    )


    if output:

        print()
        print(
            "Generated face:",
            output
        )