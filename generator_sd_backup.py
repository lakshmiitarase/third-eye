import os
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler

from description_parser import extract_features
from feature_mapping import get_feature_path


# ============================================================
# THIRD EYE
# AI-BASED FORENSIC FACE GENERATION SYSTEM
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Absolute path of the current project folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ------------------------------------------------------------
# Stable Diffusion model
# ------------------------------------------------------------

MODEL_PATH = (
    r"C:\Users\sumee\.cache\huggingface\hub"
    r"\models--SG161222--Realistic_Vision_V6.0_B1_noVAE"
    r"\snapshots\9a857a696b9aabbf509073e0aa55ec8200b6ef7d"
)


# ------------------------------------------------------------
# IP-Adapter
# ------------------------------------------------------------

IP_ADAPTER_PATH = (
    r"C:\Users\sumee\.cache\huggingface\hub"
    r"\models--h94--IP-Adapter"
    r"\snapshots\018e402774aeeddd60609b4ecdb7e298259dc729"
    r"\models"
)


# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------

OUTPUT_DIR = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "static",
        "outputs"
    )
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
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
print("Project folder:", BASE_DIR)
print("Output folder:", OUTPUT_DIR)
print("======================================")


# ============================================================
# 3. CHECK MODEL PATHS
# ============================================================

if not os.path.isdir(MODEL_PATH):

    raise FileNotFoundError(
        f"Stable Diffusion model folder not found:\n{MODEL_PATH}"
    )


if not os.path.isdir(IP_ADAPTER_PATH):

    print("\nWARNING:")
    print("IP-Adapter folder not found:")
    print(IP_ADAPTER_PATH)
    print("The system will continue without IP-Adapter.")


# ============================================================
# 4. LOAD STABLE DIFFUSION
# ============================================================

print("\nLoading Stable Diffusion...")
print("Please wait...")


scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH,
    subfolder="scheduler"
)


pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    scheduler=scheduler,
    torch_dtype=DTYPE,
    safety_checker=None
)


pipe = pipe.to(DEVICE)


# Reduce CPU/RAM usage
pipe.enable_attention_slicing()

try:

    pipe.enable_vae_slicing()

except Exception:

    pass


print("Stable Diffusion loaded successfully.")


# ============================================================
# 5. LOAD IP-ADAPTER
# ============================================================

print("\nLoading IP-Adapter...")


IP_ADAPTER_ENABLED = False


if os.path.isdir(IP_ADAPTER_PATH):

    try:

        pipe.load_ip_adapter(
            IP_ADAPTER_PATH,
            subfolder=".",
            weight_name="ip-adapter-plus_sd15.bin"
        )

        pipe.set_ip_adapter_scale(0.45)

        IP_ADAPTER_ENABLED = True

        print("IP-Adapter loaded successfully.")

    except Exception as e:

        print("IP-Adapter could not be loaded.")
        print("Reason:", e)
        print("Continuing with text-based generation.")

else:

    print("IP-Adapter folder not available.")
    print("Continuing with text-based generation.")


# ============================================================
# 6. DETECT GENDER
# ============================================================

def detect_gender(description):

    text = description.lower()

    female_words = [
        "female",
        "woman",
        "girl",
        "lady"
    ]

    male_words = [
        "male",
        "man",
        "boy",
        "gentleman"
    ]

    if any(word in text for word in female_words):

        return "female"

    if any(word in text for word in male_words):

        return "male"

    return "person"


# ============================================================
# 7. CHECK DATABASE IMAGE
# ============================================================

def check_image(path, feature_name):

    if path is None:

        print(
            f"✗ {feature_name}: "
            "No database image was mapped."
        )

        return None


    path = os.path.normpath(path)


    if not os.path.isfile(path):

        print(
            f"✗ {feature_name}: "
            "Database image NOT found."
        )

        print("  Path:", path)

        return None


    print(
        f"✓ {feature_name}: "
        "Database image found."
    )

    print("  Path:", path)

    return path


# ============================================================
# 8. LOAD REFERENCE IMAGE
# ============================================================

def load_reference_image(path):

    if path is None:

        return None


    try:

        image = Image.open(path).convert("RGB")

        print(
            "Reference image size:",
            image.size
        )

        return image


    except Exception as e:

        print(
            "Could not load reference image."
        )

        print("Reason:", e)

        return None


# ============================================================
# 9. GENERATE ONE COMPLETE FACE
# ============================================================

def generate_face(description):

    print("\n")
    print("======================================")
    print("          GENERATING FACE")
    print("======================================")


    if not description or not description.strip():

        raise ValueError(
            "Description is empty."
        )


    print("\nDescription:")
    print(description)


    # ========================================================
    # STEP 1
    # EXTRACT FEATURES
    # ========================================================

    print("\n========== EXTRACTED FEATURES ==========")


    features = extract_features(description)


    if not features:

        raise ValueError(
            "No features could be extracted "
            "from the description."
        )


    for key, value in features.items():

        print(
            f"{key}: {value}"
        )


    # ========================================================
    # STEP 2
    # DETECT GENDER
    # ========================================================

    gender = detect_gender(description)

    print("\ngender:", gender)


    # ========================================================
    # STEP 3
    # FIND DATABASE IMAGES
    # ========================================================

    print("\n")
    print("========== DATABASE FEATURE SELECTION ==========")


    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face_path = get_feature_path(
        "face_shapes",
        features.get("face")
    )


    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    eyes_path = get_feature_path(
        "eyes",
        features.get("eyes")
    )


    # --------------------------------------------------------
    # NOSE
    # --------------------------------------------------------

    nose_path = get_feature_path(
        "nose",
        features.get("nose")
    )


    # --------------------------------------------------------
    # EYEBROWS
    # --------------------------------------------------------

    eyebrows_path = get_feature_path(
        "eyebrows",
        features.get("eyebrows")
    )


    # ========================================================
    # STEP 4
    # CHECK DATABASE FILES
    # ========================================================

    print("\n========== SELECTED DATABASE IMAGES ==========")


    face_path = check_image(
        face_path,
        "Face"
    )


    eyes_path = check_image(
        eyes_path,
        "Eyes"
    )


    nose_path = check_image(
        nose_path,
        "Nose"
    )


    eyebrows_path = check_image(
        eyebrows_path,
        "Eyebrows"
    )


    # ========================================================
    # STEP 5
    # DISPLAY SELECTED FEATURES
    # ========================================================

    print("\n")
    print("======================================")
    print("SELECTED DATABASE FEATURES")
    print("======================================")


    print("Face:")
    print(face_path)


    print("\nEyes:")
    print(eyes_path)


    print("\nNose:")
    print(nose_path)


    print("\nEyebrows:")
    print(eyebrows_path)


    # ========================================================
    # STEP 6
    # TEXT FEATURES
    # ========================================================

    age = features.get(
        "age_group"
    ) or "young adult"


    face = features.get(
        "face"
    ) or "natural"


    eyes = features.get(
        "eyes"
    ) or "natural eyes"


    nose = features.get(
        "nose"
    ) or "natural nose"


    eyebrows = features.get(
        "eyebrows"
    ) or "natural eyebrows"


    lips = features.get(
        "lips"
    ) or "natural lips"


    hair = features.get(
        "hair"
    ) or "natural hair"


    skin = features.get(
        "skin_tone"
    ) or "natural skin tone"


    beard = features.get(
        "beard"
    ) or "no beard"


    moustache = features.get(
        "moustache"
    ) or "no moustache"


    # ========================================================
    # STEP 7
    # CREATE ONE-FACE PROMPT
    # ========================================================

    prompt = f"""
photorealistic forensic portrait photograph
of one {age} {gender},

ONE PERSON ONLY,
ONE COMPLETE HUMAN FACE,
ONE HEAD,

front-facing portrait,
looking directly at the camera,
head and shoulders,

{face} face shape,
{eyes},
{nose},
{eyebrows},
{lips},
{hair},
{skin},
{beard},
{moustache},

natural human facial proportions,
realistic facial anatomy,
realistic skin texture,
realistic eyes,
realistic nose,
realistic eyebrows,
realistic lips,
realistic hair,

neutral facial expression,
symmetrical face,
centered face,
full visible face,
both eyes visible,
complete nose visible,
complete mouth visible,

natural studio lighting,
realistic photography,
high facial detail,
photorealistic,
sharp focus
"""


    # ========================================================
    # STEP 8
    # NEGATIVE PROMPT
    # ========================================================

    negative_prompt = """
collage,
photo collage,
image collage,
2x2 collage,
four panels,
three panels,
two panels,
grid,
image grid,
split image,
split screen,
divided image,
four images,
multiple images,

feature sheet,
feature board,
reference sheet,
facial feature chart,

separate face parts,
isolated eyes,
isolated nose,
isolated eyebrows,
isolated mouth,
isolated lips,
isolated moustache,

multiple people,
multiple persons,
multiple faces,
two faces,
three faces,
four faces,
duplicate face,

extra eyes,
extra nose,
extra mouth,
extra eyebrows,
extra ears,
extra facial features,

deformed face,
distorted face,
disfigured face,
malformed face,
bad facial anatomy,
unnatural proportions,

cropped face,
cut face,
partial face,
missing face,

side profile,
profile face,
looking away,

cartoon,
anime,
drawing,
illustration,
painting,
sketch,
3d model,
3d render,

text,
labels,
watermark,

blurry,
low quality,
low resolution
"""


    # ========================================================
    # STEP 9
    # LOAD FACE DATABASE REFERENCE
    # ========================================================

    face_reference = None


    if face_path is not None:

        print("\n")
        print(
            "Loading face-shape database reference..."
        )

        face_reference = load_reference_image(
            face_path
        )


    # ========================================================
    # STEP 10
    # GENERATE ONE COMPLETE FACE
    # ========================================================

    print("\n")
    print("======================================")
    print("Generating ONE complete face...")
    print("Please wait...")
    print("======================================")


    try:

        if (
            IP_ADAPTER_ENABLED
            and face_reference is not None
        ):

            print(
                "Using face database image "
                "+ text description."
            )


            result = pipe(

                prompt=prompt,

                negative_prompt=negative_prompt,

                ip_adapter_image=face_reference,

                num_inference_steps=20,

                guidance_scale=7.0,

                height=512,

                width=512

            )


        else:

            print(
                "Using text description only."
            )


            result = pipe(

                prompt=prompt,

                negative_prompt=negative_prompt,

                num_inference_steps=20,

                guidance_scale=7.0,

                height=512,

                width=512

            )


    except Exception as e:

        print("\n")
        print("======================================")
        print("GENERATION ERROR")
        print("======================================")

        print(e)

        raise


    # ========================================================
    # STEP 11
    # TAKE ONLY FIRST GENERATED IMAGE
    # ========================================================

    generated_image = result.images[0]


    # ========================================================
    # STEP 12
    # SAVE OUTPUT SAFELY
    # ========================================================

    output_path = os.path.normpath(
        os.path.join(
            OUTPUT_DIR,
            "generated_face.png"
        )
    )


    generated_image.save(
        output_path
    )


    # Make sure file was actually created

    if not os.path.isfile(output_path):

        raise FileNotFoundError(
            "Generated image was not saved:\n"
            + output_path
        )


    # ========================================================
    # STEP 13
    # SUCCESS
    # ========================================================

    print("\n")
    print("======================================")
    print("       GENERATION SUCCESSFUL")
    print("======================================")


    print(
        "ONE complete face generated."
    )


    print("\nOutput:")
    print(output_path)


    print("\nFile exists:")
    print(os.path.isfile(output_path))


    print("======================================")


    # IMPORTANT:
    # Flask should receive a web-relative path,
    # NOT the Windows absolute path.

    return "outputs/generated_face.png"


# ============================================================
# 14. DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_description = """
    Male, young adult, square face,
    narrow eyes, long straight nose,
    straight eyebrows, medium lips,
    black short hair, medium brown skin,
    light stubble, thick moustache,
    front-facing face, neutral expression.
    """


    try:

        final_image = generate_face(
            test_description
        )


        print("\n")
        print("======================================")
        print("FINAL IMAGE PATH")
        print("======================================")

        print(final_image)

        print("======================================")


    except Exception as e:

        print("\n")
        print("======================================")
        print("GENERATION FAILED")
        print("======================================")

        print(e)