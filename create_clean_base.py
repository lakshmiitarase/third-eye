import cv2
import numpy as np
import os


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INPUT_IMAGE = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "features",
    "face shapes",
    "base_male.png"
)

OUTPUT_IMAGE = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "features",
    "face shapes",
    "clean_base_male.png"
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(INPUT_IMAGE)

if image is None:
    print("ERROR: base_male.png not found")
    print(INPUT_IMAGE)
    exit()

print("Loaded:", INPUT_IMAGE)

height, width = image.shape[:2]

print("Image size:", width, "x", height)


# =========================================================
# CREATE MASK
# =========================================================

mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


# =========================================================
# EYES
# =========================================================

cv2.ellipse(
    mask,
    (190, 181),
    (38, 25),
    0,
    0,
    360,
    255,
    -1
)

cv2.ellipse(
    mask,
    (285, 181),
    (38, 25),
    0,
    0,
    360,
    255,
    -1
)


# =========================================================
# EYEBROWS
# =========================================================

cv2.ellipse(
    mask,
    (190, 155),
    (45, 15),
    0,
    0,
    360,
    255,
    -1
)

cv2.ellipse(
    mask,
    (285, 155),
    (45, 15),
    0,
    0,
    360,
    255,
    -1
)


# =========================================================
# NOSE
# =========================================================

nose_points = np.array([
    [238, 195],
    [270, 195],
    [282, 225],
    [285, 260],
    [290, 295],
    [275, 315],
    [250, 315],
    [225, 295],
    [230, 260],
    [232, 225]
], np.int32)

cv2.fillPoly(
    mask,
    [nose_points],
    255
)


# =========================================================
# LIPS / MOUTH
# =========================================================

mouth_points = np.array([
    [220, 315],
    [245, 308],
    [270, 308],
    [300, 315],
    [305, 335],
    [290, 350],
    [260, 355],
    [230, 350],
    [215, 335]
], np.int32)

cv2.fillPoly(
    mask,
    [mouth_points],
    255
)


# =========================================================
# EXPAND MASK
# =========================================================

kernel = np.ones(
    (9, 9),
    np.uint8
)

mask = cv2.dilate(
    mask,
    kernel,
    iterations=1
)


# =========================================================
# FEATHER MASK
# =========================================================

mask = cv2.GaussianBlur(
    mask,
    (15, 15),
    0
)


# =========================================================
# INPAINT
# =========================================================

print("Removing existing facial features...")

clean_image = cv2.inpaint(
    image,
    mask,
    7,
    cv2.INPAINT_TELEA
)


# =========================================================
# SAVE
# =========================================================

success = cv2.imwrite(
    OUTPUT_IMAGE,
    clean_image
)

if success:
    print()
    print("======================================")
    print("CLEAN BASE CREATED")
    print("======================================")
    print()
    print(OUTPUT_IMAGE)
else:
    print("ERROR: Could not save clean_base_male.png")