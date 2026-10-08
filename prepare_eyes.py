import os
import cv2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "features",
    "eyes",
    "narrow eye.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "processed_features",
    "eyes"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(INPUT_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not find image:\n{INPUT_PATH}"
    )

height, width = image.shape[:2]

print("Original size:", width, "x", height)


# ============================================================
# LEFT EYE
# ============================================================
#
# Keep the same eye crop.
# Only reduce the empty/front space on the LEFT side.
#

left_x1 = int(width * 0.08)
left_x2 = int(width * 0.58)


# ============================================================
# RIGHT EYE
# ============================================================
#
# Keep the same eye crop.
# Only reduce the empty/front space on the RIGHT side.
#

right_x1 = int(width * 0.42)
right_x2 = int(width * 0.92)


# ============================================================
# VERTICAL CROP
# ============================================================
#
# DON'T change the top/bottom area.
#

y1 = int(height * 0.10)
y2 = int(height * 0.90)


# ============================================================
# CROP
# ============================================================

left_eye = image[
    y1:y2,
    left_x1:left_x2
]

right_eye = image[
    y1:y2,
    right_x1:right_x2
]


# ============================================================
# SAVE
# ============================================================

left_path = os.path.join(
    OUTPUT_DIR,
    "narrow_left.png"
)

right_path = os.path.join(
    OUTPUT_DIR,
    "narrow_right.png"
)

cv2.imwrite(left_path, left_eye)
cv2.imwrite(right_path, right_eye)


# ============================================================
# RESULT
# ============================================================

print()
print("======================================")
print("EYES CROPPED SUCCESSFULLY")
print("======================================")

print()
print("LEFT:")
print(left_path)

print()
print("RIGHT:")
print(right_path)