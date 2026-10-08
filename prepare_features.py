import os
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "features",
    "eyebrows",
    "straight eyebrows.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "assets",
    "processed_features",
    "eyebrows"
)

LEFT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "straight_left.png"
)

RIGHT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "straight_right.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(
    INPUT_FILE,
    cv2.IMREAD_COLOR
)

if image is None:
    raise FileNotFoundError(
        f"Image not found:\n{INPUT_FILE}"
    )

h, w = image.shape[:2]

print("Original image size:", w, "x", h)


# ============================================================
# CREATE DARK-HAIR MASK
# ============================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

value = hsv[:, :, 2]

mask = np.zeros(
    value.shape,
    dtype=np.uint8
)

# Keep dark eyebrow hairs
mask[value < 125] = 255


# ============================================================
# LIMIT TO EYEBROW AREA
# ============================================================

region_mask = np.zeros_like(mask)

# LEFT eyebrow region
cv2.rectangle(
    region_mask,
    (0, int(h * 0.05)),
    (int(w * 0.48), int(h * 0.72)),
    255,
    -1
)

# RIGHT eyebrow region
cv2.rectangle(
    region_mask,
    (int(w * 0.52), int(h * 0.05)),
    (w - 1, int(h * 0.72)),
    255,
    -1
)

mask = cv2.bitwise_and(
    mask,
    region_mask
)


# ============================================================
# REMOVE SMALL NOISE
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)


# ============================================================
# SLIGHTLY THICKEN EYEBROW HAIRS
# ============================================================

thicken_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2, 2)
)

mask = cv2.dilate(
    mask,
    thicken_kernel,
    iterations=1
)


# ============================================================
# CONNECT HAIRS
# ============================================================

connect_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    connect_kernel
)


# ============================================================
# REMOVE VERY SMALL COMPONENTS
# ============================================================

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask,
    connectivity=8
)

clean_mask = np.zeros_like(mask)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]

    if area > 80 and width > 20 and height > 3:
        clean_mask[labels == i] = 255


mask = clean_mask


# ============================================================
# FUNCTION TO CREATE TRANSPARENT CROPPED FEATURE
# ============================================================

def create_feature(
    original_image,
    feature_mask,
    output_path
):

    ys, xs = np.where(
        feature_mask > 0
    )

    if len(xs) == 0 or len(ys) == 0:
        print("No feature detected:", output_path)
        return

    # Small transparent margin
    margin = 5

    x1 = max(
        0,
        xs.min() - margin
    )

    y1 = max(
        0,
        ys.min() - margin
    )

    x2 = min(
        original_image.shape[1],
        xs.max() + margin + 1
    )

    y2 = min(
        original_image.shape[0],
        ys.max() + margin + 1
    )

    cropped_image = original_image[
        y1:y2,
        x1:x2
    ]

    cropped_mask = feature_mask[
        y1:y2,
        x1:x2
    ]

    # Convert to RGBA
    rgba = cv2.cvtColor(
        cropped_image,
        cv2.COLOR_BGR2BGRA
    )

    rgba[:, :, 3] = cropped_mask

    # Save
    cv2.imwrite(
        output_path,
        rgba
    )

    print(
        "Saved:",
        output_path
    )

    print(
        "Size:",
        rgba.shape[1],
        "x",
        rgba.shape[0]
    )


# ============================================================
# SPLIT LEFT AND RIGHT
# ============================================================

middle = w // 2

left_mask = mask.copy()

# Remove right half
left_mask[:, middle:] = 0


right_mask = mask.copy()

# Remove left half
right_mask[:, :middle] = 0


# ============================================================
# CREATE LEFT EYEBROW
# ============================================================

create_feature(
    image,
    left_mask,
    LEFT_OUTPUT
)


# ============================================================
# CREATE RIGHT EYEBROW
# ============================================================

create_feature(
    image,
    right_mask,
    RIGHT_OUTPUT
)


# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("EYEBROW PROCESSING COMPLETED")
print("========================================")
print()
print("LEFT:")
print(LEFT_OUTPUT)
print()
print("RIGHT:")
print(RIGHT_OUTPUT)