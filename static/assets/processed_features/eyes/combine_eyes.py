import os
from PIL import Image


EYES_DIR = os.path.abspath(os.path.dirname(__file__))

LEFT_PATH = os.path.join(EYES_DIR, "narrow_left.png")
RIGHT_PATH = os.path.join(EYES_DIR, "narrow_right.png")

OUTPUT_PATH = os.path.join(
    EYES_DIR,
    "narrow_eyes_reference.png"
)


left = Image.open(LEFT_PATH).convert("RGBA")
right = Image.open(RIGHT_PATH).convert("RGBA")


# ---------------------------------------------------------
# Remove ONLY the inner/front face area
# ---------------------------------------------------------
#
# LEFT eye:
# remove some pixels from its RIGHT side
#
# RIGHT eye:
# remove some pixels from its LEFT side
#
# The eye itself is kept.
# ---------------------------------------------------------

left_trim = int(left.width * 0.42)

right_trim = int(right.width * 0.42)


left = left.crop(
    (
        0,
        0,
        left.width - left_trim,
        left.height
    )
)


right = right.crop(
    (
        right_trim,
        0,
        right.width,
        right.height
    )
)


# ---------------------------------------------------------
# Keep original vertical appearance
# ---------------------------------------------------------

target_height = max(left.height, right.height)

if left.height != target_height:
    left = left.resize(
        (
            int(left.width * target_height / left.height),
            target_height
        ),
        Image.Resampling.LANCZOS
    )

if right.height != target_height:
    right = right.resize(
        (
            int(right.width * target_height / right.height),
            target_height
        ),
        Image.Resampling.LANCZOS
    )


# ---------------------------------------------------------
# Gap between eyes
# ---------------------------------------------------------

gap = 20


# ---------------------------------------------------------
# Transparent canvas
# ---------------------------------------------------------

canvas = Image.new(
    "RGBA",
    (
        left.width + gap + right.width,
        target_height
    ),
    (0, 0, 0, 0)
)


# ---------------------------------------------------------
# Place eyes
# ---------------------------------------------------------

canvas.alpha_composite(left, (0, 0))

canvas.alpha_composite(
    right,
    (left.width + gap, 0)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

canvas.save(
    OUTPUT_PATH,
    "PNG"
)


print("======================================")
print("EYE REFERENCE CREATED")
print("======================================")
print(OUTPUT_PATH)