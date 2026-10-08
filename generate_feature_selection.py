from description_parser import extract_features
from feature_mapping import get_feature_path


# ---------------------------------------------------------
# WITNESS DESCRIPTION
# ---------------------------------------------------------

description = """
Male with a square face,
narrow eyes,
a long nose,
and straight eyebrows.
"""

# ---------------------------------------------------------
# 1. EXTRACT FEATURES
# ---------------------------------------------------------

features = extract_features(description)


print("\n========== EXTRACTED FEATURES ==========")

print("Face:", features["face"])
print("Eyes:", features["eyes"])
print("Nose:", features["nose"])
print("Eyebrows:", features["eyebrows"])


# ---------------------------------------------------------
# 2. FIND CORRESPONDING IMAGE PATHS
# ---------------------------------------------------------

print("\n========== SELECTED IMAGES ==========")


# ---------------------------------------------------------
# FACE
# ---------------------------------------------------------

if features["face"]:

    face_path = get_feature_path(
        "face_shapes",
        features["face"]
    )

    print("Face image:", face_path)


# ---------------------------------------------------------
# EYES
# ---------------------------------------------------------

if features["eyes"]:

    eye_path = get_feature_path(
        "eyes",
        features["eyes"]
    )

    print("Eye image:", eye_path)


# ---------------------------------------------------------
# NOSE
# ---------------------------------------------------------

if features["nose"]:

    nose_path = get_feature_path(
        "nose",
        features["nose"]
    )

    print("Nose image:", nose_path)


# ---------------------------------------------------------
# EYEBROWS
# ---------------------------------------------------------

if features["eyebrows"]:

    eyebrow_path = get_feature_path(
        "eyebrows",
        features["eyebrows"]
    )

    print("Eyebrow image:", eyebrow_path)