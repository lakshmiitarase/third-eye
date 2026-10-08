import os

from feature_mapping import get_feature_path


# =========================================================
# BASE PATH
# =========================================================

BASE_PATH = os.path.join(
    "static",
    "assets",
    "features"
)


# =========================================================
# SELECT FEATURE
# =========================================================

def select_feature(category, feature):
    """
    Select the exact feature image from the local
    feature database using feature_mapping.py.

    No random selection is used.
    """

    if not category:
        print("Category is empty.")
        return None

    if not feature:
        print("Feature is empty.")
        return None

    category = category.lower().strip()
    feature = feature.lower().strip()

    print("\n===================================")
    print("Category:", category)
    print("Requested feature:", feature)
    print("===================================")

    # -----------------------------------------------------
    # Get exact mapped image
    # -----------------------------------------------------

    selected_path = get_feature_path(
        category,
        feature
    )

    # -----------------------------------------------------
    # Mapping not found
    # -----------------------------------------------------

    if selected_path is None:

        print(
            "No mapped image found for:",
            category,
            "->",
            feature
        )

        return None

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not os.path.isfile(selected_path):

        print(
            "Image file does not exist:",
            selected_path
        )

        return None

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print(
        "Selected image:",
        selected_path
    )

    return selected_path


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\nTesting feature selection...\n")

    face = select_feature(
        "face_shapes",
        "square"
    )

    eyes = select_feature(
        "eyes",
        "narrow eyes"
    )

    nose = select_feature(
        "nose",
        "long straight nose"
    )

    eyebrows = select_feature(
        "eyebrows",
        "straight eyebrows"
    )

    print("\n")
    print("========== FINAL RESULTS ==========")

    print("Face:", face)
    print("Eyes:", eyes)
    print("Nose:", nose)
    print("Eyebrows:", eyebrows)

    print("===================================")