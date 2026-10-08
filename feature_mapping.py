import os


# =========================================================
# BASE FEATURE DIRECTORY
# =========================================================

BASE_PATH = os.path.join(
    "static",
    "assets",
    "features"
)


# =========================================================
# FEATURE DATABASE
# =========================================================

FEATURES = {

    # -----------------------------------------------------
    # FACE SHAPES
    # -----------------------------------------------------

    "face_shapes": {

        "oval": "oval shape.jpg",
        "round": "round.jpg",
        "square": "square.jpg",

        # Add these only if the corresponding files
        # actually exist in your face shapes folder.
        #
        # "heart": "heart.jpg",
        # "oblong": "oblong.jpg",
        # "long": "long.jpg",
    },


    # -----------------------------------------------------
    # EYES
    # -----------------------------------------------------

    "eyes": {

        # Natural feature names
        "large eyes": "eye6.png",
        "small eyes": "eye4.png",
        "round eyes": "eye11.png",
        "almond eyes": "eye3.png",
        "narrow eyes": "eye5.png",
        "hooded eyes": "eye14.png",

        # Direct image names
        "eye1": "eye1.png",
        "eye2": "eye2.png",
        "eye3": "eye3.png",
        "eye4": "eye4.png",
        "eye5": "eye5.png",
        "eye6": "eye6.png",
        "eye7": "eye7.png",
        "eye8": "eye8.png",
        "eye9": "eye9.png",
        "eye10": "eye10.png",
        "eye11": "eye11.png",
        "eye12": "eye12.png",
        "eye13": "eye13.png",
        "eye14": "eye14.png",
        "eye15": "eye15.png",
    },


    # -----------------------------------------------------
    # NOSE
    # -----------------------------------------------------

    "nose": {

        # Natural feature names
        "straight nose": "nose_02.png",
        "small nose": "nose_03.png",
        "narrow nose": "nose_02.png",
        "broad nose": "nose_06.png",
        "wide nose": "nose_06.png",
        "rounded nose": "nose_07.png",
        "long nose": "nose_11.png",
        "long straight nose": "nose_11.png",
        "upturned nose": "nose_14.png",
        "aquiline nose": "nose_15.png",

        # Direct image names
        "nose1": "nose_01.png",
        "nose2": "nose_02.png",
        "nose3": "nose_03.png",
        "nose4": "nose_04.png",
        "nose5": "nose_05.png",
        "nose6": "nose_06.png",
        "nose7": "nose_07.png",
        "nose8": "nose_08.png",
        "nose9": "nose_09.png",
        "nose10": "nose_10.png",
        "nose11": "nose_11.png",
        "nose12": "nose_12.png",
        "nose13": "nose_13.png",
        "nose14": "nose_14.png",
        "nose15": "nose_15.png",
    },


    # -----------------------------------------------------
    # EYEBROWS
    # -----------------------------------------------------

    "eyebrows": {

        "arched eyebrows": "arched eyebrow.png",
        "common eyebrows": "common eyebrow.png",
        "hard angled eyebrows": "hard angled.jpg",
        "high arch eyebrows": "high arch.jpg",
        "oval eyebrows": "oval eyebrow.png",
        "round eyebrows": "round eyebrow.jpg",
        "short eyebrows": "short eyebrow.jpg",
        "soft angled eyebrows": "soft angled eyebrow.png",
        "straight eyebrows": "straight eyebrows.png",
        "thin eyebrows": "thin eyebrows.png",
    }
}


# =========================================================
# GET FEATURE IMAGE PATH
# =========================================================

def get_feature_path(category, feature):

    if not category or not feature:
        return None

    category = category.lower().strip()
    feature = feature.lower().strip()

    # -----------------------------------------------------
    # Check category
    # -----------------------------------------------------

    if category not in FEATURES:

        print(
            f"Unknown feature category: {category}"
        )

        return None

    # -----------------------------------------------------
    # Check feature
    # -----------------------------------------------------

    if feature not in FEATURES[category]:

        print(
            f"Feature not mapped: {feature}"
        )

        return None

    # -----------------------------------------------------
    # Get filename
    # -----------------------------------------------------

    filename = FEATURES[category][feature]

    # -----------------------------------------------------
    # Convert category name to actual folder name
    # -----------------------------------------------------

    folder_name = category.replace(
        "_",
        " "
    )

    # -----------------------------------------------------
    # Build complete path
    # -----------------------------------------------------

    path = os.path.join(
        BASE_PATH,
        folder_name,
        filename
    )

    # -----------------------------------------------------
    # Verify file exists
    # -----------------------------------------------------

    if not os.path.exists(path):

        print(
            "Mapped image does not exist:",
            path
        )

        return None

    return path