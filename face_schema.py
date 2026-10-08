"""Canonical facial-feature schema for the Third Eye project.

Unknown or unmentioned features use None.
"""

from copy import deepcopy

SCHEMA_VERSION = "1.0"


def feature_definition(label, allowed_values, control_method="component", required=False):
    """Create one facial-feature definition."""
    return {
        "label": label,
        "allowed_values": allowed_values,
        "control_method": control_method,
        "required": required,
    }


FEATURE_SCHEMA = {
    # General appearance
    "gender": feature_definition("Gender presentation", ["male", "female"], "prompt"),
    "age_group": feature_definition(
        "Age group",
        ["child", "teen", "young_adult", "adult", "middle_aged", "elderly"],
        "prompt",
    ),
    "skin_tone": feature_definition(
        "Skin tone",
        ["fair", "light", "light_brown", "medium_brown", "dark_brown", "dark"],
        "prompt",
    ),

    # Face
    "face_shape": feature_definition(
        "Face shape", ["oval", "round", "square", "heart", "oblong", "triangular"]
    ),
    "face_width": feature_definition("Face width", ["narrow", "medium", "broad"]),
    "jaw_shape": feature_definition("Jaw shape", ["soft", "rounded", "square", "angular"]),
    "chin_shape": feature_definition("Chin shape", ["rounded", "pointed", "square", "double"]),

    # Hair
    "hair_length": feature_definition(
        "Hair length", ["bald", "very_short", "short", "medium", "long"]
    ),
    "hair_texture": feature_definition(
        "Hair texture", ["straight", "wavy", "curly", "coily"]
    ),
    "hair_colour": feature_definition(
        "Hair colour",
        ["black", "dark_brown", "brown", "light_brown", "blonde", "red", "grey", "white"],
        "both",
    ),
    "hairline": feature_definition(
        "Hairline", ["low", "normal", "high", "receding", "widows_peak"], "both"
    ),

    # Eyes
    "eye_shape": feature_definition(
        "Eye shape", ["almond", "round", "narrow", "hooded", "upturned", "downturned"]
    ),
    "eye_size": feature_definition("Eye size", ["small", "medium", "large"]),
    "eye_spacing": feature_definition("Eye spacing", ["close_set", "average", "wide_set"]),
    "eye_colour": feature_definition(
        "Eye colour", ["black", "dark_brown", "brown", "hazel", "green", "blue", "grey"], "prompt"
    ),

    # Eyebrows
    "eyebrow_shape": feature_definition(
        "Eyebrow shape", ["straight", "rounded", "arched", "high_arch", "soft_angled", "hard_angled"]
    ),
    "eyebrow_thickness": feature_definition("Eyebrow thickness", ["thin", "medium", "thick"]),
    "eyebrow_spacing": feature_definition("Eyebrow spacing", ["separate", "close", "joined"]),

    # Nose
    "nose_width": feature_definition("Nose width", ["narrow", "medium", "broad"]),
    "nose_length": feature_definition("Nose length", ["short", "medium", "long"]),
    "nose_bridge": feature_definition("Nose bridge", ["low", "straight", "high", "aquiline"]),
    "nose_tip": feature_definition("Nose tip", ["pointed", "rounded", "upturned", "downturned"]),

    # Mouth and lips
    "lip_thickness": feature_definition("Lip thickness", ["thin", "medium", "full"]),
    "mouth_width": feature_definition("Mouth width", ["narrow", "medium", "wide"]),

    # Facial hair
    "beard_style": feature_definition(
        "Beard style",
        ["none", "stubble", "short_beard", "full_beard", "goatee", "chin_strap", "soul_patch", "van_dyke"],
        "both",
    ),
    "moustache_style": feature_definition(
        "Moustache style", ["none", "thin", "pencil", "thick", "handlebar"], "both"
    ),

    # Ears and neck
    "ear_size": feature_definition("Ear size", ["small", "medium", "large", "prominent"]),
    "neck_width": feature_definition("Neck width", ["narrow", "medium", "broad"]),

    # Accessories
    "glasses": feature_definition(
        "Glasses", ["none", "regular", "round", "rectangular", "rimless", "sunglasses"], "both"
    ),
}


DEFAULT_FEATURE_STATE = {
    "value": None,
    "confidence": 0.0,
    "source_text": "",
    "mentioned": False,
    "negated": False,
}


def create_empty_features():
    """Create a new empty state for every facial feature."""
    return {
        feature_name: deepcopy(DEFAULT_FEATURE_STATE)
        for feature_name in FEATURE_SCHEMA
    }


def create_empty_analysis(description=""):
    """Create an empty facial-description analysis."""
    return {
        "schema_version": SCHEMA_VERSION,
        "original_description": description,
        "features": create_empty_features(),
        "warnings": [],
        "conflicts": [],
        "missing_important_features": [],
    }


def get_allowed_values(feature_name):
    """Return the allowed values for one feature."""
    if feature_name not in FEATURE_SCHEMA:
        raise KeyError(f"Unknown facial feature: {feature_name}")
    return list(FEATURE_SCHEMA[feature_name]["allowed_values"])


def is_valid_feature_value(feature_name, value):
    """Check whether a feature value is valid."""
    if feature_name not in FEATURE_SCHEMA:
        return False
    if value is None:
        return True
    return value in FEATURE_SCHEMA[feature_name]["allowed_values"]


def set_feature(
    analysis,
    feature_name,
    value,
    confidence,
    source_text,
    mentioned=True,
    negated=False,
):
    """Safely store one detected facial feature."""
    if feature_name not in FEATURE_SCHEMA:
        raise ValueError(f"Unknown facial feature: {feature_name}")

    if not is_valid_feature_value(feature_name, value):
        allowed_values = get_allowed_values(feature_name)
        raise ValueError(
            f"Invalid value '{value}' for '{feature_name}'. "
            f"Allowed values: {allowed_values}"
        )

    if not isinstance(confidence, (int, float)):
        raise ValueError("Confidence must be a number.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")
    if not isinstance(source_text, str):
        raise ValueError("source_text must be a string.")

    analysis["features"][feature_name] = {
        "value": value,
        "confidence": float(confidence),
        "source_text": source_text,
        "mentioned": bool(mentioned),
        "negated": bool(negated),
    }


def validate_feature_state(feature_name, feature_state):
    """Validate one feature state and return a list of errors."""
    errors = []

    if feature_name not in FEATURE_SCHEMA:
        return [f"Unknown feature: {feature_name}"]
    if not isinstance(feature_state, dict):
        return [f"{feature_name} must be a dictionary."]

    required_keys = {"value", "confidence", "source_text", "mentioned", "negated"}
    missing_keys = required_keys - set(feature_state.keys())
    if missing_keys:
        return [f"{feature_name} is missing keys: {sorted(missing_keys)}"]

    value = feature_state["value"]
    confidence = feature_state["confidence"]
    source_text = feature_state["source_text"]
    mentioned = feature_state["mentioned"]
    negated = feature_state["negated"]

    if not is_valid_feature_value(feature_name, value):
        errors.append(f"Invalid value '{value}' for {feature_name}.")
    if not isinstance(confidence, (int, float)):
        errors.append(f"{feature_name}.confidence must be a number.")
    elif not 0.0 <= confidence <= 1.0:
        errors.append(f"{feature_name}.confidence must be between 0.0 and 1.0.")
    if not isinstance(source_text, str):
        errors.append(f"{feature_name}.source_text must be a string.")
    if not isinstance(mentioned, bool):
        errors.append(f"{feature_name}.mentioned must be Boolean.")
    if not isinstance(negated, bool):
        errors.append(f"{feature_name}.negated must be Boolean.")
    if value is None and confidence != 0.0:
        errors.append(f"{feature_name} has no value but confidence is not 0.0.")
    if mentioned is False and source_text:
        errors.append(
            f"{feature_name} is marked as not mentioned but contains source text."
        )

    return errors


def validate_analysis(analysis):
    """Validate a complete facial-description analysis."""
    errors = []

    if not isinstance(analysis, dict):
        return ["Analysis must be a dictionary."]

    required_keys = {
        "schema_version",
        "original_description",
        "features",
        "warnings",
        "conflicts",
        "missing_important_features",
    }
    missing_keys = required_keys - set(analysis.keys())
    if missing_keys:
        return [f"Analysis is missing keys: {sorted(missing_keys)}"]

    if analysis["schema_version"] != SCHEMA_VERSION:
        errors.append(f"Schema version must be {SCHEMA_VERSION}.")
    if not isinstance(analysis["original_description"], str):
        errors.append("original_description must be a string.")

    features = analysis["features"]
    if not isinstance(features, dict):
        return errors + ["features must be a dictionary."]

    missing_features = set(FEATURE_SCHEMA.keys()) - set(features.keys())
    if missing_features:
        errors.append(f"Missing facial features: {sorted(missing_features)}")

    unexpected_features = set(features.keys()) - set(FEATURE_SCHEMA.keys())
    if unexpected_features:
        errors.append(f"Unexpected facial features: {sorted(unexpected_features)}")

    for feature_name in FEATURE_SCHEMA:
        if feature_name in features:
            errors.extend(
                validate_feature_state(feature_name, features[feature_name])
            )

    if not isinstance(analysis["warnings"], list):
        errors.append("warnings must be a list.")
    if not isinstance(analysis["conflicts"], list):
        errors.append("conflicts must be a list.")
    if not isinstance(analysis["missing_important_features"], list):
        errors.append("missing_important_features must be a list.")

    return errors


def get_simple_feature_values(analysis):
    """Return only the value of every facial feature."""
    return {
        feature_name: feature_state["value"]
        for feature_name, feature_state in analysis["features"].items()
    }


def print_schema_summary():
    """Print the supported facial-feature schema."""
    print("=" * 70)
    print("THIRD EYE FACIAL FEATURE SCHEMA")
    print("=" * 70)

    for feature_name, settings in FEATURE_SCHEMA.items():
        print()
        print("Feature:", feature_name)
        print("Label:", settings["label"])
        print("Control method:", settings["control_method"])
        print("Allowed values:", ", ".join(settings["allowed_values"]))


if __name__ == "__main__":
    print_schema_summary()
