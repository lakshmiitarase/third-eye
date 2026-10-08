import pytest

from description_parser import analyze_description, extract_features
from face_schema import validate_analysis


def value(result, feature_name):
    return result["features"][feature_name]["value"]


def test_complete_description():
    result = analyze_description(
        "A young man with a round face, narrow eyes, a broad nose, "
        "short curly black hair, a thin moustache and no beard."
    )

    assert value(result, "gender") == "male"
    assert value(result, "age_group") == "young_adult"
    assert value(result, "face_shape") == "round"
    assert value(result, "eye_shape") == "narrow"
    assert value(result, "nose_width") == "broad"
    assert value(result, "hair_length") == "short"
    assert value(result, "hair_texture") == "curly"
    assert value(result, "hair_colour") == "black"
    assert value(result, "moustache_style") == "thin"
    assert value(result, "beard_style") == "none"


def test_long_hair_does_not_create_long_face():
    result = analyze_description(
        "A woman with long black hair and an oval face."
    )

    assert value(result, "hair_length") == "long"
    assert value(result, "face_shape") == "oval"


def test_round_eyes_do_not_create_round_face():
    result = analyze_description(
        "A man with a square face and round eyes."
    )

    assert value(result, "face_shape") == "square"
    assert value(result, "eye_shape") == "round"


def test_short_hair_does_not_create_short_face():
    result = analyze_description(
        "A man with short hair and an oblong face."
    )

    assert value(result, "hair_length") == "short"
    assert value(result, "face_shape") == "oblong"


def test_numeric_age_child():
    result = analyze_description("A 10-year-old child with round eyes.")
    assert value(result, "age_group") == "child"


def test_numeric_age_teen():
    result = analyze_description("A person aged 17 years old with short hair.")
    assert value(result, "age_group") == "teen"


def test_numeric_age_young_adult():
    result = analyze_description("A 25-year-old man with an oval face.")
    assert value(result, "age_group") == "young_adult"


def test_numeric_age_middle_aged():
    result = analyze_description("A woman approximately 52 years old.")
    assert value(result, "age_group") == "middle_aged"


def test_numeric_age_elderly():
    result = analyze_description("A man about 70 years old.")
    assert value(result, "age_group") == "elderly"


def test_no_beard_is_explicitly_negated():
    result = analyze_description("A clean-shaven man with short hair.")
    beard = result["features"]["beard_style"]

    assert beard["value"] == "none"
    assert beard["mentioned"] is True
    assert beard["negated"] is True


def test_no_moustache_is_explicitly_negated():
    result = analyze_description("A man without a moustache.")
    moustache = result["features"]["moustache_style"]

    assert moustache["value"] == "none"
    assert moustache["negated"] is True


def test_no_glasses_is_explicitly_negated():
    result = analyze_description("A woman not wearing glasses.")
    glasses = result["features"]["glasses"]

    assert glasses["value"] == "none"
    assert glasses["negated"] is True


def test_glasses_style():
    result = analyze_description("A man wearing rectangular glasses.")
    assert value(result, "glasses") == "rectangular"


def test_hair_attributes_with_multiple_adjectives():
    result = analyze_description("A woman with long wavy dark brown hair.")

    assert value(result, "hair_length") == "long"
    assert value(result, "hair_texture") == "wavy"
    assert value(result, "hair_colour") == "dark_brown"


def test_multiple_eye_attributes():
    result = analyze_description("A man with large narrow eyes.")

    assert value(result, "eye_size") == "large"
    assert value(result, "eye_shape") == "narrow"


def test_eyebrow_attributes():
    result = analyze_description("A woman with thick arched eyebrows.")

    assert value(result, "eyebrow_thickness") == "thick"
    assert value(result, "eyebrow_shape") == "arched"


def test_multiple_nose_attributes():
    result = analyze_description("A man with a long broad straight nose.")

    assert value(result, "nose_length") == "long"
    assert value(result, "nose_width") == "broad"
    assert value(result, "nose_bridge") == "straight"


def test_lip_and_mouth_attributes():
    result = analyze_description("A woman with full lips and a wide mouth.")

    assert value(result, "lip_thickness") == "full"
    assert value(result, "mouth_width") == "wide"


def test_empty_description():
    result = analyze_description("")

    assert result["warnings"] == ["Description is empty."]
    assert len(result["missing_important_features"]) > 0
    assert validate_analysis(result) == []


def test_invalid_description_type():
    with pytest.raises(TypeError):
        analyze_description(None)


def test_conflicting_face_shapes_are_reported():
    result = analyze_description(
        "The witness first said oval face but later said round face."
    )

    assert value(result, "face_shape") == "oval"
    assert any(
        conflict["feature"] == "face_shape"
        for conflict in result["conflicts"]
    )


def test_schema_validation_passes():
    result = analyze_description(
        "A middle-aged woman with a square face and almond eyes."
    )
    assert validate_analysis(result) == []


def test_legacy_output_has_expected_keys():
    result = extract_features(
        "A young man with a round face, narrow eyes and short black hair."
    )

    assert set(result.keys()) == {
        "gender",
        "age_group",
        "face",
        "eyes",
        "eyebrows",
        "nose",
        "lips",
        "hair",
        "skin_tone",
        "beard",
        "moustache",
    }

    assert result["gender"] == "male"
    assert result["age_group"] == "young adult"
    assert result["face"] == "round"
    assert result["eyes"] == "narrow eyes"
    assert result["hair"] == "short black hair"


def test_unmentioned_beard_remains_none_in_legacy_output():
    result = extract_features("A woman with an oval face.")
    assert result["beard"] is None


def test_parser_does_not_infer_ethnicity_or_nationality():
    result = analyze_description("A man with an oval face and black hair.")

    assert "ethnicity" not in result["features"]
    assert "nationality" not in result["features"]
