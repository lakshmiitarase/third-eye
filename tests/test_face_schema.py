import pytest

from face_schema import (
    FEATURE_SCHEMA,
    SCHEMA_VERSION,
    create_empty_analysis,
    create_empty_features,
    get_allowed_values,
    get_simple_feature_values,
    is_valid_feature_value,
    set_feature,
    validate_analysis,
)


def test_schema_version():
    assert SCHEMA_VERSION == "1.0"


def test_empty_features_contains_all_schema_fields():
    features = create_empty_features()
    assert set(features.keys()) == set(FEATURE_SCHEMA.keys())


def test_empty_features_have_default_state():
    features = create_empty_features()

    for feature_state in features.values():
        assert feature_state["value"] is None
        assert feature_state["confidence"] == 0.0
        assert feature_state["source_text"] == ""
        assert feature_state["mentioned"] is False
        assert feature_state["negated"] is False


def test_feature_states_are_independent():
    first = create_empty_features()
    second = create_empty_features()

    first["face_shape"]["value"] = "oval"

    assert second["face_shape"]["value"] is None


def test_valid_face_shape():
    assert is_valid_feature_value("face_shape", "oval") is True


def test_invalid_face_shape():
    assert is_valid_feature_value("face_shape", "banana") is False


def test_none_is_valid_for_unmentioned_feature():
    assert is_valid_feature_value("face_shape", None) is True


def test_unknown_feature_is_invalid():
    assert is_valid_feature_value("unknown_feature", "value") is False


def test_get_allowed_values():
    allowed = get_allowed_values("face_shape")

    assert "oval" in allowed
    assert "round" in allowed
    assert "square" in allowed


def test_get_allowed_values_rejects_unknown_feature():
    with pytest.raises(KeyError):
        get_allowed_values("unknown_feature")


def test_set_valid_feature():
    analysis = create_empty_analysis("A man with an oval face")

    set_feature(
        analysis=analysis,
        feature_name="face_shape",
        value="oval",
        confidence=1.0,
        source_text="oval face",
    )

    result = analysis["features"]["face_shape"]

    assert result["value"] == "oval"
    assert result["confidence"] == 1.0
    assert result["source_text"] == "oval face"
    assert result["mentioned"] is True
    assert result["negated"] is False


def test_invalid_value_raises_error():
    analysis = create_empty_analysis()

    with pytest.raises(ValueError):
        set_feature(
            analysis=analysis,
            feature_name="face_shape",
            value="banana",
            confidence=1.0,
            source_text="banana face",
        )


def test_invalid_confidence_raises_error():
    analysis = create_empty_analysis()

    with pytest.raises(ValueError):
        set_feature(
            analysis=analysis,
            feature_name="face_shape",
            value="oval",
            confidence=1.5,
            source_text="oval face",
        )


def test_explicit_no_beard():
    analysis = create_empty_analysis("A clean-shaven man")

    set_feature(
        analysis=analysis,
        feature_name="beard_style",
        value="none",
        confidence=1.0,
        source_text="clean-shaven",
        mentioned=True,
        negated=True,
    )

    beard = analysis["features"]["beard_style"]

    assert beard["value"] == "none"
    assert beard["mentioned"] is True
    assert beard["negated"] is True


def test_empty_analysis_is_valid():
    analysis = create_empty_analysis()
    assert validate_analysis(analysis) == []


def test_completed_analysis_is_valid():
    analysis = create_empty_analysis("A man with an oval face")

    set_feature(analysis, "gender", "male", 1.0, "man")
    set_feature(analysis, "face_shape", "oval", 1.0, "oval face")

    assert validate_analysis(analysis) == []


def test_simple_feature_values():
    analysis = create_empty_analysis()

    set_feature(
        analysis,
        "face_shape",
        "oval",
        1.0,
        "oval face",
    )

    simple = get_simple_feature_values(analysis)

    assert simple["face_shape"] == "oval"
    assert simple["eye_shape"] is None
