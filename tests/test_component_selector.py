from component_selector import (
    load_catalog,
    select_all_components,
    select_best_components,
    select_category,
    validate_selected_files,
)
from description_parser import analyze_description


def test_catalog_contains_88_components():
    catalog = load_catalog()
    total = sum(
        len(entries)
        for entries in catalog["components"].values()
    )
    assert total == 88


def test_round_head_selection():
    analysis = analyze_description("A person with a round face.")
    result = select_category(analysis, "heads", top_k=3)

    assert result["candidates"][0]["id"] == "head_05"
    assert result["candidates"][0]["score"] == 1.0


def test_short_curly_hair_selection():
    analysis = analyze_description("A person with short curly black hair.")
    result = select_category(analysis, "hair", top_k=3)

    assert result["candidates"][0]["id"] == "hair_08"
    assert result["candidates"][0]["score"] == 1.0


def test_narrow_eye_selection():
    analysis = analyze_description("A person with narrow eyes.")
    result = select_category(analysis, "eyes", top_k=3)

    assert result["candidates"][0]["id"] == "eyes_08"
    assert result["candidates"][0]["score"] == 1.0


def test_thick_arched_eyebrow_selection():
    analysis = analyze_description("A person with thick arched eyebrows.")
    result = select_category(analysis, "eyebrows", top_k=3)

    candidate = result["candidates"][0]
    assert candidate["id"] in {"eyebrows_03", "eyebrows_04", "eyebrows_05"}
    assert candidate["score"] > 0.0


def test_broad_short_upturned_nose_selection():
    analysis = analyze_description("A person with a broad short upturned nose.")
    result = select_category(analysis, "noses", top_k=3)

    assert result["candidates"][0]["id"] == "nose_12"
    assert result["candidates"][0]["score"] == 1.0


def test_full_wide_lips_selection():
    analysis = analyze_description("A person with full lips and a wide mouth.")
    result = select_category(analysis, "lips", top_k=3)

    assert result["candidates"][0]["id"] == "lips_09"
    assert result["candidates"][0]["score"] == 1.0


def test_thin_moustache_selection():
    analysis = analyze_description("A man with a thin moustache.")
    result = select_category(analysis, "moustaches", top_k=3)

    assert result["candidates"][0]["id"] == "moustache_04"
    assert result["candidates"][0]["score"] == 1.0


def test_no_beard_returns_no_component():
    analysis = analyze_description("A clean-shaven man.")
    result = select_category(analysis, "beards", top_k=3)

    assert result["explicitly_absent"] is True
    assert result["candidates"] == []


def test_no_moustache_returns_no_component():
    analysis = analyze_description("A man without a moustache.")
    result = select_category(analysis, "moustaches", top_k=3)

    assert result["explicitly_absent"] is True
    assert result["candidates"] == []


def test_no_relevant_attributes_returns_no_candidates():
    analysis = analyze_description("A person was seen near the station.")
    result = select_category(analysis, "noses", top_k=3)

    assert result["requested_attributes"] == {}
    assert result["candidates"] == []


def test_returns_requested_number_of_candidates():
    analysis = analyze_description("A person with an oval face.")
    result = select_category(analysis, "heads", top_k=3)

    assert len(result["candidates"]) == 3


def test_selection_is_deterministic():
    analysis = analyze_description("A person with round eyes.")

    first = select_category(analysis, "eyes", top_k=3)
    second = select_category(analysis, "eyes", top_k=3)

    assert first == second


def test_all_selected_files_exist():
    analysis = analyze_description(
        "A young man with a round face, narrow eyes, a broad nose, "
        "short curly black hair and a thin moustache."
    )
    selections = select_all_components(analysis, top_k=3)

    assert validate_selected_files(selections) == []


def test_best_component_selection():
    analysis = analyze_description(
        "A person with a round face, narrow eyes and short curly hair."
    )
    best = select_best_components(analysis)

    assert best["heads"]["id"] == "head_05"
    assert best["eyes"]["id"] == "eyes_08"
    assert best["hair"]["id"] == "hair_08"


def test_candidate_contains_placement_information():
    analysis = analyze_description("A person with a round face.")
    result = select_category(analysis, "heads", top_k=1)
    placement = result["candidates"][0]["placement"]

    assert placement["canvas_width"] == 620
    assert placement["canvas_height"] == 620
    assert isinstance(placement["x"], int)
    assert isinstance(placement["y"], int)


def test_unknown_category_is_rejected():
    analysis = analyze_description("A person with an oval face.")

    try:
        select_category(analysis, "unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown category should raise KeyError")


def test_invalid_top_k_is_rejected():
    analysis = analyze_description("A person with an oval face.")

    try:
        select_category(analysis, "heads", top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k=0 should raise ValueError")
