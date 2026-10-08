"""Route tests for the Phase 7B witness confirmation interface."""

import sqlite3
from pathlib import Path

import pytest

import app as app_module


@pytest.fixture
def sample_context():
    return {
        "confirmation_id": "witness-42",
        "description": "A round face with a broad nose and no beard.",
        "feature_summary": [
            {
                "label": "Face Shape",
                "display_value": "Round",
                "negated": False,
            }
        ],
        "warnings": [],
        "review_categories": [
            {
                "name": "heads",
                "label": "Face shape",
                "requested_attributes": {"face_shape": "round"},
                "explicitly_absent": False,
                "reason": "Ranked by face shape.",
                "candidates": [
                    {
                        "id": "head_05",
                        "display_name": "Head 05",
                        "file": "heads/head_05.png",
                        "score": 1.0,
                        "matched_attributes": {"face_shape": "round"},
                    }
                ],
            },
            {
                "name": "beards",
                "label": "Beard",
                "requested_attributes": {"beard_style": "none"},
                "explicitly_absent": True,
                "reason": "The description explicitly excludes a beard.",
                "candidates": [],
            },
            {
                "name": "lips",
                "label": "Lips",
                "requested_attributes": {},
                "explicitly_absent": False,
                "reason": "No lip attributes were requested.",
                "candidates": [],
            },
        ],
    }


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_confirmation_get_renders_description_and_candidate(
    client,
    sample_context,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        lambda confirmation_id: sample_context,
    )

    response = client.get("/confirm/witness-42")

    assert response.status_code == 200
    assert b"Confirm Facial Features" in response.data
    assert b"A round face with a broad nose and no beard." in response.data
    assert b"Head 05" in response.data
    assert b"assets/face_components/heads/head_05.png" in response.data


def test_confirmation_post_applies_only_displayed_choices(
    client,
    sample_context,
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        lambda confirmation_id: sample_context,
    )

    def fake_apply(context, choices):
        captured["context"] = context
        captured["choices"] = choices
        return {"confirmation_id": context["confirmation_id"]}

    monkeypatch.setattr(
        app_module,
        "apply_confirmed_choices",
        fake_apply,
    )

    response = client.post(
        "/confirm/witness-42",
        data={
            "heads": "head_05",
            "beards": "none",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith(
        "/confirmation/witness-42/result"
    )
    assert captured["choices"] == {
        "heads": "head_05",
        "beards": "none",
    }


def test_confirmation_post_requires_each_displayed_category(
    client,
    sample_context,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        lambda confirmation_id: sample_context,
    )

    response = client.post(
        "/confirm/witness-42",
        data={"heads": "head_05"},
    )

    assert response.status_code == 400
    assert b"Please choose an option for Beard." in response.data


def test_confirmation_returns_404_when_context_is_missing(
    client,
    monkeypatch,
):
    def missing_context(confirmation_id):
        raise FileNotFoundError(confirmation_id)

    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        missing_context,
    )

    response = client.get("/confirm/missing")
    assert response.status_code == 404


def test_confirmation_result_requires_confirmed_image(
    client,
    sample_context,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        lambda confirmation_id: sample_context,
    )
    monkeypatch.setattr(app_module.app, "static_folder", str(tmp_path))

    response = client.get("/confirmation/witness-42/result")
    assert response.status_code == 404


def test_confirmation_result_renders_existing_image(
    client,
    sample_context,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        app_module,
        "load_confirmation_context",
        lambda confirmation_id: sample_context,
    )
    monkeypatch.setattr(app_module.app, "static_folder", str(tmp_path))

    confirmed = (
        tmp_path
        / "outputs"
        / "confirmations"
        / "witness-42"
        / "confirmed_reference.png"
    )
    confirmed.parent.mkdir(parents=True)
    confirmed.write_bytes(b"test image placeholder")

    response = client.get("/confirmation/witness-42/result")

    assert response.status_code == 200
    assert b"Confirmed Forensic Reference" in response.data
    assert b"confirmed_reference.png" in response.data


def test_save_witness_creates_confirmation_and_redirects(
    client,
    monkeypatch,
    tmp_path,
):
    database_path = tmp_path / "third_eye.db"
    real_connect = sqlite3.connect

    with real_connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE witness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                witness_name TEXT,
                description TEXT NOT NULL
            )
            """
        )

    monkeypatch.setattr(
        app_module.sqlite3,
        "connect",
        lambda ignored_path: real_connect(database_path),
    )

    captured = {}

    def fake_create(description, confirmation_id, top_k):
        captured["description"] = description
        captured["confirmation_id"] = confirmation_id
        captured["top_k"] = top_k
        return {"confirmation_id": confirmation_id}

    monkeypatch.setattr(
        app_module,
        "create_confirmation",
        fake_create,
    )

    response = client.post(
        "/save_witness",
        data={
            "witness_name": "Test Witness",
            "description": "A person with a round face.",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith(
        "/confirm/witness-1"
    )
    assert captured == {
        "description": "A person with a round face.",
        "confirmation_id": "witness-1",
        "top_k": 3,
    }

    with real_connect(database_path) as connection:
        saved = connection.execute(
            "SELECT witness_name, description FROM witness"
        ).fetchone()

    assert saved == (
        "Test Witness",
        "A person with a round face.",
    )
