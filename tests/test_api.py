"""Checks for POST /predict. Train first so models/phishcheck_pipeline.joblib exists."""

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)

SPAM = (
    "You have WON a guaranteed £1000 cash or a £2000 prize. "
    "To claim yr prize call our customer service representative on "
    "08714712394 between 10am-7pm"
)
NORMAL = "Can you pick up milk on the way home? I'll be there by 7."


def test_blank_text_is_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "Paste" in response.json()["detail"]


def test_spammy_sample_is_phish():
    body = client.post("/predict", json={"text": SPAM}).json()
    assert body["label"] == "phish"
    assert body["phish_probability"] > 0.5
    words = [item["word"] for item in body["top_words"]]
    assert "prize" in words


def test_normal_sample_is_legit():
    body = client.post("/predict", json={"text": NORMAL}).json()
    assert body["label"] == "legit"
    assert body["phish_probability"] < 0.5


def test_response_shape():
    response = client.post("/predict", json={"text": SPAM})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "phish_probability", "top_words"}
    assert body["label"] in {"phish", "legit"}
    assert isinstance(body["phish_probability"], float)
    assert body["top_words"]
    for item in body["top_words"]:
        assert set(item) == {"word", "contribution"}
        assert item["contribution"] > 0
