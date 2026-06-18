"""
Live API smoke test.

Run this against a running server (uvicorn voiceprint.main:app) to exercise the
profile and adapt endpoints end to end. It is skipped automatically when the
server is not reachable, so it is safe in a normal pytest run.
"""
import pytest

requests = pytest.importorskip("requests")

BASE_URL = "http://localhost:8000/api/v1"

SAMPLE_TEXTS = [
    (
        "I have been thinking about this problem for a while now, and honestly "
        "it is trickier than I first thought. The main issue is that we need "
        "to balance performance with readability. You cannot just optimize "
        "everything and expect people to follow what is going on."
    ),
    (
        "Here is the thing about writing good code: it is not just about making "
        "it work. Anyone can hack together something that runs. The real "
        "challenge is making something other people can work with later."
    ),
    (
        "What really bugs me is when people say just use this framework without "
        "understanding what it does. Frameworks are great, but you need to know "
        "what is happening under the hood, or you are just copying answers and "
        "hoping they work."
    ),
]

SOURCE_DRAFT = (
    "Effective code organization requires careful consideration of multiple "
    "factors. One must consider maintainability, scalability, and performance "
    "simultaneously. Best practices should be followed consistently throughout "
    "the development process."
)


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason="VoicePrint API server is not running"
)


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["service"] == "VoicePrint"


def test_quick_analysis():
    response = requests.post(f"{BASE_URL}/analyze", json={"text": SAMPLE_TEXTS[0]})
    assert response.status_code == 200
    result = response.json()
    assert "linguistic_features" in result
    assert "fingerprint_axes" in result
    assert "ai_detection" not in result


def test_build_and_adapt_pipeline():
    profile_response = requests.post(
        f"{BASE_URL}/profiles",
        json={
            "user_id": "test_user_123",
            "samples": SAMPLE_TEXTS,
            "profile_name": "My casual tech writing",
        },
    )
    assert profile_response.status_code == 201, profile_response.text
    profile_id = profile_response.json()["profile_id"]

    adapt_response = requests.post(
        f"{BASE_URL}/adapt",
        json={"profile_id": profile_id, "source_draft": SOURCE_DRAFT, "use_llm": False},
    )
    assert adapt_response.status_code == 200, adapt_response.text
    result = adapt_response.json()
    assert "adapted_text" in result
    assert "voice_match_before" in result
    assert "voice_match_after" in result
    assert result["rewrite_path"] == "rule"
    assert "ai_detection_score" not in result

    requests.delete(f"{BASE_URL}/profiles/{profile_id}")
