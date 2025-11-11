"""
Quick test script to verify the API is working.

Run this after starting the server to test the basic functionality.
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# Sample writing from a user (pretending these are my writing samples)
SAMPLE_TEXTS = [
    """
    I've been thinking about this problem for a while now, and honestly, it's trickier
    than I first thought. The main issue is that we need to balance performance with
    readability – you can't just optimize everything and expect people to understand
    what's going on. I mean, sure, we could shave off a few milliseconds here and there,
    but at what cost? Sometimes the simpler approach is actually better, even if it's
    not the fastest.
    """,
    """
    So here's the thing about writing good code: it's not just about making it work.
    Anyone can hack together something that runs. The real challenge is making something
    that other people can actually work with later. I've seen way too many projects where
    someone wrote "clever" code that nobody else could figure out. Clean, straightforward
    code beats clever code every single time.
    """,
    """
    You know what really bugs me? When people say "just use this framework" without
    understanding what it actually does. Like, frameworks are great and all, but you
    need to know what's happening under the hood. Otherwise you're just copying and
    pasting Stack Overflow answers and hoping things work. That's not engineering,
    that's gambling.
    """
]

# AI-generated text to humanize
AI_TEXT = """
    It is important to note that effective code organization requires careful consideration
    of multiple factors. One must consider maintainability, scalability, and performance
    simultaneously. Furthermore, it is worth mentioning that best practices should be
    followed consistently throughout the development process. The implementation of proper
    design patterns, adherence to coding standards, and comprehensive documentation are
    essential components of successful software development.
"""


def test_health():
    """Check if the server is running."""
    print("Checking server health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"✓ Server status: {response.json()}\n")


def test_quick_analysis():
    """Test the quick analysis endpoint."""
    print("Testing quick text analysis...")
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"text": SAMPLE_TEXTS[0]}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Analysis completed")
        print(f"  Word count: {result['word_count']}")
        print(f"  Avg sentence length: {result['linguistic_features'].get('avg_sentence_length', 'N/A'):.1f}")
        print(f"  AI probability: {result['ai_detection'].get('ai_probability', 'N/A'):.2f}\n")
    else:
        print(f"✗ Analysis failed: {response.status_code}\n")


def test_full_pipeline():
    """Test the complete profile creation and humanization pipeline."""
    print("Testing full pipeline...")

    # 1. Create a style profile
    print("  Creating style profile...")
    profile_response = requests.post(
        f"{BASE_URL}/profiles",
        json={
            "user_id": "test_user_123",
            "samples": SAMPLE_TEXTS,
            "profile_name": "My Casual Tech Writing"
        }
    )

    if profile_response.status_code != 201:
        print(f"✗ Profile creation failed: {profile_response.status_code}")
        print(profile_response.text)
        return

    profile_data = profile_response.json()
    profile_id = profile_data["profile_id"]
    print(f"  ✓ Profile created: {profile_id}")
    print(f"    Total words analyzed: {profile_data['total_words']}")
    print(f"    Samples: {profile_data['num_samples']}\n")

    # 2. Humanize AI text
    print("  Humanizing AI-generated text...")
    humanize_response = requests.post(
        f"{BASE_URL}/humanize",
        json={
            "profile_id": profile_id,
            "text": AI_TEXT,
            "strength": 0.7,
            "preserve_meaning": True
        }
    )

    if humanize_response.status_code != 200:
        print(f"✗ Humanization failed: {humanize_response.status_code}")
        print(humanize_response.text)
        return

    result = humanize_response.json()
    print(f"  ✓ Text humanized!")
    print(f"\n  Original text:")
    print(f"    {AI_TEXT.strip()[:150]}...")
    print(f"\n  Humanized text:")
    print(f"    {result['humanized_text'][:150]}...")
    print(f"\n  Metrics:")
    print(f"    Style similarity: {result['similarity_score']:.2f}")
    print(f"    AI detection score: {result['ai_detection_score']:.2f}\n")

    # 3. Get profile details
    print("  Fetching profile details...")
    get_response = requests.get(f"{BASE_URL}/profiles/{profile_id}")

    if get_response.status_code == 200:
        print(f"  ✓ Profile retrieved successfully\n")
    else:
        print(f"  ✗ Failed to retrieve profile\n")

    return profile_id


def test_profile_management(profile_id):
    """Test profile listing and deletion."""
    print("Testing profile management...")

    # List user profiles
    print("  Listing user profiles...")
    list_response = requests.get(f"{BASE_URL}/users/test_user_123/profiles")

    if list_response.status_code == 200:
        profiles = list_response.json()
        print(f"  ✓ Found {len(profiles)} profile(s)\n")
    else:
        print(f"  ✗ Failed to list profiles\n")

    # Delete profile
    print("  Cleaning up test profile...")
    delete_response = requests.delete(f"{BASE_URL}/profiles/{profile_id}")

    if delete_response.status_code == 204:
        print(f"  ✓ Profile deleted\n")
    else:
        print(f"  ✗ Failed to delete profile\n")


if __name__ == "__main__":
    print("=" * 60)
    print("VoicePrint API Test Suite")
    print("=" * 60)
    print()

    try:
        # Run tests
        test_health()
        test_quick_analysis()
        profile_id = test_full_pipeline()

        if profile_id:
            test_profile_management(profile_id)

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to the API server.")
        print("  Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
