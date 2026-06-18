"""End-to-end smoke test: build a profile, adapt a draft, no API key (rule path)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from voiceprint.services.voiceprint_service import VoicePrintService

SAMPLES = [
    ("I tend to write in short, direct sentences. I don't hedge much. "
     "When I explain something, I get to the point and back it with a real example. "
     "I like contractions and I avoid filler words. My paragraphs stay tight. "
     "If an idea needs three sentences, I give it three, not ten. I read my drafts out loud. "
     "When a line trips me up, I cut it. I'd rather sound plain than clever. "
     "I keep my verbs active and my nouns concrete. I don't reach for big words to look smart. "
     "Most of what I write is meant to be skimmed, so I front-load the point. "
     "I use lists when the structure earns one, and prose when it doesn't."),
    ("Last week I rebuilt a small parser. It kept choking on nested quotes, "
     "so I rewrote the tokenizer from scratch. The fix was simple once I saw the bug: "
     "I was tracking depth with a flag instead of a counter. Small change, big difference. "
     "I added a few tests first, then watched them fail, then made them pass. "
     "That loop keeps me honest. I don't trust code I haven't run against a broken case. "
     "When it worked, I deleted the old path and moved on. I didn't gold-plate it. "
     "Good enough that ships beats perfect that sits in a branch for a month."),
    ("Most of my notes read like a log. I write what I tried, what broke, and what I learned. "
     "I rarely use long words when a short one works. If a sentence runs too long, I split it. "
     "I care more about being clear than sounding clever, and I edit hard before I ship. "
     "When I learn something the hard way, I write it down so I don't pay for it twice. "
     "I keep a running file of mistakes. It's the most useful thing I own. "
     "I try to leave the code a little cleaner than I found it. Not a rewrite, just a nudge. "
     "Small, steady improvements add up faster than big rewrites that never land."),
    ("When I review a pull request, I look for the one change that matters and ignore the noise. "
     "I ask whether the code does what it claims and whether I'd be able to fix it at midnight. "
     "If a function needs a paragraph to explain, it probably needs to be two functions. "
     "I leave comments that say why, not what. The code already says what. "
     "I push back on cleverness that buys nothing. I praise the boring fix that just works. "
     "When I'm stuck, I write the problem down in plain words. Half the time the bug shows up "
     "right there in the sentence. I don't believe in magic. I believe in reading the stack trace. "
     "Most hard bugs are just two simple assumptions that quietly disagree. "
     "So I slow down, check each one, and the fix usually falls out on its own. "
     "I have learned to trust that process even when the deadline is loud and close. "
     "It has never once let me down, so I keep coming back to it on every project."),
]

DRAFT = (
    "It is important to note that the utilization of comprehensive methodologies "
    "can significantly enhance the robustness of the overall system architecture, "
    "thereby facilitating a more seamless and efficient operational workflow for stakeholders."
)

def main():
    svc = VoicePrintService()
    print("building profile...")
    profile = svc.build_profile("smoke-user", SAMPLES, profile_name="smoke")
    print("  profile_id:", profile.profile_id)
    print("  voice_centroid set:", profile.get_voice_centroid_as_list() is not None)

    print("adapting draft (rule path, no key)...")
    result = svc.adapt_draft(profile.profile_id, DRAFT, use_llm=False)
    print("  rewrite_path:", result.get("rewrite_path"))
    print("  voice_match_before:", round(result.get("voice_match_before", 0), 4))
    print("  voice_match_after:", round(result.get("voice_match_after", 0), 4))
    print("  validation.semantic_preservation:",
          round(result.get("validation", {}).get("semantic_preservation", 0), 4))
    print("  adapted_text:", result.get("adapted_text", "")[:200])
    fb = result.get("feedback", {})
    print("  feedback keys:", list(fb.keys()))

    print("quick_analysis...")
    qa = svc.quick_analysis(DRAFT)
    print("  quick_analysis keys:", list(qa.keys()))
    print("OK: end-to-end smoke passed")

if __name__ == "__main__":
    main()
