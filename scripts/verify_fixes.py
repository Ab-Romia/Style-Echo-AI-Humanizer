"""Verify the review fixes without any network call."""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
logging.basicConfig(level=logging.WARNING)

ok = []

# Fix 6: EmbeddingAnalyzer does not load the model at construction.
from voiceprint.services.embedding_analyzer import EmbeddingAnalyzer
ea = EmbeddingAnalyzer()
assert ea._model is None, "MiniLM loaded eagerly"
ok.append("fix6 lazy MiniLM: model not loaded at init")

# Fix 1: empty choices -> RuntimeError (caught by fallback), not IndexError.
from voiceprint.services.rewrite.llm_rewriter import LlmRewriter
from voiceprint.models.style_profile import StyleProfile

class _FakeCompletions:
    def create(self, **kw):
        class R:  # mimic an empty-choices response
            choices = []
        return R()
class _FakeChat:
    completions = _FakeCompletions()
class _FakeClient:
    chat = _FakeChat()

rw = LlmRewriter(api_key="test-key")
rw._client = _FakeClient()
prof = StyleProfile(user_id="u", samples=["I write plainly. I keep it short."])
try:
    rw.rewrite("Some draft text.", prof)
    raise SystemExit("FAIL: expected RuntimeError on empty choices")
except RuntimeError as e:
    assert "no choices" in str(e).lower()
    ok.append("fix1 empty choices: raises RuntimeError (fallback-safe)")
except IndexError:
    raise SystemExit("FAIL: IndexError still leaks (fix1 broken)")

# Fix 2: validator does not divide by zero when sentence_length_std == 0.
from voiceprint.services.linguistic_analyzer import LinguisticAnalyzer
from voiceprint.services.validator import OutputValidator
la = LinguisticAnalyzer()
val = OutputValidator(la, ea)
zero_std_profile = StyleProfile(user_id="u", samples=["x"])
zero_std_profile.linguistic_features = {
    "avg_sentence_length": 12.0,
    "sentence_length_std": 0.0,
    "type_token_ratio": 0.5,
    "punctuation_density": 0.1,
    "flesch_reading_ease": 60.0,
}
matches = val._check_feature_alignment("This is a short test sentence here.", zero_std_profile)
assert "sentence_length_match" in matches
ok.append("fix2 validator zero-std: no ZeroDivisionError")

# Fix 4: passive ratio is a sentence proportion in [0, 1].
from voiceprint.services.stylometric_analyzer import StylometricAnalyzer
sa = StylometricAnalyzer()
passive_heavy = (
    "The results were recorded and were analyzed by the team. "
    "The samples were collected. The data was processed and was stored. "
    "Mistakes were made and were corrected."
)
feats = sa.analyze_text(passive_heavy)
pr = feats.get("passive_voice_ratio", feats.get("voice", {}).get("passive_voice_ratio"))
# locate the ratio wherever it lives
def find_ratio(d):
    if isinstance(d, dict):
        if "passive_voice_ratio" in d:
            return d["passive_voice_ratio"]
        for v in d.values():
            r = find_ratio(v)
            if r is not None:
                return r
    return None
pr = find_ratio(feats)
assert pr is not None, "passive_voice_ratio not found"
assert 0.0 <= pr <= 1.0, f"passive ratio out of [0,1]: {pr}"
ok.append(f"fix4 passive ratio bounded: {round(pr, 3)} in [0,1]")

print("ALL FIXES VERIFIED:")
for line in ok:
    print("  -", line)
