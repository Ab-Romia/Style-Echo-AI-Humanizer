"""
Utility for loading models with fallback and error handling.
"""
import logging
import warnings

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load spaCy model with fallback options.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        spaCy nlp object or None if loading fails
    """
    import spacy

    try:
        nlp = spacy.load(model_name)
        logger.info(f"✓ Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        logger.warning(f"Model {model_name} not found. Trying to download...")

        try:
            import subprocess
            subprocess.run([
                "python", "-m", "spacy", "download", model_name
            ], check=True, capture_output=True)

            nlp = spacy.load(model_name)
            logger.info(f"✓ Downloaded and loaded spaCy model: {model_name}")
            return nlp
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")

            # Try loading blank model as fallback
            try:
                nlp = spacy.blank("en")
                logger.warning("⚠️ Using blank spaCy model (limited functionality)")
                return nlp
            except Exception as e2:
                logger.error(f"Failed to load blank model: {e2}")
                return None


def load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load sentence transformer model with error handling.

    Args:
        model_name: Name of the sentence transformer model

    Returns:
        SentenceTransformer object or None if loading fails
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"✓ Loaded sentence transformer: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        return None


def ensure_nltk_data():
    """
    Ensure NLTK data is downloaded.
    """
    import nltk

    required_data = ['stopwords', 'punkt']

    for data_name in required_data:
        try:
            nltk.data.find(f'corpora/{data_name}')
            logger.info(f"✓ NLTK {data_name} already available")
        except LookupError:
            logger.info(f"Downloading NLTK {data_name}...")
            try:
                nltk.download(data_name, quiet=True)
                logger.info(f"✓ Downloaded NLTK {data_name}")
            except Exception as e:
                logger.error(f"Failed to download NLTK {data_name}: {e}")


def check_dependencies():
    """
    Check if all required dependencies are available.

    Returns:
        dict with status of each dependency
    """
    status = {}

    # Check spaCy
    try:
        import spacy
        status['spacy'] = True
    except ImportError:
        status['spacy'] = False

    # Check sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        status['sentence_transformers'] = True
    except ImportError:
        status['sentence_transformers'] = False

    # Check NLTK
    try:
        import nltk
        status['nltk'] = True
    except ImportError:
        status['nltk'] = False

    # Check textstat
    try:
        import textstat
        status['textstat'] = True
    except ImportError:
        status['textstat'] = False

    return status


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("Checking dependencies...")
    deps = check_dependencies()
    for dep, available in deps.items():
        print(f"  {dep}: {'✓' if available else '✗'}")

    print("\nLoading models...")
    nlp = load_spacy_model()
    print(f"  spaCy model: {'✓' if nlp else '✗'}")

    transformer = load_sentence_transformer()
    print(f"  Sentence transformer: {'✓' if transformer else '✗'}")

    print("\nEnsuring NLTK data...")
    ensure_nltk_data()

    print("\n✓ Setup complete!")
