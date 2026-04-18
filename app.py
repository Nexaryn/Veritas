"""
Fake News Detection Web Application — Veritas
A Flask-based MIS for detecting fake news using dual-layer architecture:
  Layer 1: scikit-learn ensemble ML models (Naive Bayes, LR, RF, SVM)
  Layer 2: Knowledge-Based FactChecker (physics, geography, medical scam detection)

NEW in this version:
  - URL scraping via BeautifulSoup (paste a URL directly into the text box)
  - Layer 0 input validation (gibberish / keyboard-smash detection)
  - Graceful degradation if spaCy / FactChecker is unavailable
"""

from flask import Flask, render_template, request, jsonify
import os
import csv
import re
import logging
from datetime import datetime
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

from src.data_processing import TextPreprocessor
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: FactChecker (requires spaCy en_core_web_sm)
# Graceful degradation: if it fails for ANY reason, the app keeps running
# using ML-only mode.
# ---------------------------------------------------------------------------
try:
    from src.fact_checker import FactChecker
    FACT_CHECKER_AVAILABLE = True
    log.info("✓ FactChecker module imported successfully")
except (ImportError, OSError) as e:
    log.warning(f"⚠️  FactChecker not available: {e}")
    log.warning("   Install with: pip install spacy && python -m spacy download en_core_web_sm")
    FactChecker = None
    FACT_CHECKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# --- AUDIT TRAIL SETUP ---
HISTORY_FILE = "data/audit_trail.csv"
if not os.path.exists(HISTORY_FILE):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["Time", "News Snippet", "AI Verdict", "Confidence", "User Feedback"]
        )

# --- Global state ---
models: dict = {}
vectorizer = None
fact_checker = None
current_model_name = "Naive Bayes"
model_loaded = False

# ---------------------------------------------------------------------------
# Sample news dataset (unchanged from original)
# ---------------------------------------------------------------------------
SAMPLE_NEWS = {
    "real_1": {
        "title": "India Launches Gaganyaan Mission Successfully",
        "text": (
            "The Indian Space Research Organisation successfully launched the Gaganyaan-1 "
            "unmanned test flight from Sriharikota on Monday morning. The mission marks a "
            "crucial milestone in India's human spaceflight program, testing critical systems "
            "including the crew module and escape mechanisms. ISRO Chairman Dr. S Somanath "
            "confirmed all systems performed as expected during the 15-minute suborbital flight."
        ),
    },
    "real_2": {
        "title": "Mumbai Metro Line 3 Opens After Nine Years of Construction",
        "text": (
            "The Mumbai Metro Rail Corporation inaugurated the 33.5-kilometer underground Metro "
            "Line 3 connecting Colaba to SEEPZ on Sunday. The Aqua Line features 27 stations, "
            "including India's deepest metro station at 32 meters below ground level."
        ),
    },
    "fake_1": {
        "title": "SHOCKING: Government Announces 200% Tax on All Bank Deposits",
        "text": (
            "BREAKING NEWS! Finance Ministry has announced a shocking 200 percent tax on all "
            "bank savings and fixed deposits starting next week! Share this immediately before "
            "it's too late! Your hard-earned money is at risk!"
        ),
    },
    "fake_2": {
        "title": "Doctors Reveal: Drinking Hot Water Cures Cancer, Diabetes, Heart Disease",
        "text": (
            "MEDICAL BREAKTHROUGH that Big Pharma doesn't want you to know! Top doctors have "
            "discovered that drinking hot water on empty stomach can cure cancer, diabetes, "
            "and all heart diseases within 30 days! Share this with everyone before this post "
            "gets deleted!"
        ),
    },
}


# ---------------------------------------------------------------------------
# URL Scraping helpers
# ---------------------------------------------------------------------------

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_SCRAPE_TIMEOUT = 10  # seconds


def is_url(text: str) -> bool:
    """Return True if the stripped text looks like an HTTP/HTTPS URL."""
    stripped = text.strip()
    try:
        parsed = urlparse(stripped)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def scrape_url(url: str) -> tuple[bool, str]:
    """
    Fetch a URL, parse the article body from <p> tags via BeautifulSoup.

    Returns:
        (success: bool, text_or_error_message: str)
    """
    try:
        response = requests.get(url, headers=_SCRAPE_HEADERS, timeout=_SCRAPE_TIMEOUT)
        response.raise_for_status()  # raises for 4xx / 5xx
    except requests.exceptions.Timeout:
        return False, "⏱️ The URL took too long to respond (timeout). Please paste the article text directly."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        if status == 403:
            return False, "🚫 Access denied (403 Forbidden). The website blocks automated access. Please paste the article text directly."
        if status == 404:
            return False, "🔍 Page not found (404). Please check the URL and try again."
        return False, f"❌ HTTP error {status} when fetching the URL. Please paste the article text directly."
    except requests.exceptions.ConnectionError:
        return False, "🌐 Could not connect to the URL. Please check your internet connection or paste the article text directly."
    except requests.exceptions.RequestException as e:
        return False, f"❌ Could not fetch URL: {str(e)}. Please paste the article text directly."

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Prefer <article> content; fall back to <main>, then <body>
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"article|content|story|post", re.I))
            or soup.body
        )

        paragraphs = container.find_all("p") if container else soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 80:
            return False, (
                "⚠️ Could not extract enough article text from that URL "
                "(the page may be behind a paywall or use JavaScript rendering). "
                "Please paste the article text directly."
            )

        log.info(f"✓ Scraped {len(text)} characters from {url}")
        return True, text

    except Exception as e:
        log.warning(f"BeautifulSoup parsing error: {e}")
        return False, "❌ Failed to parse the page content. Please paste the article text directly."


# ---------------------------------------------------------------------------
# Layer 0 — Input validation
# ---------------------------------------------------------------------------

def validate_input(text: str, preprocessor) -> tuple[bool, str, str]:
    """
    Layer 0 guard: reject obviously bad input before any ML work.

    Checks (in order):
      1. Empty / whitespace only
      2. Too few words (< 4)
      3. Keyboard-smash / gibberish (average word length > 12)
      4. Non-English dominant text (< 30 % ASCII letters)
      5. Nothing left after preprocessing

    Returns: (is_valid, error_message, warning_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze.", ""

    words = text.strip().split()

    # Check minimum word count
    if len(words) < 4:
        return False, "Text is too short — please enter at least 4 words of article content.", ""

    # Gibberish / keyboard-smash detection: average word length > 12
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 12:
        return False, (
            "The text looks like keyboard smash or random characters. "
            "Please paste a real news article."
        ), ""

    # Mostly non-English?
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    total_non_space = len(re.sub(r"\s", "", text))
    if total_non_space > 0 and english_chars / total_non_space < 0.3:
        return False, "This model only supports English text.", ""

    # Preprocessed content check
    preprocessed = preprocessor.preprocess(text)
    if len(preprocessed.strip()) < 10:
        return False, (
            "Text contains insufficient analyzable content after preprocessing. "
            "Please provide more meaningful English text."
        ), ""

    # Soft warning for very short preprocessed text
    warning = ""
    if len(preprocessed.split()) < 5:
        warning = "⚠️ Very short text detected. Prediction confidence may be low."

    return True, "", warning


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_all_models() -> bool:
    """Load vectorizer + all trained classifier models. Returns True on success."""
    global models, vectorizer, fact_checker, model_loaded

    try:
        vectorizer_path = os.path.join(config.MODELS_DIR, "vectorizer.joblib")
        if not os.path.exists(vectorizer_path):
            log.warning("⚠ No vectorizer found. Please train models first.")
            return False

        vectorizer = joblib.load(vectorizer_path)

        available_models = {
            "Naive Bayes":          "naive_bayes_model.joblib",
            "Random Forest":        "random_forest_model.joblib",
            "SVM":                  "svm_model.joblib",
            "Logistic Regression":  "logistic_regression_model.joblib",
        }

        models_loaded = 0
        for model_name, model_file in available_models.items():
            model_path = os.path.join(config.MODELS_DIR, model_file)
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                models_loaded += 1
                log.info(f"✓ {model_name} model loaded")

        if models_loaded == 0:
            log.warning("⚠ No trained models found. Please train models first.")
            return False

        log.info(f"✓ {models_loaded} model(s) loaded")

        # --- Initialise FactChecker (Layer 2) — graceful degradation ---
        if FACT_CHECKER_AVAILABLE and FactChecker is not None:
            try:
                google_api_key = getattr(config, "GOOGLE_FACT_CHECK_API_KEY", None)
                fact_checker = FactChecker(google_api_key=google_api_key)
                mode = "Google API" if google_api_key else "Wikipedia"
                log.info(f"✓ FactChecker initialised ({mode} mode)")
            except Exception as fc_err:
                log.warning(
                    f"⚠️  FactChecker initialisation failed: {fc_err}. "
                    "Running in ML-only mode."
                )
                fact_checker = None
        else:
            log.warning("⚠ FactChecker not available — running in ML-only mode.")
            fact_checker = None

        return True

    except Exception as e:
        log.error(f"Error loading models: {e}")
        return False


model_loaded = load_all_models()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_loaded=len(models) > 0,
        available_models=list(models.keys()),
        sample_news=SAMPLE_NEWS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Core prediction endpoint.

    Flow:
      0. Receive text (or URL)
      1. If URL → scrape article text
      2. Layer 0 validation (gibberish guard)
      3. ML ensemble prediction (Layer 1)
      4. FactChecker override (Layer 2) — skipped gracefully if unavailable
      5. Write audit trail
      6. Return JSON
    """
    log.info("=" * 70)
    log.info("🔍 PREDICTION REQUEST")
    log.info(f"   FactChecker available: {fact_checker is not None}")

    try:
        if not model_loaded:
            return jsonify({"error": "Model not loaded. Please train a model first.", "success": False})

        data = request.get_json()
        raw_input = data.get("text", "").strip()
        use_ensemble = data.get("ensemble", True)
        fact_check_mode = data.get("fact_check_mode", "wikipedia")

        # ------------------------------------------------------------------
        # Step 1 — URL scraping
        # ------------------------------------------------------------------
        scraped_from_url = False
        source_url = None

        if is_url(raw_input):
            log.info(f"🌐 URL detected — scraping: {raw_input}")
            success, scraped_text = scrape_url(raw_input)
            if not success:
                # scraped_text is an error message in failure case
                return jsonify({"error": scraped_text, "success": False})
            source_url = raw_input
            text = scraped_text
            scraped_from_url = True
            log.info(f"   Scraped {len(text)} characters")
        else:
            text = raw_input

        # ------------------------------------------------------------------
        # Step 2 — Layer 0 validation
        # ------------------------------------------------------------------
        preprocessor = TextPreprocessor()
        is_valid, error_msg, warning_msg = validate_input(text, preprocessor)
        if not is_valid:
            return jsonify({"error": error_msg, "success": False})

        # ------------------------------------------------------------------
        # Step 3 — ML ensemble prediction
        # ------------------------------------------------------------------
        processed_text = preprocessor.preprocess(text)
        text_vectorized = vectorizer.transform([processed_text])

        all_predictions: dict = {}
        fake_votes = real_votes = 0
        sum_of_confidences = 0.0
        all_fake_probs: list = []
        all_real_probs: list = []

        for model_name, model in models.items():
            prediction = model.predict(text_vectorized)[0]
            pred_proba = model.predict_proba(text_vectorized)[0]

            # Normalise if un-normalised (edge case with some RF configs)
            total_prob = pred_proba[0] + pred_proba[1]
            if total_prob > 1.01:
                pred_proba = pred_proba / total_prob

            real_prob = float(pred_proba[0] * 100)
            fake_prob = float(pred_proba[1] * 100)
            all_fake_probs.append(fake_prob)
            all_real_probs.append(real_prob)

            all_predictions[model_name] = {
                "prediction": "FAKE NEWS" if prediction == 1 else "REAL NEWS",
                "confidence": fake_prob if prediction == 1 else real_prob,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
            }

            if prediction == 1:
                fake_votes += 1
                sum_of_confidences += fake_prob
            else:
                real_votes += 1
                sum_of_confidences += real_prob

        num_models = len(models)
        avg_fake = sum(all_fake_probs) / num_models
        avg_real = sum(all_real_probs) / num_models

        if fake_votes > real_votes:
            final_prediction = "FAKE NEWS"
            final_confidence = sum_of_confidences / fake_votes
            decision_type = "Majority Vote"
        elif real_votes > fake_votes:
            final_prediction = "REAL NEWS"
            final_confidence = sum_of_confidences / real_votes
            decision_type = "Majority Vote"
        else:
            # Tie: use average confidence as tie-breaker
            if avg_fake >= avg_real:
                final_prediction = "FAKE NEWS"
                final_confidence = avg_fake
            else:
                final_prediction = "REAL NEWS"
                final_confidence = avg_real
            decision_type = "Confidence Tie-Breaker (2-2 split)"

        result = {
            "success": True,
            "prediction": final_prediction,
            "confidence": round(final_confidence, 2),
            "fake_votes": fake_votes,
            "real_votes": real_votes,
            "decision_type": decision_type,
            "fake_probability": round(avg_fake, 2),
            "real_probability": round(avg_real, 2),
            "individual_results": all_predictions,
            "total_models": num_models,
            # URL scraping metadata (frontend can surface this)
            "scraped_from_url": scraped_from_url,
            "source_url": source_url,
        }

        if warning_msg:
            result["warning"] = warning_msg

        # ------------------------------------------------------------------
        # Step 4 — FactChecker (Layer 2) — skipped gracefully if None
        # ------------------------------------------------------------------
        if fact_checker is not None:
            try:
                google_api_key = getattr(config, "GOOGLE_FACT_CHECK_API_KEY", None)

                if fact_check_mode == "google" and google_api_key:
                    log.info("🌐 Running Google API fact-check mode")
                    temp_fc = FactChecker(google_api_key=google_api_key)
                    fc_result = temp_fc.analyze(text)

                    # Intelligent fallback: if Google returns nothing, use Wikipedia mode
                    if not fc_result.get("google_fact_checks") and not fc_result["warnings"]:
                        log.info("⚠️  Google returned no results — falling back to Wikipedia")
                        wiki_fc = FactChecker()
                        fallback = wiki_fc.analyze(text)
                        fc_result.update({
                            "warnings":            fallback["warnings"],
                            "numerical_issues":    fallback["numerical_issues"],
                            "scam_issues":         fallback["scam_issues"],
                            "factual_issues":      fallback["factual_issues"],
                            "verification_results": fallback["verification_results"],
                            "confidence_adjustment": fallback["confidence_adjustment"],
                            "entities":            fallback["entities"],
                            "fallback_used":       True,
                        })
                else:
                    log.info("🔍 Running Wikipedia fact-check mode")
                    fc_result = fact_checker.analyze(text)

                log.info(
                    f"📊 FactCheck complete — warnings:{len(fc_result['warnings'])} "
                    f"numerical:{len(fc_result['numerical_issues'])} "
                    f"scam:{len(fc_result['scam_issues'])}"
                )

                is_fake_ml = final_prediction == "FAKE NEWS"
                verdict = fact_checker.get_verdict(
                    ml_prediction=is_fake_ml,
                    ml_confidence=final_confidence,
                    fact_check_result=fc_result,
                )

                result["fact_check"] = {
                    "entities_found":        fc_result["entities"],
                    "numerical_issues":      fc_result["numerical_issues"],
                    "verification_results":  fc_result["verification_results"],
                    "warnings":              fc_result["warnings"],
                    "confidence_adjustment": fc_result["confidence_adjustment"],
                    "color":                 verdict.get("color", "green"),
                    "google_api_enabled":    fc_result.get("google_api_enabled", False),
                    "google_fact_checks":    fc_result.get("google_fact_checks", []),
                    "fallback_used":         fc_result.get("fallback_used", False),
                    "scam_issues":           fc_result.get("scam_issues", []),
                    "factual_issues":        fc_result.get("factual_issues", []),
                    "selected_mode":         fact_check_mode,
                }
                result["final_verdict"]     = verdict["verdict"]
                result["adjusted_confidence"] = verdict["adjusted_confidence"]
                result["fact_check_reason"] = verdict["reason"]
                result["verdict_color"]     = verdict.get("color", "green")

            except Exception as fc_err:
                log.warning(f"❌ FactChecker error (non-fatal): {fc_err}")
                result["fact_check"] = None
        else:
            result["fact_check"] = None

        # ------------------------------------------------------------------
        # Step 5 — Audit trail
        # ------------------------------------------------------------------
        logged_verdict     = result.get("final_verdict", result["prediction"])
        logged_confidence  = result.get("adjusted_confidence", result["confidence"])
        clean_snippet      = text[:50].replace("\n", " ").replace("\r", "") + "..."

        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                clean_snippet,
                logged_verdict,
                f"{logged_confidence}%",
                "No Feedback Yet",
            ])

        return jsonify(result)

    except Exception as e:
        log.exception("Unhandled error in /predict")
        return jsonify({"error": f"An error occurred: {str(e)}", "success": False})


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "available_models": list(models.keys()),
        "current_model": current_model_name,
        "fact_checker_active": fact_checker is not None,
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    user_thought = data.get("feedback", "unknown")
    log.info(f"📢 USER FEEDBACK: {user_thought}")
    return jsonify({"success": True, "message": "Thank you for helping me learn!"})


@app.route("/history")
def show_history():
    rows = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return jsonify(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)