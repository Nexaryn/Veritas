"""
Fact Checker Module — Veritas Layer 2
Verifies factual claims in news articles by:
  1. Extracting named entities via spaCy (NER)
  2. Cross-referencing organisations with Wikipedia
  3. Detecting unrealistic numerical claims (safe int/float casting)
  4. Scanning for scam / medical-misinformation language patterns
  5. Optionally querying the Google Fact Check API

Design principles in this version:
  - ALL regex → numeric conversions are wrapped in try/except to prevent crashes
  - Wikipedia and Google API calls carry strict timeouts + graceful fallbacks
  - check_factual_claims is a single, clean, merged method
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional

import requests
import spacy
import wikipediaapi

log = logging.getLogger(__name__)


class FactChecker:
    """Dual-API fact-checking layer for the Veritas fake-news detector."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialise spaCy NLP pipeline, Wikipedia API, and optionally the
        Google Fact Check API.

        Raises:
            OSError: if the spaCy model is missing (caller should handle).
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            raise

        self.wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="FakeNewsDetector/2.0 (Educational Project)",
        )

        # Google Fact Check API
        self.google_api_key = google_api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        self.use_google_api = bool(self.google_api_key)
        self.google_api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self._api_timeout = 7  # seconds for all external HTTP calls

        if self.use_google_api:
            log.info("✓ Google Fact Check API enabled")
        else:
            log.info("ℹ️  Google Fact Check API not configured (Wikipedia mode)")

        # ---------------------------------------------------------------
        # Numerical threshold checks
        # ---------------------------------------------------------------
        # Format: { label: (regex_pattern, max_threshold) }
        # Regex MUST have exactly 2 capture groups: (value, unit)
        self.numerical_checks = {
            "distance":   (r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometres?)", 500),
            "speed":      (r"(\d+(?:\.\d+)?)\s*(km/h|kmph|mph)", 350),
            "percentage": (r"(\d+(?:\.\d+)?)\s*(%)", 100),
        }

        # ---------------------------------------------------------------
        # Scam / chain-message patterns
        # ---------------------------------------------------------------
        self.scam_patterns: List[str] = [
            r"forward\s+this",
            r"share\s+(urgently|immediately|now|this|before)",
            r"before\s+it.?s\s+(deleted|removed|too\s+late)",
            r"(whatsapp|facebook|google|telegram)\s+will\s+charge",
            r"send\s+to\s+\d+\s+(people|contacts|friends)",
            r"turn\s+(blue|green|red)",
            r"don.?t\s+ignore",
            r"only\s+\d+\s+(hours?|minutes?|days?)\s+left",
            r"urgent(ly)?.*message",
            r"breaking.*!\s*",
            r"shocking.*!",
            r"click\s+here\s+(before|now)",
            r"register\s+(now|immediately).*expire",
            r"limited\s+time\s+offer",
            r"act\s+(fast|now|quickly)",
            r"big\s+pharma.*doesn.?t\s+want",
            r"doctors?\s+(don.?t\s+want|hate|hide)",
            r"this\s+(simple|one)\s+(trick|secret|remedy)",
            r"cure.*(cancer|diabetes|disease).*\d+\s+days",
            r"(ancient|hidden|secret).*cure",
            r"completely\s+cure[ds]?",
            r"no\s+need\s+for.*(treatment|medicine|surgery)",
            r"pharmaceutical.*trying\s+to\s+ban",
            r"god\s+bless",
            r"share.*everyone.*know",
        ]

        # ---------------------------------------------------------------
        # Medical misinformation (critical severity)
        # ---------------------------------------------------------------
        self.medical_scam_patterns: List[str] = [
            r"(cures?|heals?|treats?)\s+(all|any|every)",
            r"(cancer|diabetes|heart\s+disease).*cure[ds]?.*\d+\s+days",
            r"drinking\s+(hot\s+)?water.*cure",
            r"miracle\s+(cure|treatment|remedy)",
            r"doctors?\s+reveal[^\w]*",
            r"medical\s+breakthrough.*big\s+pharma",
            r"within\s+\d+\s+days.*cure[ds]?",
            r"stage\s+\d+\s+cancer.*completely\s+cure[ds]?",
            r"vaccine.*contain.*microchip",
            r"vaccine.*track",
            r"bill\s+gates.*vaccine",
            r"vaccine.*alter.*dna",
            r"vaccine.*magnetic",
            r"5g.*covid",
            r"covid.*hoax",
            r"thousands.*died.*vaccine",
        ]

        # ---------------------------------------------------------------
        # Suspicious announcement / unverified-claim patterns
        # ---------------------------------------------------------------
        self.suspicious_claim_patterns: List[str] = [
            r"announced?\s+(plans?|timeline)",
            r"(lead|senior)\s+(researcher|scientist|administrator)\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
            r"press\s+conference.*(?:yesterday|today|last\s+\w+)",
            r"scientists?\s+(revealed?|discovered?|announced?).*(?:compelling|shocking)",
            r"discovery.*(?:changes|revolutionizes)\s+everything",
            r"accelerate.*mission.*timeline.*based\s+on",
        ]

        # ---------------------------------------------------------------
        # Known factual ranges for physics / science checks
        # ---------------------------------------------------------------
        self.factual_ranges = {
            "mars_temperature":       (-125, -14),   # °C (surface avg range)
            "mars_water":             (0, 0),         # No confirmed surface liquid water
            "space_mission_years":    (2025, 2050),
            "cricket_world_cup_start": 1975,
            "womens_cricket_wc_start": 1973,
        }

        # Known sports facts (for future cross-reference)
        self.sports_facts = {
            "cricket_batters": [
                "smriti mandhana", "virat kohli", "rohit sharma", "sachin tendulkar",
            ],
            "cricket_bowlers": [
                "jasprit bumrah", "mitchell starc", "kagiso rabada",
            ],
        }

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from *text* using spaCy NER.

        Returns::

            {
                'organizations': [...],
                'locations':     [...],
                'dates':         [...],
                'infrastructure': [...],
            }
        """
        doc = self.nlp(text)
        entities: Dict[str, List[str]] = {
            "organizations": [],
            "locations":     [],
            "dates":         [],
            "infrastructure": [],
        }
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ in ("GPE", "LOC"):
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "FAC":
                entities["infrastructure"].append(ent.text)
        return entities

    # ------------------------------------------------------------------
    # Numerical claim checks  (safe casting)
    # ------------------------------------------------------------------

    def check_numerical_claims(self, text: str) -> List[Dict]:
        """
        Scan *text* for numerically unrealistic claims.

        All regex-extracted strings are cast via try/except so a bad
        match never crashes the server.
        """
        issues: List[Dict] = []

        for check_type, (pattern, threshold) in self.numerical_checks.items():
            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                # Each match is a 2-tuple (value_str, unit_str)
                raw_value = match[0] if isinstance(match, tuple) else match
                unit      = match[1] if isinstance(match, tuple) and len(match) > 1 else ""

                try:
                    value = float(raw_value)
                except (ValueError, TypeError):
                    log.debug(f"Skipping non-numeric match '{raw_value}' in {check_type}")
                    continue

                if check_type == "percentage":
                    if value > threshold or value < 0:
                        issues.append({
                            "type":   "invalid_percentage",
                            "value":  f"{value}{unit}",
                            "reason": f"Invalid percentage: {value}% (must be 0–100)",
                        })
                else:
                    if value > threshold:
                        issues.append({
                            "type":   f"unrealistic_{check_type}",
                            "value":  f"{value}{unit}",
                            "reason": (
                                f"Unrealistic {check_type}: {value}{unit} "
                                f"(threshold: {threshold})"
                            ),
                        })

        return issues

    # ------------------------------------------------------------------
    # Scam pattern detection
    # ------------------------------------------------------------------

    def check_scam_patterns(self, text: str) -> List[Dict]:
        """Detect chain-message, medical-scam, and suspicious-claim patterns."""
        issues: List[Dict] = []
        text_lower = text.lower()

        for pattern in self.scam_patterns:
            if re.search(pattern, text_lower):
                issues.append({
                    "type":     "scam_pattern",
                    "pattern":  pattern,
                    "reason":   "Contains typical scam / chain-message language",
                    "severity": "medium",
                })

        for pattern in self.medical_scam_patterns:
            if re.search(pattern, text_lower):
                issues.append({
                    "type":     "medical_scam",
                    "pattern":  pattern,
                    "reason":   "Contains dangerous medical misinformation pattern",
                    "severity": "critical",
                })

        for pattern in self.suspicious_claim_patterns:
            if re.search(pattern, text_lower):
                issues.append({
                    "type":     "suspicious_claim",
                    "pattern":  pattern,
                    "reason":   "Contains unverified claim pattern common in fake news",
                    "severity": "medium",
                })

        # Excessive ALL-CAPS
        words     = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 3]
        if words:
            caps_ratio = len(caps_words) / len(words)
            if caps_ratio > 0.3:
                issues.append({
                    "type":     "excessive_caps",
                    "pattern":  "ALL-CAPS",
                    "reason":   f"Excessive capital letters ({int(caps_ratio * 100)}% of text)",
                    "severity": "low",
                })

        # Excessive exclamation marks
        exclamation_count = text.count("!")
        if exclamation_count > 5:
            issues.append({
                "type":     "excessive_exclamation",
                "pattern":  "!!!",
                "reason":   f"Excessive exclamation marks ({exclamation_count} found)",
                "severity": "low",
            })

        return issues

    # ------------------------------------------------------------------
    # Wikipedia verification  (with timeout + graceful fallback)
    # ------------------------------------------------------------------

    def verify_on_wikipedia(
        self, entity: str, context_entities: List[str]
    ) -> Tuple[Optional[bool], str]:
        """
        Check whether *entity* exists on Wikipedia and optionally confirm
        that *context_entities* appear on the same page.

        Returns:
            (True, message)  — verified
            (None, message)  — uncertain / network issue
            (False, message) — definitely contradicted
        """
        try:
            entity_clean = re.sub(r"^(The|A|An)\s+", "", entity.strip(), flags=re.IGNORECASE)

            # Wikipedia API calls can block; we run them with a thread-based
            # timeout via requests' underlying socket timeout setting.
            # wikipediaapi uses requests under the hood, so we set a global
            # socket timeout before each call and restore it after.
            import socket
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self._api_timeout)

            try:
                page = self.wiki.page(entity)
                if not page.exists():
                    page = self.wiki.page(entity_clean)
            finally:
                socket.setdefaulttimeout(original_timeout)

            if not page.exists():
                return None, (
                    f"Could not verify '{entity}' on Wikipedia "
                    "(may be legitimate but not yet documented)"
                )

            page_text = page.text.lower()
            found_context = [c for c in context_entities if c.lower() in page_text]

            if context_entities and not found_context:
                return None, f"'{entity}' found on Wikipedia but context could not be confirmed"

            return True, f"'{entity}' verified on Wikipedia"

        except Exception as e:
            log.warning(f"Wikipedia lookup failed for '{entity}': {e}")
            return None, f"Could not check '{entity}' (Wikipedia temporarily unavailable)"

    # ------------------------------------------------------------------
    # Google Fact Check API  (with timeout + graceful fallback)
    # ------------------------------------------------------------------

    def check_google_fact_check(self, text: str) -> List[Dict]:
        """
        Query the Google Fact Check Tools API for claims matching *text*.

        Returns an empty list if the API is disabled, times out, or errors.
        """
        if not self.use_google_api:
            log.debug("Google Fact Check API not enabled (no API key)")
            return []

        query_text = text[:500]  # keep query short
        log.info(f"🌐 Querying Google Fact Check API (first 80 chars): {query_text[:80]}…")

        params = {
            "key":          self.google_api_key,
            "query":        query_text,
            "languageCode": "en",
        }

        try:
            response = requests.get(
                self.google_api_url,
                params=params,
                timeout=self._api_timeout,
            )
        except requests.exceptions.Timeout:
            log.warning("⚠️  Google Fact Check API timed out")
            return []
        except requests.exceptions.RequestException as e:
            log.warning(f"⚠️  Google Fact Check API request error: {e}")
            return []

        if response.status_code != 200:
            log.warning(f"⚠️  Google API returned HTTP {response.status_code}")
            return []

        try:
            data = response.json()
        except ValueError:
            log.warning("⚠️  Google API returned non-JSON response")
            return []

        claims = data.get("claims", [])
        log.info(f"   ✅ Google API found {len(claims)} claim(s)")

        fact_checks: List[Dict] = []
        for claim in claims[:3]:
            review = claim.get("claimReview", [{}])[0]
            fact_checks.append({
                "claim":     claim.get("text", "Unknown claim"),
                "rating":    review.get("textualRating", "Unknown"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "url":       review.get("url", ""),
            })

        return fact_checks

    # ------------------------------------------------------------------
    # Factual-claim checker  (single merged method, safe casting)
    # ------------------------------------------------------------------

    def check_factual_claims(self, text: str) -> List[Dict]:
        """
        Validate specific scientific / statistical claims against known data.

        Covers:
          • Mars surface temperature (must be within −125 to −14 °C)
          • Liquid water on Mars surface (currently unconfirmed)
          • Multiple unverifiable named scientists / doctors
          • Suspiciously precise but unverifiable statistics

        All regex → numeric conversions use try/except to prevent crashes.
        """
        factual_issues: List[Dict] = []
        text_lower = text.lower()

        # --- Mars surface temperature ---
        mars_temp_pattern = r"mars.*?temperature.*?(-?\d+(?:\.\d+)?)\s*°?\s*c\b"
        for temp_str in re.findall(mars_temp_pattern, text_lower):
            try:
                temp = float(temp_str)
            except (ValueError, TypeError):
                continue

            min_t, max_t = self.factual_ranges["mars_temperature"]
            if temp < min_t or temp > max_t:
                factual_issues.append({
                    "type":   "false_fact",
                    "claim":  f"Mars temperature {temp}°C",
                    "reason": (
                        f"Incorrect Mars temperature ({temp}°C is outside the "
                        f"realistic surface range of {min_t} to {max_t}°C)"
                    ),
                    "severity": "high",
                })

        # --- Train / vehicle speed safety check ---
        speed_pattern = r"train.*?(\d+(?:\.\d+)?)\s*(km/h|kmph|mph)"
        for speed_str, unit in re.findall(speed_pattern, text_lower):
            try:
                speed = float(speed_str)
            except (ValueError, TypeError):
                continue

            max_speed = 350  # km/h — realistic for high-speed rail
            if "mph" in unit:
                speed_kmh = speed * 1.60934
            else:
                speed_kmh = speed

            if speed_kmh > max_speed:
                factual_issues.append({
                    "type":   "unrealistic_speed",
                    "claim":  f"Train speed {speed}{unit}",
                    "reason": (
                        f"Claimed train speed ({speed}{unit}) exceeds realistic "
                        f"maximum for conventional rail ({max_speed} km/h)"
                    ),
                    "severity": "high",
                })

        # --- Liquid water on Mars surface ---
        if (
            "mars" in text_lower
            and "liquid water" in text_lower
            and "surface" in text_lower
            and re.search(r"(discover|found|flowing|liquid water).*surface", text_lower)
        ):
            factual_issues.append({
                "type":   "false_fact",
                "claim":  "Liquid water on Mars surface",
                "reason": (
                    "Claims liquid water on Mars surface — no confirmed liquid "
                    "surface water has been found"
                ),
                "severity": "high",
            })

        # --- Multiple unverifiable named sources ---
        person_matches = re.findall(
            r"(dr\.|prof\.|doctor|professor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            text,
        )
        if len(person_matches) >= 2:
            factual_issues.append({
                "type":   "unverified_sources",
                "claim":  "Multiple named sources",
                "reason": (
                    f"Contains {len(person_matches)} specific named sources "
                    "that cannot be independently verified"
                ),
                "severity": "medium",
            })

        # --- Suspiciously precise unverifiable statistics ---
        precise_stats = re.findall(
            r"\d+(?:\.\d+)?\s+(?:liters?|images?|samples?).*?(?:per|over|spanning)\s+\d+",
            text_lower,
        )
        if len(precise_stats) >= 2:
            factual_issues.append({
                "type":   "unverifiable_precision",
                "claim":  "Suspiciously precise statistics",
                "reason": (
                    "Contains multiple overly precise statistics that are "
                    "difficult to independently verify"
                ),
                "severity": "medium",
            })

        return factual_issues

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """
        Run all fact-checking layers and return a consolidated result dict.

        Returns::

            {
                'entities':              dict,
                'numerical_issues':      list[dict],
                'scam_issues':           list[dict],
                'factual_issues':        list[dict],
                'verification_results':  list[dict],
                'google_fact_checks':    list[dict],
                'google_api_enabled':    bool,
                'confidence_adjustment': int,   # capped at 50
                'warnings':              list[str],
            }
        """
        # --- Run all checks ---
        entities            = self.extract_entities(text)
        numerical_issues    = self.check_numerical_claims(text)
        scam_issues         = self.check_scam_patterns(text)
        factual_issues      = self.check_factual_claims(text)
        google_fact_checks  = self.check_google_fact_check(text)

        # --- Wikipedia verification for top organisations ---
        verification_results: List[Dict] = []
        warnings: List[str] = []

        for org in entities["organizations"][:2]:
            verified, message = self.verify_on_wikipedia(org, entities["locations"][:3])
            verification_results.append({
                "entity":   org,
                "verified": verified,
                "message":  message,
            })
            if verified is False:
                warnings.append(f"⚠️ {message}")

        # --- Confidence penalty calculation ---
        confidence_penalty = 0

        # Critical medical scams: −30 % each
        critical_issues = [s for s in scam_issues if s.get("severity") == "critical"]
        confidence_penalty += len(critical_issues) * 30

        # High-severity factual errors: −25 % each
        high_factual = [f for f in factual_issues if f.get("severity") == "high"]
        confidence_penalty += len(high_factual) * 25

        # Numerical anomalies: −10 % each
        confidence_penalty += len(numerical_issues) * 10

        # Medium scam patterns: −15 % each
        medium_scam = [s for s in scam_issues if s.get("severity") == "medium"]
        confidence_penalty += len(medium_scam) * 15

        # Medium factual issues: −12 % each
        medium_factual = [f for f in factual_issues if f.get("severity") == "medium"]
        confidence_penalty += len(medium_factual) * 12

        # Low-severity style issues: −5 % each
        low_scam = [s for s in scam_issues if s.get("severity") == "low"]
        confidence_penalty += len(low_scam) * 5

        # Failed Wikipedia verifications: −15 % each
        failed_verifications = sum(
            1 for v in verification_results if v["verified"] is False
        )
        confidence_penalty += failed_verifications * 15

        # --- Build human-readable warnings list ---
        for issue in numerical_issues:
            warnings.append(f"⚠️ {issue['reason']}")

        if critical_issues:
            warnings.insert(0,
                f"🚨 DANGER: Detected {len(critical_issues)} dangerous "
                "medical misinformation pattern(s)"
            )

        insert_pos = 1 if critical_issues else 0
        for issue in high_factual:
            warnings.insert(insert_pos, f"❌ FALSE CLAIM: {issue['reason']}")
            insert_pos += 1

        for issue in medium_factual:
            warnings.append(f"⚠️ {issue['reason']}")

        # Scam pattern examples (at most one per type, max 3 added)
        scam_types_shown: set = set()
        scam_examples: List[str] = []
        for issue in scam_issues:
            issue_type = issue.get("type", "unknown")
            if issue_type in scam_types_shown:
                continue
            scam_types_shown.add(issue_type)

            pattern = issue.get("pattern", "")
            if issue_type in ("suspicious_claim", "scam_pattern"):
                m = re.search(pattern, text.lower())
                label = "Unverified claim" if issue_type == "suspicious_claim" else "Scam language"
                if m:
                    scam_examples.append(f'⚠️ {label} detected: "{m.group(0)}"')
                else:
                    scam_examples.append(f"⚠️ {label} pattern detected")
            elif issue_type in ("excessive_caps", "excessive_exclamation"):
                scam_examples.append(f"⚠️ {issue['reason']}")

        warnings.extend(scam_examples[:3])

        # Google fact-check warnings (prepended so they're prominent)
        for fc in google_fact_checks:
            rating = fc.get("rating", "").lower()
            if any(kw in rating for kw in ("false", "misleading", "incorrect", "inaccurate", "disputed")):
                publisher = fc.get("publisher", "Fact-checker")
                claim_text = fc.get("claim", "Unknown")[:80]
                warnings.insert(0,
                    f'🌐 {publisher}: "{claim_text}" rated as {fc.get("rating", "disputed")}'
                )
                confidence_penalty += 20

        return {
            "entities":              entities,
            "numerical_issues":      numerical_issues,
            "scam_issues":           scam_issues,
            "factual_issues":        factual_issues,
            "verification_results":  verification_results,
            "google_fact_checks":    google_fact_checks,
            "google_api_enabled":    self.use_google_api,
            "confidence_adjustment": min(confidence_penalty, 50),  # hard cap at 50 %
            "warnings":              warnings[:8],                  # max 8 displayed
        }

    # ------------------------------------------------------------------
    # Verdict combiner
    # ------------------------------------------------------------------

    def get_verdict(
        self,
        ml_prediction: bool,
        ml_confidence: float,
        fact_check_result: Dict,
    ) -> Dict:
        """
        Combine the ML prediction with Layer 2 fact-checking results.

        Rule:
          • Any fact-checker warning → FAKE NEWS (override ML), confidence 95 %
          • No warnings + ML says FAKE → FAKE NEWS at ML confidence
          • No warnings + ML says REAL → REAL NEWS at ML confidence
        """
        warnings = fact_check_result.get("warnings", [])

        if warnings:
            return {
                "verdict":             "FAKE NEWS",
                "adjusted_confidence": 95,
                "color":               "red",
                "reason":              "🚨 FACT-CHECKER OVERRIDE: " + warnings[0],
            }

        if ml_prediction:
            return {
                "verdict":             "FAKE NEWS",
                "adjusted_confidence": ml_confidence,
                "color":               "red",
                "reason":              "ML ensemble detected fake news patterns",
            }

        return {
            "verdict":             "REAL NEWS",
            "adjusted_confidence": ml_confidence,
            "color":               "green",
            "reason":              "✓ No suspicious patterns detected",
        }