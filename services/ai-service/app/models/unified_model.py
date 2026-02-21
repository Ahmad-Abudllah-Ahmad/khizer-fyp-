"""
SereneMind — Unified Mental Health Analyzer v3
===============================================
Inference layer for the epoch-trained unified model.

Key improvements over v2:
  - Supports the new bundle format (vectorizer + classifier + label_encoder)
  - Long-text chunking: paragraphs are split into overlapping windows and 
    probabilities are averaged across chunks (chunk-and-aggregate ensemble)
  - 3-tier reliability bridge retained and expanded
  - No hard 1000-char truncation — handles full journal entries
"""

import joblib
import logging
import re
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

# ─── Severity Weights per Class ──────────────────────────────────────────────
SEVERITY_WEIGHTS = {
    "crisis":     10,
    "depression":  8,
    "anxiety":     7,
    "grief":       7,
    "anger":       6,
    "fear":        6,
    "stress":      5,
    "normal":      1,
    "joy":         0,
}

# ─── Emotional Tone Tags per Class ───────────────────────────────────────────
CLASS_TAGS = {
    "crisis":     ["suicidal ideation", "self-harm risk", "hopelessness", "acute distress"],
    "depression": ["low mood", "anhedonia", "hopelessness", "fatigue", "worthlessness"],
    "anxiety":    ["worry", "panic", "nervousness", "avoidance", "hypervigilance"],
    "stress":     ["overwhelm", "burnout", "pressure", "overload", "exhaustion"],
    "grief":      ["loss", "mourning", "bereavement", "sadness", "emptiness"],
    "anger":      ["frustration", "irritability", "rage", "resentment"],
    "fear":       ["dread", "terror", "phobia", "avoidance"],
    "joy":        ["contentment", "gratitude", "optimism", "positivity"],
    "normal":     ["stable", "balanced", "neutral"],
}

# ─── Emotion mapping (from mental state → primary emotion) ───────────────────
STATE_TO_EMOTION = {
    "crisis":     "sadness",
    "depression": "sadness",
    "anxiety":    "fear",
    "grief":      "sadness",
    "anger":      "anger",
    "fear":       "fear",
    "stress":     "fear",
    "joy":        "joy",
    "normal":     "neutral",
}

# ─── Display names ────────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "crisis":     "Crisis",
    "depression": "Depression",
    "anxiety":    "Anxiety",
    "grief":      "Grief",
    "anger":      "Anger",
    "fear":       "Fear",
    "stress":     "Stress",
    "joy":        "Joy",
    "normal":     "Stable",
}

# ─── Crisis keyword override ──────────────────────────────────────────────────
# Tier 1: EXPLICIT crisis — these ALWAYS override to CRISIS
EXPLICIT_CRISIS_KEYWORDS = [
    # Suicidal ideation — direct
    "kill myself", "kill my self", "killing myself",
    "end my life", "end myself", "end my self",
    "want to die", "wanna die", "going to die",
    "don't want to live", "do not want to live", "don't wanna live",
    "no reason to live", "nothing to live for",
    "better off dead", "better off without me",
    "suicide", "suicidal",
    # Method — active intent
    "hang myself", "hang my self",
    "overdose", "take pills to die",
    "jump off", "jump from", "going to jump",
    "taking my life", "take my life", "take my own life",
    # Physical self-harm with intent
    "cut myself", "cutting myself", "slit my wrist",
    "hurt myself on purpose", "want to hurt myself",
    "hit myself", "hit my self",
    "harm myself", "self harm",
    "shooting myself",
    # Escape / disappear
    "disappear forever", "want to disappear",
    "end it all", "end everything",
    # Want to be dead
    "wish i was dead", "wish i were dead",
    "i am dying", "i will die", "i am going to die",
]

# Tier 2: HIGH-confidence implicit crisis — indirect intent signals
IMPLICIT_CRISIS_SIGNALS = [
    # Rooftop / building / jumping
    "top of building", "rooftop", "roof top",
    "jumped off", "jumping off", "jump off a bridge",
    "edge of", "standing on the edge",
    # Self-harm without "myself"
    "want to jump", "going to jump", "about to jump",
    "with rod", "with a rod", "with knife", "with a knife",
    "stab myself", "punch myself", "bang my head",
    # Pain-based death wish
    "pain is killing me", "can't take the pain", "too much pain to live",
    "pain won't stop", "unbearable pain",
    # Hopeless intent
    "no point in living", "see no point",
    "life is not worth", "not worth living",
]

# Tier 3: Elevated distress — push MEDIUM risk minimum if model says LOW
DISTRESS_SIGNALS = [
    # Physical trauma & accident
    "terrible pain", "too much pain", "in so much pain", "so much pain",
    "blood", "bleeding", "face hit", "face smashed",
    "car accident", "hit by car", "road accident",
    "neck pain", "couldn't move", "can't move", "could not move",
    "hospital", "emergency", "ambulance", "rescue",
    "numb", "shock", "traumatised", "traumatized",
    # Injured — many phrasings
    "injured", "got injured", "was injured", "were injured", "badly injured",
    "got hurt", "was hurt", "were hurt",
    "pain in my", "unbearable",
    # Witnessing trauma
    "accident in front", "accident ahead", "saw an accident",
    "someone was hit", "people injured",
    # Medical deterioration
    "condition declining", "condition worsening", "getting worse", "not improving",
    "fever", "vomiting", "fainting", "collapsed", "unconscious",
    # Psychological numbness after trauma
    "mind went blank", "mind didn't work", "lost control", "lose control",
    "out of control", "get out of control",
]


def _clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+|<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s'?!.,;:-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_severity(label: str, crisis_prob: float, all_scores: dict,
                       implicit_crisis: bool = False, distress: bool = False) -> int:
    """Compute severity rating 1-10 with reliability bridge floors."""
    base = SEVERITY_WEIGHTS.get(label, 3)
    crisis_signal = all_scores.get("crisis", 0.0) * 5
    joy_dampen    = all_scores.get("joy",    0.0) * 3
    severity = base + crisis_signal - joy_dampen
    if implicit_crisis:
        severity = max(severity, 8)
    elif distress:
        severity = max(severity, 5)
    return max(1, min(10, round(severity)))


def _get_contextual_tags(label: str, text: str, all_scores: dict) -> List[str]:
    tags = list(CLASS_TAGS.get(label, ["stable"]))
    text_lower = text.lower()

    if all_scores.get("depression", 0) > 0.20 and label != "depression":
        tags.append("depressive symptoms")
    if all_scores.get("anxiety", 0) > 0.20 and label != "anxiety":
        tags.append("anxiety symptoms")
    if all_scores.get("crisis", 0) > 0.15 and label != "crisis":
        tags.append("crisis indicators")

    if any(k in text_lower for k in ["alone", "lonely", "isolat"]):
        tags.append("social isolation")
    if any(k in text_lower for k in ["sleep", "insomnia", "can't sleep"]):
        tags.append("sleep disturbance")
    if any(k in text_lower for k in ["work", "job", "career", "boss"]):
        tags.append("work-related stress")
    if any(k in text_lower for k in ["relationship", "partner", "breakup", "divorce"]):
        tags.append("relationship difficulty")
    if any(k in text_lower for k in ["family", "parent", "mother", "father"]):
        tags.append("family dynamics")
    if any(k in text_lower for k in ["accident", "hospital", "injured", "bleeding", "rescue"]):
        tags.append("physical trauma")

    seen = set(); result = []
    for t in tags:
        if t not in seen:
            seen.add(t); result.append(t)
        if len(result) >= 5: break
    return result


def _semantic_summary(label: str, emotion: str, severity: int, confidence: float, text: str) -> str:
    summaries = {
        "crisis":     f"Text expresses severe psychological distress with crisis indicators (severity {severity}/10). Immediate support is strongly recommended.",
        "depression": f"Text reflects depressive patterns including low mood and reduced engagement (severity {severity}/10). Professional support may be beneficial.",
        "anxiety":    f"Text shows signs of anxiety, worry or panic (severity {severity}/10). Grounding and support strategies are advisable.",
        "stress":     f"Text indicates high stress or overwhelm (severity {severity}/10). Rest and stress management are recommended.",
        "grief":      f"Text reflects grief or significant loss (severity {severity}/10). Compassionate support is appropriate.",
        "anger":      f"Text expresses anger or frustration (severity {severity}/10). De-escalation support may help.",
        "fear":       f"Text reflects fear or dread (severity {severity}/10). Reassurance and safety techniques are helpful.",
        "joy":        f"Text expresses positive emotional state (severity {severity}/10). Continue supportive engagement.",
        "normal":     f"Text appears emotionally stable (severity {severity}/10). Routine wellness check-in is appropriate.",
    }
    return summaries.get(label, f"Emotional state: {label}, severity {severity}/10.")


# ─── Long-Text Chunking ───────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_words: int = 60, overlap_words: int = 20) -> List[str]:
    """
    Split a long text into overlapping word-level chunks.
    This allows the model to analyze full paragraphs by aggregating
    chunk-level probability predictions (chunk-and-aggregate ensemble).
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text]  # Short text — single chunk
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_words])
        if len(chunk) >= 10:
            chunks.append(chunk)
    return chunks if chunks else [text]


class UnifiedMentalHealthAnalyzer:
    """
    Unified model for full mental health semantic analysis.
    Supports both old (pipeline) and new (bundle dict) model formats.

    Prediction output:
    {
        "mental_state":              "Depression",
        "raw_label":                 "depression",
        "emotion":                   "sadness",
        "crisis_risk":               "HIGH",
        "crisis_probability":        0.72,
        "severity_rating":           8,
        "tags":                      ["hopelessness", "social isolation"],
        "confidence":                0.88,
        "all_scores":                { "depression": 0.72, ... },
        "requires_immediate_action": True,
        "semantic_summary":          "Text reflects...",
        "triggered_by":              "unified_model",
    }
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            bundle = joblib.load(self.model_path)

            # Support new bundle format (dict with vectorizer + classifier + calibrators)
            if isinstance(bundle, dict) and "vectorizer" in bundle:
                self.vectorizer   = bundle["vectorizer"]
                self.classifier   = bundle["classifier"]
                self.calibrators  = bundle.get("calibrators", None)  # list of IsotonicRegression
                self.le           = bundle["label_encoder"]
                self.classes_     = list(self.le.classes_)
                logger.info(
                    f"✅ Unified Mental Health model v3 loaded — {len(self.classes_)} classes "
                    f"(epoch={bundle.get('epochs_run','?')}, val_acc={bundle.get('val_accuracy',0):.1f}%)"
                )
                self._use_bundle = True
            else:
                # Legacy: sklearn pipeline with predict_proba
                self.pipeline     = bundle
                self.calibrators  = None
                self.classes_     = list(bundle.classes_)
                self._use_bundle  = False
                logger.info(f"✅ Unified Mental Health model (legacy) loaded — {len(self.classes_)} classes")
        except Exception as e:
            logger.error(f"Unified model load failed: {e}")
            raise

    def _predict_proba_raw(self, texts: list) -> np.ndarray:
        """Predict calibrated probabilities for a list of clean texts."""
        if self._use_bundle:
            X   = self.vectorizer.transform(texts)
            raw = self.classifier.predict_proba(X)
            if self.calibrators is not None:
                cal = np.zeros_like(raw)
                for i, ir in enumerate(self.calibrators):
                    cal[:, i] = ir.predict(raw[:, i])
                row_s = cal.sum(axis=1, keepdims=True)
                return cal / np.maximum(row_s, 1e-9)
            return raw
        else:
            return self.pipeline.predict_proba(texts)

    def predict(self, text: str) -> Dict:
        try:
            text_lower = text.lower()

            # ── Chunk-and-aggregate for long texts ────────────────────────────
            chunks       = _chunk_text(_clean(text))
            clean_chunks = [_clean(c) for c in chunks]
            chunk_probas = self._predict_proba_raw(clean_chunks)  # shape: (n_chunks, n_classes)
            # Weighted average: later chunks (conclusion) get slightly higher weight
            weights = np.linspace(0.8, 1.2, len(clean_chunks))
            weights /= weights.sum()
            avg_proba = (chunk_probas * weights[:, None]).sum(axis=0)

            # Build all_scores dict
            all_scores = {
                cls: round(float(avg_proba[i]), 4)
                for i, cls in enumerate(self.classes_)
            }
            top_idx   = int(avg_proba.argmax())
            top_label = self.classes_[top_idx]
            confidence = float(avg_proba[top_idx])

            # ─── RELIABILITY BRIDGE — 3 Tiers ────────────────────────────────

            # Tier 1: EXPLICIT crisis keywords → always CRISIS
            explicit_crisis = any(kw in text_lower for kw in EXPLICIT_CRISIS_KEYWORDS)
            # Tier 2: Implicit crisis signals → force CRISIS (0.75+ prob)
            implicit_crisis = any(kw in text_lower for kw in IMPLICIT_CRISIS_SIGNALS)
            # Tier 3: Elevated distress signals → raise floor to MEDIUM at minimum
            distress_signal = any(kw in text_lower for kw in DISTRESS_SIGNALS)

            if explicit_crisis:
                logger.info("Reliability bridge Tier 1: explicit crisis keyword → overriding to crisis")
                all_scores["crisis"] = max(all_scores.get("crisis", 0.0), 0.90)
                top_label  = "crisis"
                confidence = all_scores["crisis"]

            elif implicit_crisis:
                logger.info("Reliability bridge Tier 2: implicit crisis signal → overriding to crisis")
                all_scores["crisis"] = max(all_scores.get("crisis", 0.0), 0.75)
                top_label  = "crisis"
                confidence = all_scores["crisis"]

            elif distress_signal and top_label == "normal":
                logger.info("Reliability bridge Tier 3: distress signal → bumping from stable")
                non_normal = {k: v for k, v in all_scores.items() if k != "normal"}
                if non_normal:
                    best_alt   = max(non_normal, key=lambda k: non_normal[k])
                    top_label  = best_alt
                    confidence = all_scores[best_alt]
                all_scores["crisis"] = max(all_scores.get("crisis", 0.0), 0.20)

            # ─ Crisis probability ─────────────────────────────────────────────
            crisis_prob = all_scores.get("crisis", 0.0)
            if explicit_crisis:
                crisis_prob = max(crisis_prob, 0.90)
            elif implicit_crisis:
                crisis_prob = max(crisis_prob, 0.75)
            elif distress_signal:
                crisis_prob = max(crisis_prob, 0.20)

            # ─ Risk level mapping ─────────────────────────────────────────────
            if crisis_prob >= 0.60:
                risk_level = "CRISIS"; requires_action = True
            elif crisis_prob >= 0.35:
                risk_level = "HIGH";   requires_action = True
            elif crisis_prob >= 0.18:
                risk_level = "MEDIUM"; requires_action = False
            else:
                risk_level = "LOW";    requires_action = False

            # ─ Severity, tags, summary ────────────────────────────────────────
            severity = _compute_severity(top_label, crisis_prob, all_scores,
                                         implicit_crisis=implicit_crisis,
                                         distress=distress_signal)
            emotion  = STATE_TO_EMOTION.get(top_label, "neutral")
            tags     = _get_contextual_tags(top_label, text, all_scores)
            summary  = _semantic_summary(top_label, emotion, severity, confidence, text)

            return {
                "mental_state":              DISPLAY_NAMES.get(top_label, top_label.capitalize()),
                "raw_label":                 top_label,
                "emotion":                   emotion,
                "crisis_risk":               risk_level,
                "crisis_probability":        round(crisis_prob, 4),
                "severity_rating":           severity,
                "tags":                      tags,
                "confidence":                round(confidence, 4),
                "all_scores":                all_scores,
                "requires_immediate_action": requires_action,
                "semantic_summary":          summary,
                "triggered_by":              "unified_model",
            }

        except Exception as e:
            logger.error(f"Unified analysis failed: {e}", exc_info=True)
            return {
                "mental_state":              "Stable",
                "raw_label":                 "normal",
                "emotion":                   "neutral",
                "crisis_risk":               "LOW",
                "crisis_probability":        0.02,
                "severity_rating":           1,
                "tags":                      ["stable"],
                "confidence":                0.5,
                "all_scores":                {},
                "requires_immediate_action": False,
                "semantic_summary":          "Unable to analyze. Default stable state.",
                "triggered_by":              "fallback",
            }
