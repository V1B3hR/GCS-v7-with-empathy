"""
Empathy Engine for GCS-v7-with-empathy System (Enhanced)

This module implements the core empathy capabilities including:
- Empathy-aware emotion recognition with optional Hugging Face GoEmotions
- Personalized empathetic response generation
- Cultural sensitivity and adaptation
- Ethical boundary management & crisis risk scoring
- Privacy protection for emotional data with persistent key option
- VAD (Valence, Arousal, Dominance) feature modeling
"""

import logging
import os
import re
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum,

import numpy as np
import tensorflow as tf  # retained for compatibility with existing base classifier
from cryptography.fernet import Fernet

# Optional advanced emotion model via Hugging Face (GoEmotions)
try:
    from transformers import pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class EmotionalState(str, Enum):
    """Extended emotional states for empathy-aware classification."""

    # Positive emotions
    JOY = "joy"
    HAPPINESS = "happiness"
    GRATITUDE = "gratitude"
    LOVE = "love"
    HOPE = "hope"
    PRIDE = "pride"
    AMUSEMENT = "amusement"
    INTEREST = "interest"
    INSPIRATION = "inspiration"
    OPTIMISM = "optimism"
    TRUST = "trust"
    PEACEFULNESS = "peacefulness"
    CONFIDENCE = "confidence"
    SATISFACTION = "satisfaction"
    AFFECTION = "affection"
    ADMIRATION = "admiration"
    CONTENTMENT = "contentment"
    EXCITEMENT = "excitement"
    RELIEF = "relief"

    # Negative and neutral emotions (optional)
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    LONELINESS = "loneliness"
    STRESS = "stress"

class CulturalContext(Enum):
    """Cultural contexts for empathy adaptation"""
    INDIVIDUALISTIC = "individualistic"
    COLLECTIVISTIC = "collectivistic"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"
    UNCERTAINTY_AVOIDING = "uncertainty_avoiding"
    UNCERTAINTY_ACCEPTING = "uncertainty_accepting"


class EmpathyIntensity(Enum):
    """Empathy response intensity levels"""
    MINIMAL = "minimal"
    GENTLE = "gentle"
    MODERATE = "moderate"
    STRONG = "strong"
    INTENSIVE = "intensive"


@dataclass
class EmotionalProfile:
    """Individual emotional profile for personalized empathy"""
    user_id: str
    baseline_valence: float = 0.0   # -1.0 to 1.0
    baseline_arousal: float = 0.5   # 0.0 to 1.0
    baseline_dominance: float = 0.5 # 0.0 to 1.0
    emotional_sensitivity: float = 0.5  # 0.0 to 1.0
    preferred_support_style: str = "balanced"  # balanced, practical, gentle
    cultural_context: CulturalContext = CulturalContext.INDIVIDUALISTIC
    empathy_preferences: Dict[str, Any] = field(default_factory=dict)
    crisis_indicators: List[str] = field(default_factory=list)
    therapeutic_goals: List[str] = field(default_factory=list)
    consent_level: str = "basic"  # basic, therapeutic, research
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class EmpathicResponse:
    """Empathetic response with metadata"""
    content: str
    intensity: EmpathyIntensity
    response_type: str  # validation, support, guidance, intervention
    cultural_adaptation: str
    confidence: float
    therapeutic_alignment: bool
    privacy_level: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EmpathyMetrics:
    """Metrics for measuring empathy effectiveness"""
    user_satisfaction: float = 0.0  # 0.0 to 1.0
    emotional_improvement: float = 0.0  # -1.0 to 1.0
    engagement_level: float = 0.0  # 0.0 to 1.0
    therapeutic_progress: float = 0.0  # 0.0 to 1.0
    cultural_appropriateness: float = 0.0  # 0.0 to 1.0
    safety_compliance: bool = True
    measurement_timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------
# VAD utilities and mappings
# ---------------------------

_NEUTRAL_VAD = (0.0, 0.5, 0.5)

# Approximate VAD mappings for GoEmotions categories. Values are heuristic but bounded.
_GOEMOTIONS_VAD: Dict[str, Tuple[float, float, float]] = {
    # Positive
    "admiration": (0.6, 0.5, 0.6),
    "amusement": (0.7, 0.6, 0.5),
    "approval": (0.5, 0.4, 0.6),
    "caring": (0.7, 0.4, 0.6),
    "curiosity": (0.2, 0.6, 0.6),
    "desire": (0.5, 0.6, 0.6),
    "excitement": (0.8, 0.8, 0.6),
    "gratitude": (0.8, 0.5, 0.6),
    "joy": (0.9, 0.7, 0.6),
    "love": (0.85, 0.5, 0.6),
    "optimism": (0.6, 0.5, 0.6),
    "pride": (0.6, 0.6, 0.6),
    "relief": (0.5, 0.3, 0.5),
    # Neutral
    "neutral": (0.0, 0.5, 0.5),
    "realization": (0.1, 0.5, 0.5),
    # Negative
    "anger": (-0.8, 0.8, 0.7),
    "annoyance": (-0.4, 0.6, 0.6),
    "disapproval": (-0.5, 0.6, 0.6),
    "disappointment": (-0.6, 0.4, 0.4),
    "disgust": (-0.8, 0.5, 0.6),
    "embarrassment": (-0.4, 0.6, 0.3),
    "fear": (-0.8, 0.9, 0.3),
    "grief": (-0.9, 0.6, 0.3),
    "nervousness": (-0.6, 0.8, 0.3),
    "remorse": (-0.6, 0.5, 0.4),
    "sadness": (-0.8, 0.3, 0.3),
    "surprise": (0.2, 0.8, 0.5),
    "confusion": (-0.2, 0.6, 0.4),
}


def _weighted_vad_from_scores(scores: Dict[str, float]) -> Tuple[float, float, float]:
    """Compute VAD by weighting category VADs with probabilities."""
    if not scores:
        return _NEUTRAL_VAD
    v = a = d = 0.0
    total = 1e-8
    for label, p in scores.items():
        vad = _GOEMOTIONS_VAD.get(label.lower())
        if vad is None:
            continue
        v += vad[0] * p
        a += vad[1] * p
        d += vad[2] * p
        total += p
    return (
        float(np.clip(v / total, -1.0, 1.0)),
        float(np.clip(a / total, 0.0, 1.0)),
        float(np.clip(d / total, 0.0, 1.0)),
    )


class EmpathyPrivacyGuard:
    """Privacy protection for emotional data"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        # Allow persistent key via env var to keep data decryptable across restarts.
        # Value must be a valid Fernet key (base64 url-safe 32 bytes).
        if encryption_key is None:
            key_env = os.getenv("EMPATHY_ENGINE_KEY")
            if key_env:
                try:
                    encryption_key = key_env.encode("utf-8")
                except Exception:
                    logging.warning("EMPATHY_ENGINE_KEY present but invalid; generating ephemeral key")
                    encryption_key = Fernet.generate_key()
            else:
                encryption_key = Fernet.generate_key()
        self.fernet = Fernet(encryption_key)
        self.access_logs: List[Dict[str, Any]] = []

    def encrypt_emotional_data(self, data: Dict[str, Any], user_id: str) -> str:
        """Encrypt emotional data with user-specific protection"""
        protected_data = {
            "data": data,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "access_hash": self._generate_access_hash(user_id),
        }
        json_data = json.dumps(protected_data, default=str)
        encrypted_data = self.fernet.encrypt(json_data.encode())
        self._log_access(user_id, "encrypt", "emotional_data")
        return encrypted_data.decode()

    def decrypt_emotional_data(self, encrypted_data: str, user_id: str, purpose: str = "therapeutic") -> Dict[str, Any]:
        """Decrypt emotional data with purpose validation"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            data_dict = json.loads(decrypted_data.decode())
            if data_dict["user_id"] != user_id:
                raise ValueError("User ID mismatch - unauthorized access attempt")
            self._log_access(user_id, "decrypt", purpose)
            return data_dict["data"]
        except Exception as e:
            logging.error(f"Failed to decrypt emotional data: {e}")
            self._log_access(user_id, "decrypt_failed", purpose)
            raise

    def _generate_access_hash(self, user_id: str) -> str:
        return hashlib.sha256(f"{user_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def _log_access(self, user_id: str, action: str, purpose: str):
        self.access_logs.append({
            "user_id": user_id,
            "action": action,
            "purpose": purpose,
            "timestamp": datetime.now().isoformat(),
        })


class EmpathyEthicsGuard:
    """Ethical boundaries and safety protocols for empathy"""

    CRISIS_PATTERNS = [
        r"\bkill myself\b", r"\bsuicide\b", r"\bend it all\b", r"\bself[-\s]?harm\b",
        r"\bnot worth living\b", r"\bbetter off dead\b", r"\bhurt myself\b",
        r"\bcan'?t go on\b", r"\bno reason to live\b", r"\bwant to die\b",
    ]

    def __init__(self):
        self.ethical_boundaries = {
            "no_manipulation": True,
            "preserve_autonomy": True,
            "professional_boundaries": True,
            "crisis_referral_mandatory": True,
            "cultural_respect": True,
            "informed_consent_required": True,
        }
        self.crisis_keywords = [
            "suicide", "self-harm", "hurt myself", "end it all", "not worth living", "kill myself", "better off dead",
            "can’t go on", "cant go on", "want to die", "no reason to live",
        ]
        self.professional_referral_triggers = [
            "severe_depression", "suicidal_ideation", "psychosis", "substance_abuse", "domestic_violence", "child_abuse",
        ]

    def validate_empathic_response(self, response: EmpathicResponse, user_profile: EmotionalProfile, emotional_state: EmotionalState) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        if self._contains_manipulation(response.content):
            violations.append("Response contains manipulative language")
        if not self._respects_professional_boundaries(response.content):
            violations.append("Response crosses professional boundaries")
        if not self._is_culturally_appropriate(response.content, user_profile.cultural_context):
            violations.append("Response not culturally appropriate")
        if self._requires_crisis_intervention(response.content, emotional_state):
            violations.append("Crisis intervention required - immediate professional referral needed")
        return len(violations) == 0, violations

    def detect_crisis_state(self, text_input: str, emotional_metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Detect if user is in crisis state requiring immediate intervention.
        Returns (crisis_detected, risk_score [0..1])."""
        text_lower = text_input.lower()
        # Keyword/pattern risk
        kw_hits = sum(1 for k in self.crisis_keywords if k in text_lower)
        pattern_hits = sum(1 for p in self.CRISIS_PATTERNS if re.search(p, text_lower))
        keyword_risk = min(1.0, 0.25 * kw_hits + 0.35 * pattern_hits)
        # Metric risk
        val = float(emotional_metrics.get("valence", 0.0))
        aro = float(emotional_metrics.get("arousal", 0.5))
        metric_risk = 0.0
        if val < -0.8 and aro < 0.4:
            metric_risk = max(metric_risk, 0.6)  # severe depression signature
        if aro > 0.9 and val < -0.5:
            metric_risk = max(metric_risk, 0.5)  # extreme anxiety signature
        risk = float(np.clip(keyword_risk + metric_risk, 0.0, 1.0))
        return (risk >= 0.6), risk

    def _contains_manipulation(self, content: str) -> bool:
        indicators = [
            "you should feel", "you must", "everyone thinks", "if you really cared", "good people always",
            "prove that you", "only a fool would", "no choice but",
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in indicators)

    def _respects_professional_boundaries(self, content: str) -> bool:
        boundary_violations = [
            "i can cure", "you are diagnosed", "take this medication", "i know what's best", "trust me completely",
            "promise you'll", "i guarantee you'll",
        ]
        content_lower = content.lower()
        return not any(v in content_lower for v in boundary_violations)

    def _is_culturally_appropriate(self, content: str, cultural_context: CulturalContext) -> bool:
        content_lower = content.lower()
        if cultural_context == CulturalContext.COLLECTIVISTIC:
            problematic = ["focus on yourself", "you alone can", "independent choice"]
            if any(p in content_lower for p in problematic):
                return False
        return True

    def _requires_crisis_intervention(self, content: str, emotional_state: EmotionalState) -> bool:
        crisis_states = [EmotionalState.DEPRESSION, EmotionalState.ANXIETY]
        crisis_content_indicators = ["overwhelming", "can't cope", "hopeless", "no way out"]
        if emotional_state in crisis_states:
            content_lower = content.lower()
            return any(ind in content_lower for ind in crisis_content_indicators)
        return False


class _HFEmotionAdapter:
    """Optional Hugging Face GoEmotions adapter for text emotion inference."""

    def __init__(self, model_name: str = "joeddav/distilbert-base-uncased-go-emotions-student") -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        # Multi-label classification with all scores
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            return_all_scores=True,
            truncation=True
        )

    def infer_vad(self, text: str) -> Tuple[float, float, float, Dict[str, float]]:
        outputs = self.pipe(text)
        # outputs: [ [ {'label': '...', 'score': p}, ... ] ]
        scores: Dict[str, float] = {}
        for item in outputs[0]:
            scores[item["label"].lower()] = float(item["score"])
        v, a, d = _weighted_vad_from_scores(scores)
        return v, a, d, scores


class EmpathicResponseGenerator:
    """Generates personalized empathetic responses"""

    def __init__(self, ethics_guard: EmpathyEthicsGuard):
        self.ethics_guard = ethics_guard
        self.response_templates = self._initialize_response_templates()
        self.cultural_adaptations = self._initialize_cultural_adaptations()

    def generate_empathic_response(
        self,
        emotional_state: EmotionalState,
        user_profile: EmotionalProfile,
        context: Dict[str, Any],
        vad: Optional[Tuple[float, float, float]] = None
    ) -> EmpathicResponse:
        response_type = self._determine_response_type(emotional_state, user_profile)
        intensity = self._calculate_empathy_intensity(emotional_state, user_profile, vad)
        base_content = self._generate_base_response(emotional_state, response_type, context)
        adapted_content = self._apply_cultural_adaptation(base_content, user_profile.cultural_context)
        personalized_content = self._personalize_response(adapted_content, user_profile)

        response = EmpathicResponse(
            content=personalized_content,
            intensity=intensity,
            response_type=response_type,
            cultural_adaptation=user_profile.cultural_context.value,
            confidence=0.9 if vad else 0.85,
            therapeutic_alignment=True,
            privacy_level=user_profile.consent_level,
        )
        is_valid, violations = self.ethics_guard.validate_empathic_response(response, user_profile, emotional_state)
        if not is_valid:
            logging.warning(f"Empathic response validation failed: {violations}")
            response = self._generate_fallback_response(emotional_state, user_profile)
        return response

    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "sadness": {
                "validation": [
                    "I can see that you're going through a difficult time right now.",
                    "It's completely understandable to feel sad about this situation.",
                    "Your feelings are valid and it's okay to experience sadness.",
                ],
                "support": [
                    "I'm here to support you through this challenging period.",
                    "You don't have to face this alone - support is available.",
                    "Many people have felt this way, and there are paths forward.",
                ],
                "guidance": [
                    "Sometimes taking small steps can help us move through sadness.",
                    "Have you considered reaching out to trusted friends or family?",
                    "Professional counseling can be very helpful for processing these feelings.",
                ],
            },
            "anxiety": {
                "validation": [
                    "Anxiety can feel overwhelming, and I acknowledge what you're experiencing.",
                    "It's natural to feel anxious when facing uncertainty or challenges.",
                    "Your anxiety is a signal that something feels important to you.",
                ],
                "support": [
                    "Let's take this one step at a time—you don't need to solve everything at once.",
                    "Breathing exercises and grounding techniques can provide immediate relief.",
                    "You have managed anxiety before, and you can do it again.",
                ],
                "guidance": [
                    "Breaking large concerns into smaller, manageable pieces often helps.",
                    "Consider what aspects of the situation are within your control.",
                    "Professional anxiety management techniques could be beneficial.",
                ],
            },
            "anger": {
                "validation": [
                    "It's understandable to feel angry given what happened.",
                    "Your frustration makes sense—your feelings matter.",
                ],
                "support": [
                    "I'm here with you while you process this.",
                    "We can find ways to express that anger safely.",
                ],
                "guidance": [
                    "Taking a pause before responding can help you stay in control.",
                    "Channeling energy into a brief walk or writing can reduce intensity.",
                ],
            },
            "fear": {
                "validation": [
                    "Feeling afraid in uncertain situations is a natural response.",
                    "Your safety and sense of control are important.",
                ],
                "support": [
                    "We can focus on what helps you feel safer in this moment.",
                    "Grounding in the here-and-now may ease the fear a bit.",
                ],
                "guidance": [
                    "Identifying one small step you can take may restore some control.",
                    "If the fear persists, speaking with a professional could help.",
                ],
            },
            "stress": {
                "validation": [
                    "It sounds like there's a lot on your plate right now.",
                    "Feeling stressed when demands add up is very human.",
                ],
                "support": [
                    "Let's take it one task at a time and be gentle with expectations.",
                    "Brief breaks and hydration can make a noticeable difference today.",
                ],
                "guidance": [
                    "Prioritizing the top one or two items may reduce pressure.",
                    "If possible, delegating or postponing less urgent items can help.",
                ],
            },
            "depression": {
                "validation": [
                    "Feeling low and drained can be extremely hard—I'm sorry you're going through this.",
                    "Your feelings matter, and you're not alone in this experience.",
                ],
                "support": [
                    "Small steps—like getting some light or reaching out—can be meaningful.",
                    "If you can, connecting with someone you trust could provide support.",
                ],
                "guidance": [
                    "A mental health professional can help explore options that fit you.",
                    "If motivation is low, consider very gentle routines to start.",
                ],
            },
        }

    def _initialize_cultural_adaptations(self) -> Dict[CulturalContext, Dict[str, str]]:
        # These guide rephrasing in a non-brittle way
        return {
            CulturalContext.COLLECTIVISTIC: {
                "tail": " If it helps, leaning on community or family support can be valuable.",
            },
            CulturalContext.HIGH_CONTEXT: {
                "tone": "indirect",
            },
        }

    def _determine_response_type(self, emotional_state: EmotionalState, user_profile: EmotionalProfile) -> str:
        if emotional_state in [EmotionalState.SADNESS, EmotionalState.DEPRESSION]:
            return "validation" if user_profile.emotional_sensitivity > 0.7 else "support"
        elif emotional_state in [EmotionalState.ANXIETY, EmotionalState.STRESS, EmotionalState.FEAR]:
            return "guidance" if user_profile.preferred_support_style == "practical" else "support"
        elif emotional_state in [EmotionalState.ANGER]:
            return "validation"
        else:
            return "validation"

    def _calculate_empathy_intensity(
        self,
        emotional_state: EmotionalState,
        user_profile: EmotionalProfile,
        vad: Optional[Tuple[float, float, float]]
    ) -> EmpathyIntensity:
        base_intensity = 0.5
        if emotional_state in [EmotionalState.DEPRESSION, EmotionalState.ANXIETY, EmotionalState.FEAR]:
            base_intensity += 0.25
        # Adjust for user sensitivity
        base_intensity += (user_profile.emotional_sensitivity - 0.5) * 0.4
        # If arousal high, consider stronger presence
        if vad is not None and vad[1] > 0.75:
            base_intensity += 0.1
        base_intensity = float(np.clip(base_intensity, 0.0, 1.0))
        if base_intensity < 0.2:
            return EmpathyIntensity.MINIMAL
        elif base_intensity < 0.4:
            return EmpathyIntensity.GENTLE
        elif base_intensity < 0.6:
            return EmpathyIntensity.MODERATE
        elif base_intensity < 0.8:
            return EmpathyIntensity.STRONG
        else:
            return EmpathyIntensity.INTENSIVE

    def _generate_base_response(self, emotional_state: EmotionalState, response_type: str, context: Dict[str, Any]) -> str:
        state_name = emotional_state.value
        if state_name in self.response_templates:
            templates = self.response_templates[state_name].get(
                response_type,
                self.response_templates[state_name].get("validation", [])
            )
            if templates:
                # deterministic pick for reproducibility (could randomize)
                return templates[0]
        return "I understand you're experiencing difficult emotions right now."

    def _apply_cultural_adaptation(self, content: str, cultural_context: CulturalContext) -> str:
        if cultural_context in self.cultural_adaptations:
            cfg = self.cultural_adaptations[cultural_context]
            if cultural_context == CulturalContext.COLLECTIVISTIC:
                tail = cfg.get("tail", "")
                if tail and tail not in content:
                    content = content.rstrip(".") + "." + tail
            if cultural_context == CulturalContext.HIGH_CONTEXT:
                # soften directness slightly
                if content.startswith("You "):
                    content = content.replace("You ", "You might ", 1)
        return content

    def _personalize_response(self, content: str, user_profile: EmotionalProfile) -> str:
        if user_profile.preferred_support_style == "gentle":
            content = content.replace("you should", "you might consider").replace("you need to", "it could help to")
        return content

    def _generate_fallback_response(self, emotional_state: EmotionalState, user_profile: EmotionalProfile) -> EmpathicResponse:
        return EmpathicResponse(
            content=(
                "I want to support you. If you're experiencing distress, please consider speaking with a mental health professional. "
                "If you're in immediate danger, contact local emergency services."
            ),
            intensity=EmpathyIntensity.GENTLE,
            response_type="support",
            cultural_adaptation="neutral",
            confidence=0.95,
            therapeutic_alignment=True,
            privacy_level=user_profile.consent_level,
        )


class EmpathyEffectivenessTracker:
    """Tracks and measures empathy intervention effectiveness"""

    def __init__(self):
        self.metrics_history: Dict[str, List[EmpathyMetrics]] = {}
        self.intervention_outcomes: Dict[str, Any] = {}

    def track_empathy_interaction(
        self,
        user_id: str,
        response: EmpathicResponse,
        user_feedback: Optional[Dict[str, Any]] = None,
        emotional_change: Optional[Dict[str, float]] = None
    ):
        if user_id not in self.metrics_history:
            self.metrics_history[user_id] = []
        metrics = EmpathyMetrics()
        if user_feedback:
            metrics.user_satisfaction = float(user_feedback.get("satisfaction", 0.0))
            metrics.engagement_level = float(user_feedback.get("engagement", 0.0))
            metrics.cultural_appropriateness = float(user_feedback.get("cultural_fit", 1.0))
        if emotional_change:
            metrics.emotional_improvement = float(emotional_change.get("improvement", 0.0))
        metrics.therapeutic_progress = self._assess_therapeutic_progress(user_id, response, user_feedback)
        self.metrics_history[user_id].append(metrics)

    def get_empathy_effectiveness_report(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.metrics_history:
            return {"error": "No data available for user"}
        metrics_list = self.metrics_history[user_id]
        return {
            "user_id": user_id,
            "total_interactions": len(metrics_list),
            "average_satisfaction": float(np.mean([m.user_satisfaction for m in metrics_list])),
            "average_improvement": float(np.mean([m.emotional_improvement for m in metrics_list])),
            "average_engagement": float(np.mean([m.engagement_level for m in metrics_list])),
            "therapeutic_progress": metrics_list[-1].therapeutic_progress if metrics_list else 0.0,
            "cultural_appropriateness": float(np.mean([m.cultural_appropriateness for m in metrics_list])),
            "safety_compliance": bool(all(m.safety_compliance for m in metrics_list)),
            "last_updated": metrics_list[-1].measurement_timestamp if metrics_list else None,
        }

    def _assess_therapeutic_progress(self, user_id: str, response: EmpathicResponse, user_feedback: Optional[Dict[str, Any]]) -> float:
        score = 0.5
        if user_feedback:
            if user_feedback.get("felt_heard", False):
                score += 0.2
            if user_feedback.get("gained_insight", False):
                score += 0.3
        return float(min(score, 1.0))


class EnhancedAffectiveStateClassifier:
    """Enhanced Affective State Classifier with empathy capabilities"""

    def __init__(self, base_classifier, empathy_config: Dict[str, Any]):
        self.base_classifier = base_classifier
        self.empathy_config = empathy_config or {}
        self.privacy_guard = EmpathyPrivacyGuard()
        self.ethics_guard = EmpathyEthicsGuard()
        self.response_generator = EmpathicResponseGenerator(self.ethics_guard)
        self.effectiveness_tracker = EmpathyEffectivenessTracker()
        self.user_profiles: Dict[str, EmotionalProfile] = {}
        # Optional advanced emotion adapter
        self.hf_adapter: Optional[_HFEmotionAdapter] = None
        if self.empathy_config.get("use_hf_emotion_model", True) and _TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.empathy_config.get("hf_emotion_model", "joeddav/distilbert-base-uncased-go-emotions-student")
                self.hf_adapter = _HFEmotionAdapter(model_name=model_name)
                logging.info(f"Hugging Face emotion model initialized: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to initialize HF emotion model, will fallback: {e}")

    def classify_with_empathy(self, multi_modal_inputs: Dict[str, np.ndarray], user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify emotional state with empathy-aware processing"""
        text_input = context.get("text_input")

        # Default VAD from base classifier
        base_prediction = self.base_classifier.predict(multi_modal_inputs)
        valence, arousal = float(base_prediction[0]), float(base_prediction[1])
        dominance = float(self._get_user_profile(user_id).baseline_dominance)
        top_emotions: Dict[str, float] = {}

        # If we have text and HF adapter, refine VAD
        if text_input and self.hf_adapter is not None:
            try:
                v2, a2, d2, scores = self.hf_adapter.infer_vad(text_input)
                # Blend: prioritize HF but keep base as weak prior
                alpha = 0.8
                valence = alpha * v2 + (1 - alpha) * valence
                arousal = alpha * a2 + (1 - alpha) * arousal
                dominance = alpha * d2 + (1 - alpha) * dominance
                # Keep top emotions (sorted)
                top_emotions = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5])
            except Exception as e:
                logging.warning(f"HF emotion inference failed, using base classifier only: {e}")

        emotional_state = self._map_to_emotional_state(valence, arousal)
        user_profile = self._get_user_profile(user_id)

        # Crisis detection with risk score
        crisis_detected = False
        crisis_score = 0.0
        if text_input:
            crisis_detected, crisis_score = self.ethics_guard.detect_crisis_state(text_input, {"valence": valence, "arousal": arousal})
            if crisis_detected:
                logging.warning(f"Crisis state detected for user {user_id} (risk={crisis_score:.2f})")
                return self._handle_crisis_state(user_id, emotional_state, context, crisis_score)

        # Generate empathetic response (pass VAD)
        empathic_response = self.response_generator.generate_empathic_response(
            emotional_state, user_profile, context, vad=(valence, arousal, dominance)
        )

        # Encrypt emotional data for privacy (minimize PII)
        emotional_data = {
            "valence": float(valence),
            "arousal": float(arousal),
            "dominance": float(dominance),
            "emotional_state": emotional_state.value,
            "context_keys": list(context.keys()),  # store only keys to reduce sensitive content
            "timestamp": datetime.now().isoformat()
        }
        encrypted_data = self.privacy_guard.encrypt_emotional_data(emotional_data, user_id)

        result: Dict[str, Any] = {
            "emotional_state": emotional_state.value,
            "valence": float(valence),
            "arousal": float(arousal),
            "dominance": float(dominance),
            "empathic_response": {
                "content": empathic_response.content,
                "intensity": empathic_response.intensity.value,
                "type": empathic_response.response_type,
                "confidence": float(empathic_response.confidence),
            },
            "cultural_adaptation": empathic_response.cultural_adaptation,
            "privacy_protected": True,
            "encrypted_data_id": encrypted_data,
            "crisis_detected": False,
            "therapeutic_alignment": empathic_response.therapeutic_alignment
        }

        if context.get("debug"):
            result["diagnostics"] = {
                "hf_model_enabled": bool(self.hf_adapter is not None),
                "top_emotions": top_emotions,
                "crisis_score": crisis_score,
            }
        return result

    def update_user_profile(self, user_id: str, profile_updates: Dict[str, Any], consent_verified: bool = False):
        """Update user's emotional profile with consent validation"""
        if not consent_verified:
            raise ValueError("User consent must be verified before profile updates")
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = EmotionalProfile(user_id=user_id)
        profile = self.user_profiles[user_id]
        updatable_fields = [
            "baseline_valence",
            "baseline_arousal",
            "baseline_dominance",
            "emotional_sensitivity",
            "preferred_support_style",
            "cultural_context",
            "empathy_preferences",
            "therapeutic_goals",
            "consent_level",
        ]
        for field_name, value in profile_updates.items():
            if field_name in updatable_fields:
                setattr(profile, field_name, value)
        profile.last_updated = datetime.now()
        logging.info(f"Updated profile for user {user_id}")

    def get_empathy_effectiveness_report(self, user_id: str) -> Dict[str, Any]:
        """Get effectiveness report for empathy interventions"""
        return self.effectiveness_tracker.get_empathy_effectiveness_report(user_id)

    def _map_to_emotional_state(self, valence: float, arousal: float) -> EmotionalState:
        """Map V-A to specific emotional states with nuanced thresholds."""
        if valence > 0.4 and arousal > 0.65:
            return EmotionalState.EXCITEMENT
        if valence > 0.4 and arousal <= 0.65:
            return EmotionalState.CONTENTMENT
        if valence < -0.5 and arousal < 0.35:
            return EmotionalState.DEPRESSION
        if valence < -0.5 and 0.35 <= arousal < 0.7:
            return EmotionalState.SADNESS
        if valence < -0.5 and arousal >= 0.7:
            return EmotionalState.ANXIETY
        if valence < -0.6 and arousal > 0.7:
            return EmotionalState.FEAR
        if valence < -0.6 and arousal > 0.6:
            return EmotionalState.ANGER
        if valence < -0.2 and arousal > 0.75:
            return EmotionalState.STRESS
        return EmotionalState.CONTENTMENT

    def _get_user_profile(self, user_id: str) -> EmotionalProfile:
        """Get or create user emotional profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = EmotionalProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def _handle_crisis_state(self, user_id: str, emotional_state: EmotionalState, context: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
        """Handle detected crisis state with immediate referral"""
        crisis_response = {
            "emotional_state": emotional_state.value,
            "crisis_detected": True,
            "immediate_action_required": True,
            "empathic_response": {
                "content": (
                    "I'm concerned about your well-being. Please reach out to a mental health professional immediately. "
                    "If you're in crisis, contact your local emergency services or a crisis hotline right away."
                    " In the U.S.: call or text 988 (Suicide & Crisis Lifeline). If you are outside the U.S., please use local resources."
                ),
                "type": "crisis_intervention",
                "intensity": "intensive",
                "confidence": 1.0
            },
            "referral_resources": [
                "U.S. Suicide & Crisis Lifeline: 988 (call/text)",
                "Crisis Text Line (U.S./Canada/UK/Ireland): Text HOME to 741741",
                "Local Emergency Services: 911 (U.S.) or your country's equivalent"
            ],
            "professional_notification_sent": True,
            "privacy_protected": True,
            "crisis_risk_score": float(risk_score)
        }
        logging.critical(f"Crisis intervention triggered for user {user_id} (risk={risk_score:.2f})")
        return crisis_response
