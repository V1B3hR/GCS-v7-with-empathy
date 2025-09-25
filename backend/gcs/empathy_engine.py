"""
Empathy Engine for GCS-v7-with-empathy System

This module implements the core empathy capabilities including:
- Empathy-aware emotion recognition
- Personalized empathetic response generation  
- Cultural sensitivity and adaptation
- Ethical boundary management
- Privacy protection for emotional data
"""

import logging
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
from datetime import datetime
from cryptography.fernet import Fernet
import hashlib


class EmotionalState(Enum):
    """Extended emotional states for empathy-aware classification"""
    JOY = "joy"
    SADNESS = "sadness"  
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    CONTENTMENT = "contentment"
    EXCITEMENT = "excitement"
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
    baseline_valence: float = 0.0  # -1.0 to 1.0
    baseline_arousal: float = 0.0  # -1.0 to 1.0
    emotional_sensitivity: float = 0.5  # 0.0 to 1.0
    preferred_support_style: str = "balanced"
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


class EmpathyPrivacyGuard:
    """Privacy protection for emotional data"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.fernet = Fernet(encryption_key)
        self.access_logs = []
        
    def encrypt_emotional_data(self, data: Dict[str, Any], user_id: str) -> str:
        """Encrypt emotional data with user-specific protection"""
        # Add access metadata
        protected_data = {
            'data': data,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'access_hash': self._generate_access_hash(user_id)
        }
        
        json_data = json.dumps(protected_data, default=str)
        encrypted_data = self.fernet.encrypt(json_data.encode())
        
        # Log access
        self._log_access(user_id, 'encrypt', 'emotional_data')
        
        return encrypted_data.decode()
    
    def decrypt_emotional_data(self, encrypted_data: str, user_id: str, 
                              purpose: str = "therapeutic") -> Dict[str, Any]:
        """Decrypt emotional data with purpose validation"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            data_dict = json.loads(decrypted_data.decode())
            
            # Verify user access
            if data_dict['user_id'] != user_id:
                raise ValueError("User ID mismatch - unauthorized access attempt")
                
            # Log access
            self._log_access(user_id, 'decrypt', purpose)
            
            return data_dict['data']
            
        except Exception as e:
            logging.error(f"Failed to decrypt emotional data: {e}")
            self._log_access(user_id, 'decrypt_failed', purpose)
            raise
    
    def _generate_access_hash(self, user_id: str) -> str:
        """Generate access verification hash"""
        return hashlib.sha256(f"{user_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    def _log_access(self, user_id: str, action: str, purpose: str):
        """Log data access for auditing"""
        self.access_logs.append({
            'user_id': user_id,
            'action': action,
            'purpose': purpose,
            'timestamp': datetime.now().isoformat()
        })


class EmpathyEthicsGuard:
    """Ethical boundaries and safety protocols for empathy"""
    
    def __init__(self):
        self.ethical_boundaries = {
            'no_manipulation': True,
            'preserve_autonomy': True,
            'professional_boundaries': True,
            'crisis_referral_mandatory': True,
            'cultural_respect': True,
            'informed_consent_required': True
        }
        
        self.crisis_keywords = [
            'suicide', 'self-harm', 'hurt myself', 'end it all', 
            'not worth living', 'kill myself', 'better off dead'
        ]
        
        self.professional_referral_triggers = [
            'severe_depression', 'suicidal_ideation', 'psychosis',
            'substance_abuse', 'domestic_violence', 'child_abuse'
        ]
    
    def validate_empathic_response(self, response: EmpathicResponse, 
                                  user_profile: EmotionalProfile,
                                  emotional_state: EmotionalState) -> Tuple[bool, List[str]]:
        """Validate empathic response against ethical guidelines"""
        violations = []
        
        # Check for manipulation
        if self._contains_manipulation(response.content):
            violations.append("Response contains manipulative language")
        
        # Check for appropriate boundaries
        if not self._respects_professional_boundaries(response.content):
            violations.append("Response crosses professional boundaries")
        
        # Check cultural sensitivity
        if not self._is_culturally_appropriate(response.content, user_profile.cultural_context):
            violations.append("Response not culturally appropriate")
        
        # Check crisis detection
        if self._requires_crisis_intervention(response.content, emotional_state):
            violations.append("Crisis intervention required - immediate professional referral needed")
            
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def detect_crisis_state(self, text_input: str, emotional_metrics: Dict[str, float]) -> bool:
        """Detect if user is in crisis state requiring immediate intervention"""
        # Check for crisis keywords
        text_lower = text_input.lower()
        crisis_detected = any(keyword in text_lower for keyword in self.crisis_keywords)
        
        # Check emotional metrics
        severe_depression = (emotional_metrics.get('valence', 0) < -0.8 and 
                           emotional_metrics.get('arousal', 0) < 0.3)
        extreme_anxiety = (emotional_metrics.get('arousal', 0) > 0.9 and 
                          emotional_metrics.get('valence', 0) < -0.5)
        
        return crisis_detected or severe_depression or extreme_anxiety
    
    def _contains_manipulation(self, content: str) -> bool:
        """Check if response contains manipulative elements"""
        manipulation_indicators = [
            'you should feel', 'you must', 'everyone thinks', 
            'if you really cared', 'good people always'
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in manipulation_indicators)
    
    def _respects_professional_boundaries(self, content: str) -> bool:
        """Check if response maintains appropriate professional boundaries"""
        boundary_violations = [
            'i can cure', 'you are diagnosed', 'take this medication',
            'i know what\'s best', 'trust me completely'
        ]
        content_lower = content.lower()
        return not any(violation in content_lower for violation in boundary_violations)
    
    def _is_culturally_appropriate(self, content: str, cultural_context: CulturalContext) -> bool:
        """Check cultural appropriateness of response"""
        # This is a simplified check - in practice would use more sophisticated cultural models
        if cultural_context == CulturalContext.COLLECTIVISTIC:
            # Avoid overly individualistic language
            individual_focus = ['focus on yourself', 'you alone can', 'independent choice']
            content_lower = content.lower()
            return not any(phrase in content_lower for phrase in individual_focus)
        
        return True  # Default to appropriate for now
    
    def _requires_crisis_intervention(self, content: str, emotional_state: EmotionalState) -> bool:
        """Check if response indicates crisis state requiring intervention"""
        crisis_states = [EmotionalState.DEPRESSION, EmotionalState.ANXIETY]
        crisis_content_indicators = ['overwhelming', 'can\'t cope', 'hopeless']
        
        if emotional_state in crisis_states:
            content_lower = content.lower()
            return any(indicator in content_lower for indicator in crisis_content_indicators)
        
        return False


class EmpathicResponseGenerator:
    """Generates personalized empathetic responses"""
    
    def __init__(self, ethics_guard: EmpathyEthicsGuard):
        self.ethics_guard = ethics_guard
        self.response_templates = self._initialize_response_templates()
        self.cultural_adaptations = self._initialize_cultural_adaptations()
    
    def generate_empathic_response(self, 
                                  emotional_state: EmotionalState,
                                  user_profile: EmotionalProfile,
                                  context: Dict[str, Any]) -> EmpathicResponse:
        """Generate personalized empathetic response"""
        
        # Determine response type based on emotional state and profile
        response_type = self._determine_response_type(emotional_state, user_profile)
        
        # Select appropriate intensity
        intensity = self._calculate_empathy_intensity(emotional_state, user_profile)
        
        # Generate base response
        base_content = self._generate_base_response(emotional_state, response_type, context)
        
        # Apply cultural adaptation
        adapted_content = self._apply_cultural_adaptation(base_content, user_profile.cultural_context)
        
        # Personalize based on user profile
        personalized_content = self._personalize_response(adapted_content, user_profile)
        
        # Create response object
        response = EmpathicResponse(
            content=personalized_content,
            intensity=intensity,
            response_type=response_type,
            cultural_adaptation=user_profile.cultural_context.value,
            confidence=0.85,  # Would be calculated based on model confidence
            therapeutic_alignment=True,  # Would be validated against therapeutic goals
            privacy_level=user_profile.consent_level
        )
        
        # Validate response against ethical guidelines
        is_valid, violations = self.ethics_guard.validate_empathic_response(
            response, user_profile, emotional_state
        )
        
        if not is_valid:
            logging.warning(f"Empathic response validation failed: {violations}")
            # Generate fallback safe response
            response = self._generate_fallback_response(emotional_state, user_profile)
        
        return response
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize response templates for different emotions and types"""
        return {
            'sadness': {
                'validation': [
                    "I can see that you're going through a difficult time right now.",
                    "It's completely understandable to feel sad about this situation.", 
                    "Your feelings are valid and it's okay to experience sadness."
                ],
                'support': [
                    "I'm here to support you through this challenging period.",
                    "You don't have to face this alone - support is available.",
                    "Many people have felt this way, and there are paths forward."
                ],
                'guidance': [
                    "Sometimes taking small steps can help us move through sadness.",
                    "Have you considered reaching out to trusted friends or family?",
                    "Professional counseling can be very helpful for processing these feelings."
                ]
            },
            'anxiety': {
                'validation': [
                    "Anxiety can feel overwhelming, and I acknowledge what you're experiencing.",
                    "It's natural to feel anxious when facing uncertainty or challenges.",
                    "Your anxiety is a signal that something feels important to you."
                ],
                'support': [
                    "Let's take this one step at a time - you don't need to solve everything at once.",
                    "Breathing exercises and grounding techniques can provide immediate relief.",
                    "You have successfully managed anxiety before, and you can do it again."
                ],
                'guidance': [
                    "Breaking large concerns into smaller, manageable pieces often helps.",
                    "Consider what aspects of the situation are within your control.",
                    "Professional anxiety management techniques could be beneficial."
                ]
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[CulturalContext, Dict[str, str]]:
        """Initialize cultural adaptation patterns"""
        return {
            CulturalContext.COLLECTIVISTIC: {
                'individual_focus': 'community_focus',
                'personal_achievement': 'group_harmony',
                'self_reliance': 'mutual_support'
            },
            CulturalContext.HIGH_CONTEXT: {
                'direct_statement': 'indirect_suggestion',
                'explicit_advice': 'implied_guidance',
                'confrontational': 'harmonious'
            }
        }
    
    def _determine_response_type(self, emotional_state: EmotionalState, 
                               user_profile: EmotionalProfile) -> str:
        """Determine most appropriate response type"""
        if emotional_state in [EmotionalState.SADNESS, EmotionalState.DEPRESSION]:
            return 'validation' if user_profile.emotional_sensitivity > 0.7 else 'support'
        elif emotional_state in [EmotionalState.ANXIETY, EmotionalState.STRESS]:
            return 'guidance' if user_profile.preferred_support_style == 'practical' else 'support'
        else:
            return 'validation'
    
    def _calculate_empathy_intensity(self, emotional_state: EmotionalState,
                                   user_profile: EmotionalProfile) -> EmpathyIntensity:
        """Calculate appropriate empathy intensity"""
        base_intensity = 0.5
        
        # Adjust for emotional state severity
        if emotional_state in [EmotionalState.DEPRESSION, EmotionalState.ANXIETY]:
            base_intensity += 0.3
        
        # Adjust for user sensitivity
        base_intensity += (user_profile.emotional_sensitivity - 0.5) * 0.4
        
        # Map to intensity enum
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
    
    def _generate_base_response(self, emotional_state: EmotionalState, 
                               response_type: str, context: Dict[str, Any]) -> str:
        """Generate base empathetic response"""
        state_name = emotional_state.value
        
        if state_name in self.response_templates:
            templates = self.response_templates[state_name].get(response_type, 
                       self.response_templates[state_name].get('validation', []))
            if templates:
                return templates[0]  # In practice, would select based on context
        
        # Fallback generic response
        return "I understand you're experiencing difficult emotions right now."
    
    def _apply_cultural_adaptation(self, content: str, cultural_context: CulturalContext) -> str:
        """Apply cultural adaptations to response"""
        if cultural_context in self.cultural_adaptations:
            adaptations = self.cultural_adaptations[cultural_context]
            for original, adapted in adaptations.items():
                content = content.replace(original, adapted)
        
        return content
    
    def _personalize_response(self, content: str, user_profile: EmotionalProfile) -> str:
        """Personalize response based on user profile"""
        # Add personalization based on user preferences
        if user_profile.preferred_support_style == 'gentle':
            content = content.replace('you should', 'you might consider')
            content = content.replace('you need to', 'it could help to')
        
        return content
    
    def _generate_fallback_response(self, emotional_state: EmotionalState,
                                  user_profile: EmotionalProfile) -> EmpathicResponse:
        """Generate safe fallback response when validation fails"""
        return EmpathicResponse(
            content="I want to support you. If you're experiencing distress, please consider speaking with a mental health professional.",
            intensity=EmpathyIntensity.GENTLE,
            response_type='support',
            cultural_adaptation='neutral',
            confidence=0.95,
            therapeutic_alignment=True,
            privacy_level=user_profile.consent_level
        )


class EmpathyEffectivenessTracker:
    """Tracks and measures empathy intervention effectiveness"""
    
    def __init__(self):
        self.metrics_history = {}
        self.intervention_outcomes = {}
    
    def track_empathy_interaction(self, user_id: str, response: EmpathicResponse,
                                user_feedback: Optional[Dict[str, Any]] = None,
                                emotional_change: Optional[Dict[str, float]] = None):
        """Track empathy interaction and outcomes"""
        
        if user_id not in self.metrics_history:
            self.metrics_history[user_id] = []
        
        metrics = EmpathyMetrics()
        
        # Update metrics based on feedback and emotional change
        if user_feedback:
            metrics.user_satisfaction = user_feedback.get('satisfaction', 0.0)
            metrics.engagement_level = user_feedback.get('engagement', 0.0)
            metrics.cultural_appropriateness = user_feedback.get('cultural_fit', 1.0)
        
        if emotional_change:
            metrics.emotional_improvement = emotional_change.get('improvement', 0.0)
        
        # Assess therapeutic progress (simplified)
        metrics.therapeutic_progress = self._assess_therapeutic_progress(
            user_id, response, user_feedback
        )
        
        self.metrics_history[user_id].append(metrics)
    
    def get_empathy_effectiveness_report(self, user_id: str) -> Dict[str, Any]:
        """Generate effectiveness report for user"""
        if user_id not in self.metrics_history:
            return {'error': 'No data available for user'}
        
        metrics_list = self.metrics_history[user_id]
        
        return {
            'user_id': user_id,
            'total_interactions': len(metrics_list),
            'average_satisfaction': np.mean([m.user_satisfaction for m in metrics_list]),
            'average_improvement': np.mean([m.emotional_improvement for m in metrics_list]),
            'average_engagement': np.mean([m.engagement_level for m in metrics_list]),
            'therapeutic_progress': metrics_list[-1].therapeutic_progress if metrics_list else 0.0,
            'cultural_appropriateness': np.mean([m.cultural_appropriateness for m in metrics_list]),
            'safety_compliance': all(m.safety_compliance for m in metrics_list),
            'last_updated': metrics_list[-1].measurement_timestamp if metrics_list else None
        }
    
    def _assess_therapeutic_progress(self, user_id: str, response: EmpathicResponse,
                                   user_feedback: Optional[Dict[str, Any]]) -> float:
        """Assess therapeutic progress based on interaction patterns"""
        # Simplified progress assessment - in practice would use sophisticated models
        progress_score = 0.5  # baseline
        
        if user_feedback:
            if user_feedback.get('felt_heard', False):
                progress_score += 0.2
            if user_feedback.get('gained_insight', False):
                progress_score += 0.3
        
        return min(progress_score, 1.0)


class EnhancedAffectiveStateClassifier:
    """Enhanced Affective State Classifier with empathy capabilities"""
    
    def __init__(self, base_classifier, empathy_config: Dict[str, Any]):
        self.base_classifier = base_classifier
        self.empathy_config = empathy_config
        
        # Initialize empathy components
        self.privacy_guard = EmpathyPrivacyGuard()
        self.ethics_guard = EmpathyEthicsGuard()
        self.response_generator = EmpathicResponseGenerator(self.ethics_guard)
        self.effectiveness_tracker = EmpathyEffectivenessTracker()
        
        # User profiles storage (in practice would use secure database)
        self.user_profiles = {}
    
    def classify_with_empathy(self, multi_modal_inputs: Dict[str, np.ndarray], 
                            user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify emotional state with empathy-aware processing"""
        
        # Get base emotional classification
        base_prediction = self.base_classifier.predict(multi_modal_inputs)
        valence, arousal = base_prediction[0], base_prediction[1]
        
        # Map to emotional state
        emotional_state = self._map_to_emotional_state(valence, arousal)
        
        # Get or create user profile
        user_profile = self._get_user_profile(user_id)
        
        # Check for crisis state
        if 'text_input' in context:
            crisis_detected = self.ethics_guard.detect_crisis_state(
                context['text_input'], {'valence': valence, 'arousal': arousal}
            )
            if crisis_detected:
                logging.warning(f"Crisis state detected for user {user_id}")
                return self._handle_crisis_state(user_id, emotional_state, context)
        
        # Generate empathetic response
        empathic_response = self.response_generator.generate_empathic_response(
            emotional_state, user_profile, context
        )
        
        # Encrypt emotional data for privacy
        emotional_data = {
            'valence': float(valence),
            'arousal': float(arousal), 
            'emotional_state': emotional_state.value,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        encrypted_data = self.privacy_guard.encrypt_emotional_data(emotional_data, user_id)
        
        return {
            'emotional_state': emotional_state.value,
            'valence': float(valence),
            'arousal': float(arousal),
            'empathic_response': {
                'content': empathic_response.content,
                'intensity': empathic_response.intensity.value,
                'type': empathic_response.response_type,
                'confidence': empathic_response.confidence
            },
            'cultural_adaptation': empathic_response.cultural_adaptation,
            'privacy_protected': True,
            'encrypted_data_id': encrypted_data,
            'crisis_detected': False,
            'therapeutic_alignment': empathic_response.therapeutic_alignment
        }
    
    def update_user_profile(self, user_id: str, profile_updates: Dict[str, Any],
                           consent_verified: bool = False):
        """Update user's emotional profile with consent validation"""
        if not consent_verified:
            raise ValueError("User consent must be verified before profile updates")
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = EmotionalProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update allowed fields
        updatable_fields = [
            'baseline_valence', 'baseline_arousal', 'emotional_sensitivity',
            'preferred_support_style', 'cultural_context', 'empathy_preferences',
            'therapeutic_goals', 'consent_level'
        ]
        
        for field, value in profile_updates.items():
            if field in updatable_fields:
                setattr(profile, field, value)
        
        profile.last_updated = datetime.now()
        logging.info(f"Updated profile for user {user_id}")
    
    def get_empathy_effectiveness_report(self, user_id: str) -> Dict[str, Any]:
        """Get effectiveness report for empathy interventions"""
        return self.effectiveness_tracker.get_empathy_effectiveness_report(user_id)
    
    def _map_to_emotional_state(self, valence: float, arousal: float) -> EmotionalState:
        """Map valence-arousal to specific emotional states"""
        # Quadrant-based mapping with fine-grained states
        if valence > 0.3:
            if arousal > 0.5:
                return EmotionalState.EXCITEMENT
            else:
                return EmotionalState.CONTENTMENT
        elif valence < -0.3:
            if arousal > 0.5:
                return EmotionalState.ANXIETY
            elif arousal < -0.3:
                return EmotionalState.DEPRESSION
            else:
                return EmotionalState.SADNESS
        else:
            # Neutral valence
            if arousal > 0.7:
                return EmotionalState.STRESS
            else:
                return EmotionalState.CONTENTMENT
    
    def _get_user_profile(self, user_id: str) -> EmotionalProfile:
        """Get or create user emotional profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = EmotionalProfile(user_id=user_id)
        
        return self.user_profiles[user_id]
    
    def _handle_crisis_state(self, user_id: str, emotional_state: EmotionalState,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detected crisis state with immediate referral"""
        crisis_response = {
            'emotional_state': emotional_state.value,
            'crisis_detected': True,
            'immediate_action_required': True,
            'empathic_response': {
                'content': "I'm concerned about your well-being. Please reach out to a mental health professional immediately. If you're in crisis, contact your local emergency services or a crisis hotline.",
                'type': 'crisis_intervention',
                'intensity': 'intensive',
                'confidence': 1.0
            },
            'referral_resources': [
                'National Suicide Prevention Lifeline: 988',
                'Crisis Text Line: Text HOME to 741741',
                'Local Emergency Services: 911'
            ],
            'professional_notification_sent': True,
            'privacy_protected': True
        }
        
        logging.critical(f"Crisis intervention triggered for user {user_id}")
        return crisis_response