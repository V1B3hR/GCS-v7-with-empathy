import os
import time
import logging
from typing import Dict, Tuple, Optional, List
from collections import deque
import tempfile
import yaml

import numpy as np
import tensorflow as tf

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

from .consent_gate import ConsentDecisionError, ConsentExecutionError, InterventionConsentGate
from .inference import GCSInference
from .neuromodulation_controller import NeuromodulationController
from .online_learning_module import OnlineLearningModule
from .feedback_detector import AdaptiveFeedbackDetector

# --- Helper Functions ---
def softmax_emotion_to_valence_arousal(prob: np.ndarray, config: Dict) -> Tuple[float, float]:
    # ... (bez zmian jak w oryginale) ...
    if not isinstance(prob, np.ndarray):
        raise TypeError(f"prob must be numpy array, got {type(prob)}")
    if not isinstance(config, dict):
        raise TypeError(f"config must be dictionary, got {type(config)}")
    if prob.size == 0:
        raise ValueError("prob array cannot be empty")
    if prob.ndim == 2:
        if prob.shape[0] == 0:
            raise ValueError("prob array cannot have zero rows")
        prob = prob[0]
    elif prob.ndim != 1:
        raise ValueError(f"prob must be 1D or 2D array, got {prob.ndim}D")
    affective_config = config.get("affective_model", {})
    if not isinstance(affective_config, dict):
        raise ValueError("affective_model section in config must be a dictionary")
    mapping_config = affective_config.get("emotion_mapping", {})
    if not isinstance(mapping_config, dict):
        raise ValueError("emotion_mapping section in config must be a dictionary")
    keys = mapping_config.get("class_order", [])
    anchors = mapping_config.get("anchors", {})
    if not keys:
        raise ValueError("class_order in emotion_mapping cannot be empty")
    if not isinstance(keys, list):
        raise ValueError("class_order must be a list")
    if not isinstance(anchors, dict):
        raise ValueError("anchors must be a dictionary")
    if len(prob) != len(keys):
        raise ValueError(f"Model output size ({len(prob)}) does not match emotion classes in config ({len(keys)})")
    for key in keys:
        if key not in anchors:
            logging.warning(f"Missing anchor data for emotion class '{key}', using default values")
            continue
        anchor = anchors[key]
        if not isinstance(anchor, dict):
            raise ValueError(f"Anchor for '{key}' must be a dictionary")
        if "valence" not in anchor or "arousal" not in anchor:
            logging.warning(f"Missing valence/arousal in anchor for '{key}', using default values")
    try:
        val = sum(prob[i] * anchors.get(k, {}).get("valence", 0.0) for i, k in enumerate(keys))
        aro = sum(prob[i] * anchors.get(k, {}).get("arousal", 0.0) for i, k in enumerate(keys))
    except Exception as e:
        raise ValueError(f"Failed to calculate valence/arousal from emotion probabilities: {e}")
    val_scaled = (val + 1.0) * 5.0
    if not (0 <= val_scaled <= 10):
        logging.warning(f"Valence {val_scaled:.3f} is outside expected range [0, 10]")
    if not (-10 <= aro <= 10):
        logging.warning(f"Arousal {aro:.3f} is outside typical range [-10, 10]")
    return float(val_scaled), float(aro)

# --- The Master Agent ---
class ClosedLoopAgent:
    """
    The master agent that orchestrates the full SENSE -> DECIDE -> ACT -> LEARN loop.
    Now includes SESSION MEMORY: tracks last N (valence, arousal) state tuples for greater context-awareness.
    """

    def __init__(self, config: Dict, session_memory_len: int = 100):
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(config)}")
        required_keys = ["output_model_dir", "graph_scaffold_path"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        self.config = config
        self.is_running = False

        try:
            foundational_model_path = os.path.join(config["output_model_dir"], "gcs_fold_1.h5")
            if not os.path.exists(foundational_model_path):
                logging.warning(f"Foundational model not found at {foundational_model_path}")
            affective_model_path = config.get("affective_model", {}).get("output_model_path")
            if not affective_model_path:
                raise ValueError("affective_model.output_model_path is required in config")
            if not os.path.exists(affective_model_path):
                logging.warning(f"Affective model not found at {affective_model_path}")
        except Exception as e:
            logging.error(f"Failed to validate model paths: {e}")
            raise

        try:
            graph_path = config["graph_scaffold_path"]
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph scaffold file not found: {graph_path}")

            temp_config = {
                "model_path": foundational_model_path,
                "graph_path": graph_path,
                "labels": {0: "LEFT_HAND", 1: "RIGHT_HAND"},
                "batch_size": config.get("batch_size", 16),
                "safe_mode": False,
                "confidence_threshold": 0.4,
                "monitoring": True,
                "output_layers": None,
                "attention_extractor": None
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                yaml.dump(temp_config, temp_file)
                temp_config_path = temp_file.name
            try:
                self.inference_engine = GCSInference(temp_config_path)
            finally:
                try:
                    os.unlink(temp_config_path)
                except OSError as e:
                    logging.warning(f"Failed to clean up temporary config file {temp_config_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize GCS inference engine: {e}")
            raise

        try:
            self.affective_model = tf.keras.models.load_model(affective_model_path, safe_mode=False)
            logging.info(f"Affective model loaded successfully from {affective_model_path}")
        except Exception as e:
            logging.error(f"Failed to load affective model from {affective_model_path}: {e}")
            raise

        try:
            self.mod_controller = NeuromodulationController(config)
        except Exception as e:
            logging.error(f"Failed to initialize NeuromodulationController: {e}")
            raise

        try:
            self.consent_gate = self._build_consent_gate(config.get("consent"))
        except Exception as e:
            logging.error(f"Failed to initialize InterventionConsentGate: {e}")
            raise

        try:
            self.feedback_detector = AdaptiveFeedbackDetector(config)
        except Exception as e:
            logging.error(f"Failed to initialize AdaptiveFeedbackDetector: {e}")
            raise

        try:
            self.olm = OnlineLearningModule(self.inference_engine.model, config)
        except Exception as e:
            logging.error(f"Failed to initialize OnlineLearningModule: {e}")
            raise

        self.last_state = None
        self.last_action_index = None

        # ---- SESSION MEMORY ADDITION ----
        self.session_history = deque(maxlen=session_memory_len)  # stores dicts with valence/arousal/time/intent etc.
        # ---------------------------------

        logging.info("Closed-Loop Agent initialized successfully with all components loaded.")

    def _build_consent_gate(self, consent_config: Optional[Dict]) -> InterventionConsentGate:
        """Create the explicit intervention consent gate with safe defaults."""
        if consent_config is None:
            consent_config = {}
        if not isinstance(consent_config, dict):
            raise ValueError("consent config section must be a dictionary")

        cooldown_default = consent_config.get("cooldown_seconds", 300)
        gate = InterventionConsentGate(
            request_ttl_seconds=consent_config.get("request_ttl_seconds", 60),
            decline_cooldown_seconds=consent_config.get("decline_cooldown_seconds", cooldown_default),
            defer_cooldown_seconds=consent_config.get("defer_cooldown_seconds", cooldown_default),
            revoke_cooldown_seconds=consent_config.get("revoke_cooldown_seconds", cooldown_default),
        )
        return gate

    def _get_neuromodulation_target(self) -> str:
        target = self.config.get("neuromodulation", {}).get("default_target_nerve", getattr(self.mod_controller, "target", "unknown"))
        if not isinstance(target, str) or not target.strip():
            raise ValueError("neuromodulation.default_target_nerve must be a non-empty string")
        return target.strip()

    def _append_session_event(self, event_type: str, **metadata) -> None:
        event = {"time": time.time(), "event": event_type}
        event.update(metadata)
        self.session_history.append(event)

    def _policy_engine(self, cognitive_state: Dict, affective_state: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        if not isinstance(cognitive_state, dict):
            raise TypeError(f"cognitive_state must be dictionary, got {type(cognitive_state)}")
        if not isinstance(affective_state, dict):
            raise TypeError(f"affective_state must be dictionary, got {type(affective_state)}")
        required_cognitive_keys = ["intent", "confidence"]
        required_affective_keys = ["valence", "arousal"]
        missing_cognitive = [k for k in required_cognitive_keys if k not in cognitive_state]
        missing_affective = [k for k in required_affective_keys if k not in affective_state]
        if missing_cognitive:
            raise KeyError(f"Missing required cognitive state keys: {missing_cognitive}")
        if missing_affective:
            raise KeyError(f"Missing required affective state keys: {missing_affective}")

        valence = float(affective_state["valence"])
        arousal = float(affective_state["arousal"])

        # ---- SESSION MEMORY ADDITION: Contextual decision based on trends ----
        # example: if last 10 measurements valence below 3.0, escalate or suggest a soft intervention
        affective_history = [entry for entry in self.session_history if "valence" in entry]
        n_trend = min(10, len(affective_history))
        if n_trend >= 5:
            recent_vals = [entry["valence"] for entry in affective_history[-n_trend:]]
            mean_val = sum(recent_vals) / len(recent_vals)
            if mean_val < 3.0:
                logging.info(f"[SESSION_MEM] Prolonged low valence detected (mean of last {n_trend}: {mean_val:.2f})")
                # Here: escalate, suggest break, or increase support intensity
        # ---------------------------------------------------------------------

        if "neuromodulation" not in self.config:
            logging.warning("neuromodulation section missing from config - interventions disabled")
            return None, None

        pain_confidence = cognitive_state.get("confidence", 0.0)
        try:
            pain_confidence = float(pain_confidence)
        except (ValueError, TypeError):
            logging.error(f"Invalid confidence value: {pain_confidence}")
            return None, None

        arousal_threshold = 7.0
        confidence_threshold = 0.85

        logging.debug(f"[POLICY] Decision factors: intent={cognitive_state.get('intent')}, "
                     f"confidence={pain_confidence:.3f}, arousal={arousal:.3f}, valence={valence:.3f}")

        if (cognitive_state.get("intent") == "PAIN_SIGNATURE" and 
            pain_confidence > confidence_threshold and 
            arousal > arousal_threshold):
            logging.warning(f"[POLICY] Condition met: High-confidence pain signature "
                          f"({pain_confidence:.2f} > {confidence_threshold}) and high arousal "
                          f"({arousal:.2f} > {arousal_threshold})")
            logging.info("[POLICY] Recommending 'ultrasound' intervention")
            modality = "ultrasound"
            try:
                params = self.config["neuromodulation"].get("ultrasound_params", {})
                if not isinstance(params, dict):
                    logging.error("ultrasound_params must be a dictionary, using empty params")
                    params = {}
            except Exception as e:
                logging.error(f"Failed to get ultrasound params: {e}, using empty params")
                params = {}
            return modality, params

        logging.debug("[POLICY] No intervention criteria met")
        return None, None

    def _run_affective_inference(self, live_data: Dict) -> Tuple[float, float]:
        if not isinstance(live_data, dict):
            raise TypeError(f"live_data must be a dictionary, got {type(live_data)}")

        try:
            if not hasattr(self, 'affective_model') or self.affective_model is None:
                raise RuntimeError("Affective model is not loaded")

            input_names = [t.name for t in self.affective_model.inputs]
            if not input_names:
                raise RuntimeError("Affective model has no inputs defined")
            name_to_data = {
                "node_input": live_data.get("source_eeg"), 
                "physio_input": live_data.get("physio"), 
                "voice_input": live_data.get("voice")
            }
            model_inputs = []
            for nm in input_names:
                key = nm.split(":")[0]
                if key not in name_to_data:
                    raise KeyError(f"Affective model expects input '{key}', but it's not mapped in name_to_data")
                if name_to_data[key] is None:
                    raise ValueError(f"Affective model expects '{key}', but no data was provided in live_data")
                input_data = name_to_data[key]
                if not isinstance(input_data, np.ndarray):
                    raise TypeError(f"Input '{key}' must be a numpy array, got {type(input_data)}")
                if input_data.size == 0:
                    raise ValueError(f"Input '{key}' cannot be empty")
                model_inputs.append(input_data)
            try:
                preds = self.affective_model.predict(model_inputs, verbose=0)
            except Exception as e:
                logging.error(f"Affective model prediction failed: {e}")
                raise RuntimeError(f"Affective model prediction failed: {e}")
            if isinstance(preds, np.ndarray):
                if preds.size == 0:
                    raise ValueError("Model prediction returned empty array")
                return softmax_emotion_to_valence_arousal(preds, self.config)
            elif isinstance(preds, (list, tuple)):
                if len(preds) == 0:
                    raise ValueError("Model prediction returned empty list/tuple")
                output_names = [t.name for t in self.affective_model.outputs]
                if len(output_names) != len(preds):
                    raise ValueError(f"Model output names ({len(output_names)}) don't match predictions ({len(preds)})")
                name_to_pred = {output_names[i].split(":")[0]: preds[i] for i in range(len(preds))}
                if "valence_output" in name_to_pred and "arousal_output" in name_to_pred:
                    try:
                        v, a = name_to_pred["valence_output"], name_to_pred["arousal_output"]
                        valence = float(np.array(v).reshape(-1)[0])
                        arousal = float(np.array(a).reshape(-1)[0])
                        return valence, arousal
                    except (IndexError, ValueError) as e:
                        logging.error(f"Failed to extract valence/arousal from direct outputs: {e}")
                        raise ValueError(f"Failed to extract valence/arousal from direct outputs: {e}")
                return softmax_emotion_to_valence_arousal(preds[0], self.config)
            else:
                raise RuntimeError(f"Unexpected affective model prediction type: {type(preds)}")
        except Exception as e:
            logging.error(f"Affective inference failed: {e}")
            raise

    def _extract_cognitive_state(self, prediction_results: List[Dict]) -> Dict[str, Optional[float]]:
        """Normalize inference output into a single cognitive state for policy logic."""
        if not isinstance(prediction_results, list) or not prediction_results:
            raise ValueError("Inference output must be a non-empty list")

        first = prediction_results[0]
        if not isinstance(first, dict):
            raise ValueError("Inference output item must be a dictionary")

        payload = first.get("mi_output") if "mi_output" in first else first
        if not isinstance(payload, dict):
            raise ValueError("Inference output payload must be a dictionary")
        if "error" in payload:
            raise RuntimeError(f"Inference output error: {payload['error']}")

        intent = payload.get("label")
        confidence = payload.get("confidence", 0.0)
        attention = payload.get("attention")

        if intent is None:
            raise ValueError("Inference output missing 'label'")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            raise ValueError(f"Inference output has invalid confidence: {confidence}")

        return {"intent": intent, "confidence": confidence, "attention": attention}

    def run_cycle(self, live_data: Dict):
        if not isinstance(live_data, dict):
            raise TypeError(f"live_data must be a dictionary, got {type(live_data)}")
        try:
            required_data_keys = ["source_eeg", "adj_matrix"]
            missing_keys = [key for key in required_data_keys if key not in live_data]
            if missing_keys:
                raise ValueError(f"Missing required data keys: {missing_keys}")
            try:
                source_chunk = live_data["source_eeg"]
                adj_matrix = live_data["adj_matrix"]
                if not isinstance(source_chunk, np.ndarray):
                    raise TypeError(f"source_eeg must be numpy array, got {type(source_chunk)}")
                if not isinstance(adj_matrix, np.ndarray):
                    raise TypeError(f"adj_matrix must be numpy array, got {type(adj_matrix)}")
                if source_chunk.size == 0:
                    raise ValueError("source_eeg cannot be empty")
                if adj_matrix.size == 0:
                    raise ValueError("adj_matrix cannot be empty")
                prediction_results = self.inference_engine.predict(source_chunk)
                cognitive_state = self._extract_cognitive_state(prediction_results)
                valence, arousal = self._run_affective_inference(live_data)
                affective_state = {"valence": valence, "arousal": arousal}
                logging.info(
                    f"[SENSE] Cognitive: {cognitive_state['intent']} ({cognitive_state['confidence']:.2f}) "
                    f"| Affective: Valence={valence:.2f}, Arousal={arousal:.2f}"
                )
            except Exception as e:
                logging.error(f"SENSE phase failed: {e}")
                raise RuntimeError(f"SENSE phase failed: {e}")

            # ---- SESSION MEMORY LOGGING ----
            mem_point = {
                "time": time.time(),
                "valence": valence,
                "arousal": arousal,
                "intent": cognitive_state.get("intent"),
                "confidence": cognitive_state.get("confidence")
                # Możesz rozszerzyć o inne: feedback, action, user input, itp.
            }
            self.session_history.append(mem_point)
            # ---------------------------------

            try:
                modality, params = self._policy_engine(cognitive_state, affective_state)
            except Exception as e:
                logging.error(f"DECIDE phase failed: {e}")
                raise RuntimeError(f"DECIDE phase failed: {e}")
            if modality:
                try:
                    target = self._get_neuromodulation_target()
                    proposal_result = self.consent_gate.propose(modality, params or {}, target)
                    request = proposal_result.get("request")
                    if proposal_result.get("created") and request:
                        logging.info(
                            f"[ACTION] Created consent request {request['request_id']} for {request['modality']} on {request['target']}."
                        )
                        self._append_session_event(
                            "consent_proposed",
                            request_id=request["request_id"],
                            modality=request["modality"],
                            target=request["target"],
                            status=request["status"],
                        )
                    elif proposal_result.get("reason") == "cooldown":
                        logging.info(
                            f"[ACTION] Consent request suppressed during {proposal_result.get('cooldown_reason')} cooldown "
                            f"until {proposal_result.get('cooldown_until')}."
                        )
                    elif request:
                        logging.debug(
                            f"[ACTION] Consent request {request['request_id']} already active with status {request['status']}."
                        )
                except Exception as e:
                    logging.error(f"ACT phase failed: {e}")
                    logging.warning("Continuing despite ACT phase failure")
            else:
                logging.debug("[ACTION] No intervention recommended")
            # (Feedback loop and online learning placeholder)
        except Exception as e:
            logging.error(f"run_cycle failed: {e}")
            self.is_running = False
            raise

    def cleanup(self):
        try:
            self.is_running = False
            logging.info("Closed-loop agent cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def get_status(self) -> Dict:
        try:
            consent_request = self.consent_gate.get_current_request() if hasattr(self, 'consent_gate') and self.consent_gate else None
            return {
                "is_running": self.is_running,
                "has_inference_engine": hasattr(self, 'inference_engine') and self.inference_engine is not None,
                "has_affective_model": hasattr(self, 'affective_model') and self.affective_model is not None,
                "has_mod_controller": hasattr(self, 'mod_controller') and self.mod_controller is not None,
                "has_feedback_detector": hasattr(self, 'feedback_detector') and self.feedback_detector is not None,
                "has_olm": hasattr(self, 'olm') and self.olm is not None,
                # ---- SESSION MEMORY STATUS ----
                "session_history_len": len(self.session_history),
                "consent_request": (
                    {
                        "request_id": consent_request["request_id"],
                        "status": consent_request["status"],
                        "expires_at": consent_request["expires_at"],
                    }
                    if consent_request else None
                ),
                # ------------------------------
            }
        except Exception as e:
            logging.error(f"Error getting status: {e}")
            return {"error": str(e)}

    # ---- SESSION HISTORY ACCESSOR (OPTIONAL) ----
    def get_session_history(self) -> List[Dict]:
        """Returns the tracked session emotional states history."""
        return list(self.session_history)
    # ---------------------------------------------

    def get_pending_consent_request(self) -> Optional[Dict]:
        """Return the current consent request for presentation to an authenticated UI/operator."""
        request = self.consent_gate.get_current_request()
        if request and request.get("status") in {"pending_consent", "approved"}:
            return request
        return None

    def record_consent_decision(self, request_id: str, decision: str, actor_id: str) -> Dict:
        """
        Record an explicit authenticated human/operator decision for the current request.

        Callers must authenticate and authorize `actor_id` before invoking this method.
        Model output and inferred signals are never valid consent sources.
        """
        try:
            decision_record = self.consent_gate.record_decision(request_id, decision, actor_id)
        except ConsentDecisionError:
            raise
        self._append_session_event(
            "consent_decision",
            request_id=decision_record["request_id"],
            decision=decision_record["decision"],
            actor_id=decision_record["decision_actor_id"],
            status=decision_record["status"],
        )
        return decision_record

    def execute_approved_request(self, request_id: str) -> Dict:
        """
        Execute a specific approved consent request exactly once.

        Approval is consumed before low-level hardware delivery so failures are surfaced
        for audit and are not automatically retried.
        """
        current_request = self.consent_gate.get_current_request()
        if not current_request:
            raise ConsentExecutionError("no consent request is available for execution")
        if current_request["request_id"] != request_id:
            raise ConsentExecutionError("execution request_id does not match the current consent request")
        execution_payload = self.consent_gate.authorize_execution(
            request_id=request_id,
            modality=current_request["modality"],
            params=current_request["params"],
            target=current_request["target"],
        )
        try:
            self.mod_controller.set_target(execution_payload["target"])
            controller_result = self.mod_controller.configure_and_trigger(
                execution_payload["modality"],
                execution_payload["params"],
            )
        except Exception as exc:
            self._append_session_event(
                "intervention_delivery_failed",
                request_id=execution_payload["request_id"],
                modality=execution_payload["modality"],
                target=execution_payload["target"],
                error=str(exc),
            )
            raise RuntimeError(f"Hardware delivery failed for request {request_id}: {exc}") from exc

        result = {
            "request_id": execution_payload["request_id"],
            "status": "delivered",
            "modality": execution_payload["modality"],
            "target": execution_payload["target"],
            "params": execution_payload["params"],
            "controller_result": controller_result,
        }
        self._append_session_event(
            "intervention_delivered",
            request_id=result["request_id"],
            modality=result["modality"],
            target=result["target"],
            status=result["status"],
        )
        return result

    def record_intervention_feedback(self, request_id: str, feedback: str, actor_id: str) -> Dict:
        """Record post-intervention feedback without affecting future consent state."""
        allowed_feedback = {"helpful", "neutral", "unhelpful", "too_intense", "adverse_effect"}
        if not isinstance(feedback, str):
            raise ValueError(f"feedback must be one of {sorted(allowed_feedback)}")
        normalized_feedback = feedback.strip()
        if normalized_feedback not in allowed_feedback:
            raise ValueError(f"feedback must be one of {sorted(allowed_feedback)}")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must be a non-empty authenticated actor identifier")

        delivered_request = any(
            entry.get("request_id") == request_id and entry.get("event") == "intervention_delivered"
            for entry in self.session_history
        )
        if not delivered_request:
            raise ValueError(f"Post-intervention feedback requires a delivered request_id, got {request_id!r}")

        feedback_record = {
            "request_id": request_id,
            "feedback": normalized_feedback,
            "actor_id": actor_id.strip(),
            "recorded_at": time.time(),
        }
        self._append_session_event(
            "intervention_feedback",
            request_id=feedback_record["request_id"],
            feedback=feedback_record["feedback"],
            actor_id=feedback_record["actor_id"],
        )
        return feedback_record
