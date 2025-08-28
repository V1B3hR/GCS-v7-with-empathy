import os
import time
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

from .inference import GCSInference
from .neuromodulation_controller import NeuromodulationController
from .online_learning_module import OnlineLearningModule
from .feedback_detector import FeedbackDetector

# --- Helper Functions ---
def softmax_emotion_to_valence_arousal(prob: np.ndarray, config: Dict) -> Tuple[float, float]:
    """Maps emotion softmax probabilities to a continuous (valence, arousal) space using anchors from the config."""
    if prob.ndim == 2:
        prob = prob[0]

    # BEST PRACTICE: Load mapping from config, not hard-coded
    mapping_config = config.get("affective_model", {}).get("emotion_mapping", {})
    keys = mapping_config.get("class_order", [])
    anchors = mapping_config.get("anchors", {})

    if len(prob) != len(keys):
        raise ValueError(f"Model output size ({len(prob)}) does not match emotion classes in config ({len(keys)}).")

    val = sum(prob[i] * anchors.get(k, {}).get("valence", 0.0) for i, k in enumerate(keys))
    aro = sum(prob[i] * anchors.get(k, {}).get("arousal", 0.0) for i, k in enumerate(keys))

    # Scale valence from [-1, 1] to [0, 10] for consistency
    val_scaled = (val + 1.0) * 5.0
    return float(val_scaled), float(aro)

# --- The Master Agent ---
class ClosedLoopAgent:
    """The master agent that orchestrates the full SENSE -> DECIDE -> ACT -> LEARN loop."""
    def __init__(self, config: Dict):
        # ... (init logic remains the same, it's already excellent)
        self.config = config
        self.is_running = False
        foundational_model_path = os.path.join(config["output_model_dir"], "gcs_fold_1.h5")
        affective_model_path = config.get("affective_model", {}).get("output_model_path")
        self.inference_engine = GCSInference(foundational_model_path, config["graph_scaffold_path"])
        
        self.affective_model = tf.keras.models.load_model(affective_model_path, safe_mode=False)
        
        self.mod_controller = NeuromodulationController(config)
        self.feedback_detector = FeedbackDetector(config)
        self.olm = OnlineLearningModule(self.inference_engine.model, config)
        self.last_state = None
        self.last_action_index = None
        logging.info("Closed-Loop Agent is online and all components are loaded.")

    def _policy_engine(self, cognitive_state: Dict, affective_state: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Ethical and empathetic decision-making core."""
        valence = affective_state["valence"]
        arousal = affective_state["arousal"]
        
        # Example policy with enhanced logging
        pain_confidence = cognitive_state.get("confidence", 0.0)
        arousal_threshold = 7.0
        
        if cognitive_state.get("intent") == "PAIN_SIGNATURE" and pain_confidence > 0.85 and arousal > arousal_threshold:
            # RECOMMENDATION: Add detailed logging for transparency
            logging.warning(f"[POLICY] Condition met: High-confidence pain signature ({pain_confidence:.2f}) and high arousal ({arousal:.2f} > {arousal_threshold}).")
            logging.info("[POLICY] Recommending 'ultrasound' intervention.")
            modality = "ultrasound"
            params = self.config["neuromodulation"].get("ultrasound_params", {})
            return modality, params

        return None, None

    def _run_affective_inference(self, live_data: Dict) -> Tuple[float, float]:
        # ... (This function is already excellent and requires no changes)
        model_inputs = []
        input_names = [t.name for t in self.affective_model.inputs]
        name_to_data = {"node_input": live_data.get("source_eeg"), 
                       "physio_input": live_data.get("physio"), 
                       "voice_input": live_data.get("voice")}
        for nm in input_names:
            key = nm.split(":")[0]
            if key not in name_to_data or name_to_data[key] is None:
                raise KeyError(f"Affective model expects '{key}', but no data was provided.")
            model_inputs.append(name_to_data[key])
        preds = self.affective_model.predict(model_inputs, verbose=0)
        if isinstance(preds, np.ndarray):
            return softmax_emotion_to_valence_arousal(preds, self.config)
        if isinstance(preds, (list, tuple)):
            output_names = [t.name for t in self.affective_model.outputs]
            name_to_pred = {output_names[i].split(":")[0]: preds[i] for i in range(len(preds))}
            if "valence_output" in name_to_pred and "arousal_output" in name_to_pred:
                v, a = name_to_pred["valence_output"], name_to_pred["arousal_output"]
                return float(np.array(v).reshape(-1)[0]), float(np.array(a).reshape(-1)[0])
            return softmax_emotion_to_valence_arousal(preds[0], self.config)
        raise RuntimeError("Unexpected affective_model.predict() return type.")

    def run_cycle(self, live_data: Dict):
        # ... (This function is already excellent and requires no changes)
        # --- 1) SENSE ---
        source_chunk = live_data["source_eeg"]
        adj_matrix = live_data["adj_matrix"]
        intent, conf, attention = self.inference_engine.predict([source_chunk, adj_matrix], return_legacy_format=True)
        cognitive_state = {"intent": intent, "confidence": float(conf)}
        valence, arousal = self._run_affective_inference(live_data)
        affective_state = {"valence": valence, "arousal": arousal}
        logging.info(f"[SENSE] Cognitive: {intent} ({conf:.2f}) | Affective: Valence={valence:.2f}, Arousal={arousal:.2f}")
        # --- 2) DECIDE ---
        modality, params = self._policy_engine(cognitive_state, affective_state)
        # --- 3) ACT ---
        if modality:
            logging.info("[ACTION] Proposing intervention to user...")
            self.mod_controller.configure_and_trigger(modality, params)
        # --- 4) LEARN ---
        # (Feedback loop integration placeholder)
