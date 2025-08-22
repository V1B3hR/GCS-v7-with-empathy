import logging
import time
import numpy as np
from .inference import GCSInference
from .affective_state_classifier import AffectiveStateFactory
from .neuromodulation_controller import NeuromodulationController
from .online_learning_module import OnlineLearningModule
from .feedback_detector import FeedbackDetector

class ClosedLoopAgent:
    """
    The master agent that orchestrates the full SENSE -> DECIDE -> ACT -> LEARN loop.
    This is the highest level of the GCS's operational logic.
    """
    def __init__(self, config: dict):
        self.config = config
        self.is_running = False
        
        # --- Load Foundational & Affective Models ---
        foundational_model_path = f"{config['output_model_dir']}gcs_fold_1.h5"
        affective_model_path = config['affective_model']['output_model_path']
        
        # --- Initialize Core Components ---
        self.inference_engine = GCSInference(foundational_model_path, config['graph_scaffold_path'])
        self.affective_model = tf.keras.models.load_model(affective_model_path)
        self.mod_controller = NeuromodulationController(config)
        self.feedback_detector = FeedbackDetector(config)
        self.olm = OnlineLearningModule(self.inference_engine.model, config)

        # --- State Management for Learning ---
        self.last_state = None
        self.last_action_index = None
        
        logging.info("Closed-Loop Agent is online and all components are loaded.")

    def _policy_engine(self, cognitive_state: dict, affective_state: dict) -> (str, dict) or (None, None):
        """
        The ethical and empathetic decision-making core.
        Decides if an action is needed based on the full context of the user's state.
        """
        # EXAMPLE POLICY: If a high-confidence pain signature is detected AND the user
        # is in a high-arousal (stressed) state, recommend intervention.
        valence = affective_state['valence']
        arousal = affective_state['arousal']
        
        if cognitive_state['intent'] == "PAIN_SIGNATURE" and cognitive_state['confidence'] > 0.85 and arousal > 7.0:
            logging.warning("[POLICY] High-confidence pain signature detected in a high-stress state. Recommending intervention.")
            modality = "ultrasound"
            params = self.config['neuromodulation']['ultrasound_params']
            return modality, params
        
        return None, None

    def run_cycle(self, live_data: dict):
        """Runs a single cycle of the main operational loop."""
        
        # --- 1. SENSE ---
        source_chunk = live_data['source_eeg']
        adj_matrix = live_data['adj_matrix']
        
        intent, conf, attention = self.inference_engine.predict([source_chunk, adj_matrix])
        cognitive_state = {'intent': intent, 'confidence': conf}
        
        valence, arousal = self.affective_model.predict([live_data['source_eeg'], live_data['physio'], live_data['voice']], verbose=0)
        affective_state = {'valence': valence[0][0], 'arousal': arousal[0][0]}
        
        print(f"[SENSE] Cognitive: {intent} ({conf:.2f}) | Affective: Valence={valence[0][0]:.2f}, Arousal={arousal[0][0]:.2f}")

        # --- 2. DECIDE ---
        modality, params = self._policy_engine(cognitive_state, affective_state)

        # --- 3. ACT ---
        if modality:
            # CRITICAL: Human-in-the-Loop Consent would be requested here.
            logging.info("[ACTION] Proposing intervention to user...")
            # if user_gives_consent():
            self.mod_controller.configure_and_trigger(modality, params)
        
        # --- 4. LEARN ---
        # This part would be triggered by the live connector's feedback loop
        # For now, we simulate it.
        # error_detected = self.feedback_detector.detect_corrective_signal(...)
        # self.olm.apply_feedback(error_detected)
