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
from .feedback_detector import AdaptiveFeedbackDetector

# --- Helper Functions ---
def softmax_emotion_to_valence_arousal(prob: np.ndarray, config: Dict) -> Tuple[float, float]:
    """Maps emotion softmax probabilities to a continuous (valence, arousal) space using anchors from the config.
    
    Args:
        prob: Probability array from emotion classification model
        config: Configuration containing emotion mapping information
        
    Returns:
        Tuple of (valence, arousal) values
        
    Raises:
        TypeError: If prob is not a numpy array or config is not a dict
        ValueError: If prob dimensions or mapping configuration is invalid
    """
    if not isinstance(prob, np.ndarray):
        raise TypeError(f"prob must be numpy array, got {type(prob)}")
    if not isinstance(config, dict):
        raise TypeError(f"config must be dictionary, got {type(config)}")
    if prob.size == 0:
        raise ValueError("prob array cannot be empty")
    
    # Handle 2D probability arrays by taking first row
    if prob.ndim == 2:
        if prob.shape[0] == 0:
            raise ValueError("prob array cannot have zero rows")
        prob = prob[0]
    elif prob.ndim != 1:
        raise ValueError(f"prob must be 1D or 2D array, got {prob.ndim}D")

    # Load mapping from config with validation
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

    # Validate anchor data
    for key in keys:
        if key not in anchors:
            logging.warning(f"Missing anchor data for emotion class '{key}', using default values")
            continue
        anchor = anchors[key]
        if not isinstance(anchor, dict):
            raise ValueError(f"Anchor for '{key}' must be a dictionary")
        if "valence" not in anchor or "arousal" not in anchor:
            logging.warning(f"Missing valence/arousal in anchor for '{key}', using default values")

    # Calculate weighted valence and arousal
    try:
        val = sum(prob[i] * anchors.get(k, {}).get("valence", 0.0) for i, k in enumerate(keys))
        aro = sum(prob[i] * anchors.get(k, {}).get("arousal", 0.0) for i, k in enumerate(keys))
    except Exception as e:
        raise ValueError(f"Failed to calculate valence/arousal from emotion probabilities: {e}")

    # Scale valence from [-1, 1] to [0, 10] for consistency
    val_scaled = (val + 1.0) * 5.0
    
    # Validate output ranges
    if not (0 <= val_scaled <= 10):
        logging.warning(f"Valence {val_scaled:.3f} is outside expected range [0, 10]")
    if not (-10 <= aro <= 10):  # Allow wider range for arousal
        logging.warning(f"Arousal {aro:.3f} is outside typical range [-10, 10]")
    
    return float(val_scaled), float(aro)

# --- The Master Agent ---
class ClosedLoopAgent:
    """The master agent that orchestrates the full SENSE -> DECIDE -> ACT -> LEARN loop."""
    def __init__(self, config: Dict):
        """Initialize the Closed-Loop Agent with robust error handling and validation."""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(config)}")
        
        # Validate required config keys
        required_keys = ["output_model_dir", "graph_scaffold_path"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        self.config = config
        self.is_running = False
        
        # Validate and construct model paths
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
        
        # Initialize GCS inference engine with error handling
        try:
            # Check if GCSInference expects a config file path (as per PR #5 changes)
            graph_path = config["graph_scaffold_path"]
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph scaffold file not found: {graph_path}")
            
            # Try the original constructor first
            self.inference_engine = GCSInference(foundational_model_path, graph_path)
        except TypeError:
            # If that fails, try with config file approach as in PR #5
            logging.info("Attempting GCSInference initialization with config file approach")
            try:
                import tempfile
                import yaml
                
                # Create temporary config for GCSInference
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
                
                # Use context manager for safe temp file handling
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                    yaml.dump(temp_config, temp_file)
                    temp_config_path = temp_file.name
                
                try:
                    self.inference_engine = GCSInference(temp_config_path)
                finally:
                    # Ensure cleanup even if initialization fails
                    try:
                        os.unlink(temp_config_path)
                    except OSError as e:
                        logging.warning(f"Failed to clean up temporary config file {temp_config_path}: {e}")
            except Exception as e:
                logging.error(f"Failed to initialize GCSInference with config file approach: {e}")
                raise
        except Exception as e:
            logging.error(f"Failed to initialize GCS inference engine: {e}")
            raise
        
        # Load affective model with error handling
        try:
            self.affective_model = tf.keras.models.load_model(affective_model_path, safe_mode=False)
            logging.info(f"Affective model loaded successfully from {affective_model_path}")
        except Exception as e:
            logging.error(f"Failed to load affective model from {affective_model_path}: {e}")
            raise
        
        # Initialize other components with error handling
        try:
            self.mod_controller = NeuromodulationController(config)
        except Exception as e:
            logging.error(f"Failed to initialize NeuromodulationController: {e}")
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
        logging.info("Closed-Loop Agent initialized successfully with all components loaded.")

    def _policy_engine(self, cognitive_state: Dict, affective_state: Dict) -> Tuple[Optional[str], Optional[Dict]]:
        """Ethical and empathetic decision-making core with robust validation.
        
        Args:
            cognitive_state: Dictionary containing cognitive inference results
            affective_state: Dictionary containing affective inference results
            
        Returns:
            Tuple of (modality, parameters) or (None, None) if no intervention
            
        Raises:
            TypeError: If inputs are not dictionaries
            KeyError: If required keys are missing
            ValueError: If values are invalid
        """
        if not isinstance(cognitive_state, dict):
            raise TypeError(f"cognitive_state must be dictionary, got {type(cognitive_state)}")
        if not isinstance(affective_state, dict):
            raise TypeError(f"affective_state must be dictionary, got {type(affective_state)}")
        
        # Validate required keys
        required_cognitive_keys = ["intent", "confidence"]
        required_affective_keys = ["valence", "arousal"]
        
        missing_cognitive = [k for k in required_cognitive_keys if k not in cognitive_state]
        missing_affective = [k for k in required_affective_keys if k not in affective_state]
        
        if missing_cognitive:
            raise KeyError(f"Missing required cognitive state keys: {missing_cognitive}")
        if missing_affective:
            raise KeyError(f"Missing required affective state keys: {missing_affective}")
        
        try:
            valence = float(affective_state["valence"])
            arousal = float(affective_state["arousal"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid valence/arousal values: {e}")
        
        # Validate config for neuromodulation if intervention might be needed
        if "neuromodulation" not in self.config:
            logging.warning("neuromodulation section missing from config - interventions disabled")
            return None, None
        
        # Example policy with enhanced logging and validation
        pain_confidence = cognitive_state.get("confidence", 0.0)
        try:
            pain_confidence = float(pain_confidence)
        except (ValueError, TypeError):
            logging.error(f"Invalid confidence value: {pain_confidence}")
            return None, None
        
        arousal_threshold = 7.0
        confidence_threshold = 0.85
        
        # Log decision factors for transparency
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
        """Run affective model inference with robust error handling and input validation."""
        if not isinstance(live_data, dict):
            raise TypeError(f"live_data must be a dictionary, got {type(live_data)}")
        
        try:
            # Validate that affective model is loaded
            if not hasattr(self, 'affective_model') or self.affective_model is None:
                raise RuntimeError("Affective model is not loaded")
            
            # Get model input specifications
            input_names = [t.name for t in self.affective_model.inputs]
            if not input_names:
                raise RuntimeError("Affective model has no inputs defined")
            
            # Map expected inputs to available data
            name_to_data = {
                "node_input": live_data.get("source_eeg"), 
                "physio_input": live_data.get("physio"), 
                "voice_input": live_data.get("voice")
            }
            
            # Validate and prepare model inputs
            model_inputs = []
            for nm in input_names:
                key = nm.split(":")[0]
                if key not in name_to_data:
                    raise KeyError(f"Affective model expects input '{key}', but it's not mapped in name_to_data")
                if name_to_data[key] is None:
                    raise ValueError(f"Affective model expects '{key}', but no data was provided in live_data")
                
                # Validate input data shape and type
                input_data = name_to_data[key]
                if not isinstance(input_data, np.ndarray):
                    raise TypeError(f"Input '{key}' must be a numpy array, got {type(input_data)}")
                if input_data.size == 0:
                    raise ValueError(f"Input '{key}' cannot be empty")
                    
                model_inputs.append(input_data)
            
            # Run model prediction with error handling
            try:
                preds = self.affective_model.predict(model_inputs, verbose=0)
            except Exception as e:
                logging.error(f"Affective model prediction failed: {e}")
                raise RuntimeError(f"Affective model prediction failed: {e}")
            
            # Process prediction outputs
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
                
                # Check for direct valence/arousal outputs
                if "valence_output" in name_to_pred and "arousal_output" in name_to_pred:
                    try:
                        v, a = name_to_pred["valence_output"], name_to_pred["arousal_output"]
                        valence = float(np.array(v).reshape(-1)[0])
                        arousal = float(np.array(a).reshape(-1)[0])
                        return valence, arousal
                    except (IndexError, ValueError) as e:
                        logging.error(f"Failed to extract valence/arousal from direct outputs: {e}")
                        raise ValueError(f"Failed to extract valence/arousal from direct outputs: {e}")
                
                # Fall back to emotion mapping
                return softmax_emotion_to_valence_arousal(preds[0], self.config)
            
            else:
                raise RuntimeError(f"Unexpected affective model prediction type: {type(preds)}")
                
        except Exception as e:
            logging.error(f"Affective inference failed: {e}")
            raise

    def run_cycle(self, live_data: Dict):
        """Execute the full SENSE -> DECIDE -> ACT -> LEARN loop with robust error handling."""
        if not isinstance(live_data, dict):
            raise TypeError(f"live_data must be a dictionary, got {type(live_data)}")
        
        try:
            # Validate required input data
            required_data_keys = ["source_eeg", "adj_matrix"]
            missing_keys = [key for key in required_data_keys if key not in live_data]
            if missing_keys:
                raise ValueError(f"Missing required data keys: {missing_keys}")
            
            # --- 1) SENSE ---
            try:
                source_chunk = live_data["source_eeg"]
                adj_matrix = live_data["adj_matrix"]
                
                # Validate input data types and shapes
                if not isinstance(source_chunk, np.ndarray):
                    raise TypeError(f"source_eeg must be numpy array, got {type(source_chunk)}")
                if not isinstance(adj_matrix, np.ndarray):
                    raise TypeError(f"adj_matrix must be numpy array, got {type(adj_matrix)}")
                if source_chunk.size == 0:
                    raise ValueError("source_eeg cannot be empty")
                if adj_matrix.size == 0:
                    raise ValueError("adj_matrix cannot be empty")
                
                # Run cognitive inference
                intent, conf, attention = self.inference_engine.predict([source_chunk, adj_matrix])
                cognitive_state = {"intent": intent, "confidence": float(conf)}
                
                # Run affective inference
                valence, arousal = self._run_affective_inference(live_data)
                affective_state = {"valence": valence, "arousal": arousal}
                
                logging.info(f"[SENSE] Cognitive: {intent} ({conf:.2f}) | Affective: Valence={valence:.2f}, Arousal={arousal:.2f}")
                
            except Exception as e:
                logging.error(f"SENSE phase failed: {e}")
                raise RuntimeError(f"SENSE phase failed: {e}")
            
            # --- 2) DECIDE ---
            try:
                modality, params = self._policy_engine(cognitive_state, affective_state)
            except Exception as e:
                logging.error(f"DECIDE phase failed: {e}")
                raise RuntimeError(f"DECIDE phase failed: {e}")
            
            # --- 3) ACT ---
            if modality:
                try:
                    logging.info(f"[ACTION] Proposing {modality} intervention to user...")
                    self.mod_controller.configure_and_trigger(modality, params)
                except Exception as e:
                    logging.error(f"ACT phase failed: {e}")
                    # Don't raise here as this might be recoverable
                    logging.warning("Continuing despite ACT phase failure")
            else:
                logging.debug("[ACTION] No intervention recommended")
            
            # --- 4) LEARN ---
            # (Feedback loop integration placeholder)
            # Future: Add online learning and feedback integration here
            
        except Exception as e:
            logging.error(f"run_cycle failed: {e}")
            # Set agent to safe state
            self.is_running = False
            raise
    
    def cleanup(self):
        """Clean up resources and reset agent state."""
        try:
            self.is_running = False
            logging.info("Closed-loop agent cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict:
        """Get current agent status for monitoring."""
        try:
            return {
                "is_running": self.is_running,
                "has_inference_engine": hasattr(self, 'inference_engine') and self.inference_engine is not None,
                "has_affective_model": hasattr(self, 'affective_model') and self.affective_model is not None,
                "has_mod_controller": hasattr(self, 'mod_controller') and self.mod_controller is not None,
                "has_feedback_detector": hasattr(self, 'feedback_detector') and self.feedback_detector is not None,
                "has_olm": hasattr(self, 'olm') and self.olm is not None
            }
        except Exception as e:
            logging.error(f"Error getting status: {e}")
            return {"error": str(e)}
