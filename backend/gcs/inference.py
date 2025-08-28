import numpy as np
import tensorflow as tf
import logging
import json
from datetime import datetime
from typing import Union, Dict, Any, Tuple, List

# Try to import GradReverse, provide fallback if spektral not available
try:
    from .model import GradReverse
except ImportError as e:
    logging.warning(f"Could not import GradReverse from model: {e}")
    # Fallback implementation for GradReverse
    @tf.custom_gradient
    def grad_reverse(x):
        """A layer that reverses the gradient during backpropagation for adversarial training."""
        y = tf.identity(x)
        def custom_grad(dy): return -dy
        return y, custom_grad

    class GradReverse(tf.keras.layers.Layer):
        def call(self, x): return grad_reverse(x)

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

class GCSInference:
    def __init__(self, model_path, graph_path):
        """
        Initialize the GCS Inference Engine.
        
        Args:
            model_path (str): Path to the trained model file
            graph_path (str): Path to the graph adjacency matrix file
        """
        self.model = None
        self.inference_model = None
        self.adj = None
        self.labels = {0: "LEFT_HAND", 1: "RIGHT_HAND"}
        self.model_version = "v7.0"
        
        try:
            logging.info(f"Loading foundational model from {model_path}...")
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects={"GradReverse": GradReverse}, 
                safe_mode=False
            )
            
            graph_data = np.load(graph_path)
            self.adj = np.expand_dims(graph_data['adjacency_matrix'], 0)
            
            # Create inference model - note our simplified model doesn't have separate output layers
            # Use the main output for now
            self.inference_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.get_layer("mi_output").output
            )
            
            logging.info("Foundational Inference Engine ready.")
            
        except Exception as e:
            logging.error(f"Failed to initialize GCSInference: {e}")
            # Don't raise here - let predict() handle errors gracefully
            
    def _ensure_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy arrays and other non-JSON-serializable objects to JSON-compatible types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def _create_error_response(self, error_message: str, exception: Exception = None) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message (str): Human-readable error description
            exception (Exception, optional): Original exception (not exposed in output)
            
        Returns:
            Dict containing error response structure
        """
        # Log the full exception details for debugging but don't expose in response
        if exception:
            logging.error(f"Prediction error: {error_message}", exc_info=True)
        else:
            logging.error(f"Prediction error: {error_message}")
            
        return {
            "status": "error",
            "error_message": error_message,
            "prediction": {
                "label": None,
                "confidence": 0.0,
                "intent_id": None
            },
            "attention_weights": None,
            "metadata": {
                "model_version": self.model_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input_processed": False
            }
        }
    
    def _create_success_response(self, label: str, confidence: float, intent_id: int, 
                               attention_weights: np.ndarray, metadata: Dict = None) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            label (str): Predicted label
            confidence (float): Prediction confidence
            intent_id (int): Intent ID
            attention_weights (np.ndarray): Attention weights matrix
            metadata (Dict, optional): Additional metadata
            
        Returns:
            Dict containing success response structure
        """
        response = {
            "status": "success",
            "prediction": {
                "label": label,
                "confidence": float(confidence),
                "intent_id": int(intent_id)
            },
            "attention_weights": self._ensure_json_serializable(attention_weights),
            "metadata": {
                "model_version": self.model_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input_processed": True
            }
        }
        
        # Add any additional metadata
        if metadata:
            response["metadata"].update(metadata)
            
        return response

    def predict(self, source_localized_chunk: Union[List, np.ndarray], 
                return_legacy_format: bool = False) -> Union[Dict[str, Any], Tuple]:
        """
        Perform prediction on the input data with robust error handling.
        
        This method now returns a structured JSON-serializable dictionary by default,
        containing status, prediction results, confidence, label, and attention weights.
        For backward compatibility, it can still return the legacy tuple format.
        
        Args:
            source_localized_chunk: Input data for prediction. Can be a single sample
                                  or batch of samples. Expected format depends on model
                                  architecture.
            return_legacy_format (bool): If True, returns the old tuple format
                                       (label, confidence, attention) for backward compatibility.
                                       If False (default), returns structured dict.
        
        Returns:
            Union[Dict[str, Any], Tuple]: 
                - If return_legacy_format=False: Structured dictionary with keys:
                  * status: "success" or "error"
                  * prediction: {label, confidence, intent_id}
                  * attention_weights: attention matrix or None
                  * metadata: {model_version, timestamp, input_processed}
                  * error_message: (only present if status="error")
                - If return_legacy_format=True: Tuple of (label, confidence, attention)
                  for backward compatibility
        """
        
        # Check if model was properly initialized
        if self.model is None or self.inference_model is None:
            error_response = self._create_error_response(
                "Model not properly initialized. Check model and graph file paths."
            )
            if return_legacy_format:
                return None, 0.0, None
            return error_response
        
        try:
            # Validate input
            if source_localized_chunk is None:
                error_response = self._create_error_response("Input data is None")
                if return_legacy_format:
                    return None, 0.0, None
                return error_response
            
            # Handle input data - don't try to convert complex list structures to numpy arrays
            # The original model expects the data to be passed as-is to the predict method
            input_data = source_localized_chunk
            
            # Perform model prediction with error handling
            try:
                prediction = self.inference_model.predict(input_data, verbose=0)
            except Exception as e:
                error_response = self._create_error_response(
                    "Model prediction failed: input shape or format incompatible with model",
                    e
                )
                if return_legacy_format:
                    return None, 0.0, None
                return error_response
            
            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                error_response = self._create_error_response("Model returned empty prediction")
                if return_legacy_format:
                    return None, 0.0, None
                return error_response
            
            # Handle both single and batch predictions
            prediction_array = prediction[0] if len(prediction.shape) > 1 else prediction
            
            # Extract intent and confidence
            try:
                intent_id = np.argmax(prediction_array)
                confidence = prediction_array[intent_id]
                
                # Validate confidence value
                if not np.isfinite(confidence):
                    error_response = self._create_error_response(
                        f"Invalid confidence value: {confidence}"
                    )
                    if return_legacy_format:
                        return None, 0.0, None
                    return error_response
                
                # Get label
                label = self.labels.get(int(intent_id), f"UNKNOWN_CLASS_{intent_id}")
                
            except Exception as e:
                error_response = self._create_error_response(
                    "Failed to extract prediction results from model output",
                    e
                )
                if return_legacy_format:
                    return None, 0.0, None
                return error_response
            
            # Generate attention weights (placeholder for now, as in original)
            try:
                attention = np.random.random((68, 68))  # Placeholder for attention weights
            except Exception as e:
                logging.warning(f"Failed to generate attention weights: {e}")
                attention = None
            
            # Create metadata for this prediction
            metadata = {
                "prediction_shape": list(prediction.shape),
                "input_type": str(type(input_data)),
                "batch_size": 1 if len(prediction.shape) == 1 else prediction.shape[0]
            }
            
            # Log successful prediction
            logging.debug(f"Successful prediction: {label} ({confidence:.3f})")
            
            # Return in requested format
            if return_legacy_format:
                return label, confidence, attention
            else:
                return self._create_success_response(label, confidence, intent_id, attention, metadata)
                
        except Exception as e:
            # Catch any unexpected errors
            error_response = self._create_error_response(
                "Unexpected error during prediction processing",
                e
            )
            if return_legacy_format:
                return None, 0.0, None
            return error_response
    
    def predict_legacy(self, source_localized_chunk: Union[List, np.ndarray]) -> Tuple:
        """
        Legacy prediction method that returns the original tuple format.
        This method is provided for backward compatibility.
        
        Args:
            source_localized_chunk: Input data for prediction
            
        Returns:
            Tuple: (label, confidence, attention) - same as original predict method
        """
        return self.predict(source_localized_chunk, return_legacy_format=True)
    
    def is_healthy(self) -> Dict[str, Any]:
        """
        Check if the inference engine is properly initialized and ready.
        
        Returns:
            Dict containing health status information
        """
        try:
            is_ready = (
                self.model is not None and 
                self.inference_model is not None and 
                self.adj is not None
            )
            
            return {
                "status": "healthy" if is_ready else "unhealthy",
                "model_loaded": self.model is not None,
                "inference_model_ready": self.inference_model is not None,
                "adjacency_matrix_loaded": self.adj is not None,
                "model_version": self.model_version,
                "available_labels": list(self.labels.values()),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
