import numpy as np
import tensorflow as tf
import logging
from .model import GradReverse

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

class GCSInference:
    def __init__(self, model_path, graph_path):
        logging.info(f"Loading foundational model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, custom_objects={"GradReverse": GradReverse}, safe_mode=False)
        graph_data = np.load(graph_path)
        self.adj = np.expand_dims(graph_data['adjacency_matrix'], 0)
        # Create inference model - note our simplified model doesn't have separate output layers
        # Use the main output for now
        self.inference_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer("mi_output").output
        )
        self.labels = {0: "LEFT_HAND", 1: "RIGHT_HAND"}
        logging.info("Foundational Inference Engine ready.")

    def predict(self, source_localized_chunk):
        prediction = self.inference_model.predict([source_localized_chunk], verbose=0)
        intent_id = np.argmax(prediction[0])
        confidence = prediction[0][intent_id]
        attention = np.random.random((68, 68))  # Placeholder for attention weights
        return self.labels[intent_id], confidence, attention
