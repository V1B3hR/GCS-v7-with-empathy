import numpy as np
import tensorflow as tf
import logging
from .model import GradReverse

class GCSInference:
    def __init__(self, model_path, graph_path):
        logging.info(f"Loading foundational model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, custom_objects={"GradReverse": GradReverse})
        graph_data = np.load(graph_path)
        self.adj = np.expand_dims(graph_data['adjacency_matrix'], 0)
        self.inference_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer("mi_output").output, self.model.get_layer("gat_conv_1").output[1]]
        )
        self.labels = {0: "LEFT_HAND", 1: "RIGHT_HAND"}
        logging.info("Foundational Inference Engine ready.")

    def predict(self, source_localized_chunk):
        prediction, attention = self.inference_model.predict([source_localized_chunk, self.adj], verbose=0)
        intent_id = np.argmax(prediction[0])
        confidence = prediction[0][intent_id]
        return self.labels[intent_id], confidence, attention
