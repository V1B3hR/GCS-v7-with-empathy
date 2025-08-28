import numpy as np
import tensorflow as tf
import logging
from .model import GradReverse

tf.keras.config.enable_unsafe_deserialization()

class GCSInference:
    def __init__(self, model_path, graph_path, labels=None, output_layers=None):
        logging.info(f"Loading foundational model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, custom_objects={"GradReverse": GradReverse}, safe_mode=False)
        graph_data = np.load(graph_path)
        self.adj = np.expand_dims(graph_data['adjacency_matrix'], 0)
        
        # Detect output layers
        if output_layers:
            outputs = [self.model.get_layer(name).output for name in output_layers]
            self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=outputs)
            self.output_names = output_layers
        else:
            # Use all outputs if available, otherwise fallback
            if len(self.model.outputs) > 1:
                self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.outputs)
                self.output_names = [out.name for out in self.model.outputs]
            else:
                self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.get_layer("mi_output").output)
                self.output_names = ["mi_output"]

        # Configurable labels
        self.labels = labels or {0: "LEFT_HAND", 1: "RIGHT_HAND"}
        logging.info("Foundational Inference Engine ready.")

    def predict(self, source_localized_chunks):
        # Batch prediction
        predictions = self.inference_model.predict(source_localized_chunks, verbose=0)
        results = []
        # Handle multi-output
        if isinstance(predictions, list):
            for i in range(len(source_localized_chunks)):
                result = {}
                for idx, output in enumerate(predictions):
                    pred = output[i]
                    intent_id = np.argmax(pred)
                    confidence = pred[intent_id]
                    result[self.output_names[idx]] = {
                        "label": self.labels.get(intent_id, intent_id),
                        "confidence": float(confidence),
                        "raw": pred.tolist()
                    }
                # Placeholder for attention
                result["attention"] = np.random.random((68, 68)).tolist()
                results.append(result)
        else:
            for pred in predictions:
                intent_id = np.argmax(pred)
                confidence = pred[intent_id]
                results.append({
                    "label": self.labels.get(intent_id, intent_id),
                    "confidence": float(confidence),
                    "raw": pred.tolist(),
                    "attention": np.random.random((68, 68)).tolist()
                })
        return results

    def get_model_info(self):
        return {
            "inputs": [layer.name for layer in self.model.inputs],
            "outputs": self.output_names,
            "layers": [layer.name for layer in self.model.layers],
        }
