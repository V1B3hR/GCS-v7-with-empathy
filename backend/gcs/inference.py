import numpy as np
import tensorflow as tf
import logging
import yaml
import time
from .model import GradReverse

class GCSInference:
    def __init__(self, config_path="backend/gcs/config.yaml"):
        # Load config
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.critical("Failed to load config file", exc_info=True)
            raise

        # Centralized config values
        self.model_path = config["model_path"]
        self.graph_path = config["graph_path"]
        self.labels = config.get("labels", {0: "LEFT_HAND", 1: "RIGHT_HAND"})
        self.batch_size = config.get("batch_size", 32)
        self.safe_mode = config.get("safe_mode", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.monitoring = config.get("monitoring", False)
        self.output_layers = config.get("output_layers")
        self.attention_extractor = config.get("attention_extractor")  # Optional plugin

        # Model loading (no unsafe deserialization by default)
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={"GradReverse": GradReverse},
                safe_mode=self.safe_mode
            )
        except Exception as e:
            logging.critical("Error loading model", exc_info=True)
            raise

        # Graph adjacency matrix validation
        try:
            graph_data = np.load(self.graph_path)
            adj = graph_data["adjacency_matrix"]
            if adj.shape[0] != adj.shape[1]:
                raise ValueError("Adjacency matrix must be square.")
            if not np.allclose(adj, adj.T):
                logging.warning("Adjacency matrix is not symmetric.")
            self.adj = np.expand_dims(adj, 0)
        except Exception as e:
            logging.error("Error loading graph data", exc_info=True)
            raise

        # Output layers selection/validation
        layer_names = [layer.name for layer in self.model.layers]
        if self.output_layers:
            missing = [name for name in self.output_layers if name not in layer_names]
            if missing:
                logging.error(f"Output layers not found in model: {missing}")
                raise ValueError(f"Output layers not found in model: {missing}")
            outputs = [self.model.get_layer(name).output for name in self.output_layers]
            self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=outputs)
            self.output_names = self.output_layers
        else:
            if len(self.model.outputs) > 1:
                self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.outputs)
                self.output_names = [out.name for out in self.model.outputs]
            else:
                if "mi_output" in layer_names:
                    self.inference_model = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.get_layer("mi_output").output)
                    self.output_names = ["mi_output"]
                else:
                    logging.critical("Fallback layer 'mi_output' not found in model.")
                    raise ValueError("Fallback layer 'mi_output' not found in model.")

        logging.info("Foundational Inference Engine ready.")

    def predict(self, source_localized_chunks):
        # Input batch dimension validation
        expected_shape = tuple(self.model.inputs[0].shape[1:])
        if source_localized_chunks.shape[1:] != expected_shape:
            logging.error(f"Input shape mismatch. Expected {expected_shape}, got {source_localized_chunks.shape[1:]}")
            raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {source_localized_chunks.shape[1:]}")

        start = time.perf_counter()
        try:
            predictions = self.inference_model.predict(source_localized_chunks, batch_size=self.batch_size, verbose=0)
        except Exception as e:
            logging.critical("Prediction failed for entire batch", exc_info=True)
            raise

        results = []
        # Multi-output handling
        if isinstance(predictions, list):
            for i in range(len(source_localized_chunks)):
                result = {}
                for idx, output in enumerate(predictions):
                    pred = output[i]
                    try:
                        if pred.size == 0:
                            raise ValueError(f"Prediction output {self.output_names[idx]} is empty for sample {i}.")
                        intent_id = np.argmax(pred)
                        confidence = pred[intent_id]
                        label = self.labels.get(intent_id, intent_id)
                        # Attention extraction if enabled
                        if self.attention_extractor:
                            attention = self.attention_extractor(pred)
                        else:
                            attention = None
                        result[self.output_names[idx]] = {
                            "label": label,
                            "confidence": float(confidence),
                            "raw": pred.tolist(),
                            "attention": attention
                        }
                        if self.monitoring and confidence < self.confidence_threshold:
                            logging.warning(f"Low confidence for sample {i}, output {self.output_names[idx]}: {confidence}")
                    except Exception as e:
                        logging.error(f"Sample {i}, output {self.output_names[idx]} prediction error", exc_info=True)
                        result[self.output_names[idx]] = {"error": str(e)}
                results.append(result)
        else:
            for i, pred in enumerate(predictions):
                try:
                    if pred.size == 0:
                        raise ValueError("Prediction array is empty.")
                    intent_id = np.argmax(pred)
                    confidence = pred[intent_id]
                    label = self.labels.get(intent_id, intent_id)
                    if self.attention_extractor:
                        attention = self.attention_extractor(pred)
                    else:
                        attention = None
                    result = {
                        "label": label,
                        "confidence": float(confidence),
                        "raw": pred.tolist(),
                        "attention": attention
                    }
                    if self.monitoring and confidence < self.confidence_threshold:
                        logging.warning(f"Low confidence for sample {i}: {confidence}")
                except Exception as e:
                    logging.error(f"Sample {i} prediction error", exc_info=True)
                    result = {"error": str(e)}
                results.append(result)

        latency = time.perf_counter() - start
        if self.monitoring:
            logging.info(f"Prediction latency: {latency:.4f}s")
            # Class distribution histogram
            try:
                class_ids = [
                    np.argmax(r[self.output_names[0]]["raw"])
                    for r in results
                    if self.output_names[0] in r and "raw" in r[self.output_names[0]]
                ]
                class_hist = {k: class_ids.count(k) for k in set(class_ids)}
                logging.info(f"Class distribution: {class_hist}")
            except Exception as e:
                logging.warning("Unable to compute class distribution histogram", exc_info=True)
        return results

    def get_model_info(self):
        return {
            "inputs": [layer.name for layer in self.model.inputs],
            "outputs": self.output_names,
            "layers": [layer.name for layer in self.model.layers],
        }
