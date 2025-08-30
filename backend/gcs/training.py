import os
import numpy as np
import logging
import tensorflow as tf
from .data_pipeline import DataPipeline
from .model import GCSModelFactory
from .affective_state_classifier import AffectiveModelBuilder

class Trainer:
    def __init__(self, config):
        self.config = config
        self.pipeline = DataPipeline(config)
        graph_data = np.load(self.config["graph_scaffold_path"])
        self.adj_matrix = graph_data['adjacency_matrix']

    def run_loso_cross_validation(self):
        logging.info("Starting Leave-One-Subject-Out Cross-Validation...")
        try:
            all_data = self.pipeline.load_real_data_for_loso()
            model = GCSModelFactory.build_affective_model(self.config)
            for fold, held_out_subject in enumerate(all_data):
                logging.info(f"Processing fold {fold + 1}/{len(all_data)}")
                train_data = [subj for i, subj in enumerate(all_data) if i != fold]
                if not train_data:
                    logging.warning(f"No training data for fold {fold + 1}")
                    continue
                # TODO: Extract features/labels from train_data
                # Example:
                # X_train, y_train = extract_features_labels(train_data)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # Uncomment when feature extraction is implemented:
                # model.fit(X_train, y_train, batch_size=self.config.get('batch_size', 32), epochs=self.config.get('epochs', 10))
                model_path = os.path.join(self.config.get('output_model_dir', 'models'), f"gcs_fold_{fold + 1}.h5")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                try:
                    model.save(model_path)
                    logging.info(f"Model for fold {fold + 1} saved to {model_path}")
                except Exception as e:
                    logging.error(f"Failed to save model for fold {fold + 1}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"LOSO Cross-Validation failed: {e}", exc_info=True)

        logging.info("LOSO Cross-Validation Complete.")

    def train_affective_model(self):
        logging.info("--- Starting Training for Affective State Classifier ---")
        try:
            foundational_model = GCSModelFactory.build_affective_model(self.config)
            model = AffectiveModelBuilder.build_fused_classifier(self.config, foundational_model)
            dataset_paths = [self.config['affective_model']['deap_dataset_path']]
            X, y = self.pipeline.load_affective_data(dataset_paths)
            model.compile(optimizer='adam', loss={'valence_output': 'mae', 'arousal_output': 'mae'})

            logging.info(f"Model expects {len(model.inputs)} inputs: {[inp.name for inp in model.inputs]}")
            logging.info(f"Data shapes: EEG {X.get('eeg', 'missing')}, Physio {X.get('physio', 'missing')}, Voice {X.get('voice', 'missing')}")
            inputs = [X['eeg'], X['physio'], X['voice']]
            targets = [y['valence'], y['arousal']]
            model.fit(inputs, targets,
                      batch_size=self.config['batch_size'], epochs=self.config['epochs'], validation_split=0.2)
            model_path = self.config['affective_model']['output_model_path']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                model.save(model_path)
                logging.info(f"Affective model saved to {model_path}")
            except Exception as e:
                logging.error(f"Failed to save affective model: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Affective model training failed: {e}", exc_info=True)
