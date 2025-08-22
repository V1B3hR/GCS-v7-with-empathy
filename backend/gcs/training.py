
---
#### **FILE: `backend/gcs/training.py`**
```python
import numpy as np
import logging
import tensorflow as tf
from .data_pipeline import DataPipeline
from .model import FoundationalModelFactory
from .affective_state_classifier import AffectiveStateFactory
from .security import encrypt_file

class Trainer:
    def __init__(self, config):
        self.config = config
        self.pipeline = DataPipeline(config)
        graph_data = np.load(self.config["graph_scaffold_path"])
        self.adj_matrix = graph_data['adjacency_matrix']

    def run_loso_cross_validation(self):
        logging.info("Starting Leave-One-Subject-Out Cross-Validation...")
        all_data = self.pipeline.load_real_data_for_loso()
        # ... (Full LOSO logic as previously defined) ...
        logging.info("LOSO Cross-Validation Complete.")

    def train_affective_model(self):
        logging.info("--- Starting Training for Affective State Classifier ---")
        dataset_paths = [self.config['affective_model']['deap_dataset_path']]
        X, y = self.pipeline.load_affective_data(dataset_paths)
        model = AffectiveStateFactory.build_classifier(self.config)
        model.compile(optimizer='adam', loss={'valence_output': 'mae', 'arousal_output': 'mae'})
        model.fit([X['eeg'], X['physio'], X['voice']], [y['valence'], y['arousal']],
                  batch_size=self.config['batch_size'], epochs=self.config['epochs'], validation_split=0.2)
        model_path = self.config['affective_model']['output_model_path']
        model.save(model_path)
        logging.info(f"Affective model saved to {model_path}")
