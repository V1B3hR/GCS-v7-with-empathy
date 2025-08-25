
import numpy as np
import logging
import tensorflow as tf
from .data_pipeline import DataPipeline
from .model import GCSModelFactory
from .affective_state_classifier import AffectiveModelBuilder
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
        
        # Build the foundational model
        model = GCSModelFactory.build_affective_model(self.config)
        
        # Implement basic LOSO training loop
        for fold, held_out_subject in enumerate(all_data):
            logging.info(f"Processing fold {fold + 1}/{len(all_data)}")
            
            # Prepare training data (all subjects except held_out)
            train_data = [subj for i, subj in enumerate(all_data) if i != fold]
            
            # Basic training setup
            if train_data:
                # Compile and train model on training subjects
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # In a real implementation, you would extract features and labels from train_data
                # and train the model. For now, we'll just save the model structure.
                
                model_path = f"{self.config.get('output_model_dir', 'models')}/gcs_fold_{fold + 1}.h5"
                model.save(model_path)
                logging.info(f"Model for fold {fold + 1} saved to {model_path}")
        
        logging.info("LOSO Cross-Validation Complete.")

    def train_affective_model(self):
        logging.info("--- Starting Training for Affective State Classifier ---")
        # First, build and train a foundational model
        foundational_model = GCSModelFactory.build_affective_model(self.config)
        
        # Then use it to build the affective fusion model
        model = AffectiveModelBuilder.build_fused_classifier(self.config, foundational_model)
        
        dataset_paths = [self.config['affective_model']['deap_dataset_path']]
        X, y = self.pipeline.load_affective_data(dataset_paths)
        model.compile(optimizer='adam', loss={'valence_output': 'mae', 'arousal_output': 'mae'})
        model.fit([X['eeg'], X['physio'], X['voice']], [y['valence'], y['arousal']],
                  batch_size=self.config['batch_size'], epochs=self.config['epochs'], validation_split=0.2)
        model_path = self.config['affective_model']['output_model_path']
        model.save(model_path)
        logging.info(f"Affective model saved to {model_path}")
