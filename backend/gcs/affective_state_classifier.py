import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

class AffectiveModelBuilder:
    """
    Builds the definitive multi-modal fusion model for affective state classification.
    This class leverages the power of the pre-trained GCS foundational model
    as an expert EEG feature extractor (Transfer Learning).
    """
    @staticmethod
    def build_fused_classifier(config: dict, gcs_foundational_model: Model) -> Model:
        """
        Builds the multi-modal fusion network.

        Args:
            config (dict): The global configuration dictionary.
            gcs_foundational_model (Model): The pre-trained, audited GCS GNN model.

        Returns:
            A new, compiled Keras Model ready for training.
        """
        logging.info("Building the multi-modal Affective State Classifier...")

        # --- 1. Create new inputs matching the foundational model ---
        node_input = Input(shape=(config["cortical_nodes"], config["timesteps"]), name="node_input")
        
        # CRITICAL: Freeze the foundational model. We don't want to retrain it.
        gcs_foundational_model.trainable = False
        logging.info("Pre-trained GCS foundational model has been frozen for transfer learning.")

        # Get graph embedding by calling the foundational model on our input
        graph_embedding = gcs_foundational_model(node_input)
        # The foundational model returns [mi_output, adversary_output], we want the features before classification
        # Let's use the output from before the final classification layer
        feature_extractor = Model(inputs=gcs_foundational_model.input, 
                                outputs=gcs_foundational_model.get_layer("flatten").output)
        feature_extractor.trainable = False
        graph_embedding = feature_extractor(node_input)

        # --- 2. Define Other Modality Inputs ---
        physio_input = Input(shape=(config["physio_features"],), name="physio_input")
        voice_input = Input(shape=(128,), name="voice_input") # Assuming 128 prosody features

        # --- 3. Process Peripheral Branches ---
        # VIBE CODING: Use config for flexibility
        physio_units = config['affective_model']['physio_branch_units']
        voice_units = config['affective_model']['voice_branch_units']
        
        x_physio = Dense(physio_units, activation="relu")(physio_input)
        x_physio = BatchNormalization()(x_physio)

        x_voice = Dense(voice_units, activation="relu")(voice_input)
        x_voice = BatchNormalization()(x_voice)

        # --- 4. Fuse All Feature Vectors ---
        logging.info("Fusing EEG, physiological, and voice feature streams...")
        fused_features = Concatenate()([graph_embedding, x_physio, x_voice])
        
        # --- 5. Final Dense Layers for Regression ---
        fusion_units = config['affective_model']['fusion_units']
        dropout_rate = config['affective_model']['dropout_rate']

        x = Dense(fusion_units, activation="relu")(fused_features)
        x = Dropout(dropout_rate)(x)

        # --- 6. Output Heads ---
        valence_output = Dense(1, name="valence_output")(x)
        arousal_output = Dense(1, name="arousal_output")(x)

        # --- 7. Assemble the Final Model ---
        # The inputs now include all our inputs
        final_inputs = [node_input, physio_input, voice_input]
        
        final_model = Model(
            inputs=final_inputs,
            outputs=[valence_output, arousal_output],
            name="GCS_Affective_Fusion_Model"
        )
        
        logging.info("Affective State Classifier built successfully.")
        return final_model
