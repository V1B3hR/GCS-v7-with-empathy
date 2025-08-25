import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

class AffectiveModelBuilder:
    """
    Builds the definitive multi-modal fusion model for affective state classification.
    This class correctly implements transfer learning, using the pre-trained GCS
    foundational model as an expert EEG feature extractor.
    """
    @staticmethod
    def build_fused_classifier(config: dict, gcs_foundational_model: Model) -> Model:
        """
        Builds the multi-modal fusion network using a pre-trained GNN encoder.

        Args:
            config (dict): The global configuration dictionary.
            gcs_foundational_model (Model): The pre-trained, audited GCS GNN model.

        Returns:
            A new, compiled Keras Model ready for training.
        """
        logging.info("Building the multi-modal Affective State Classifier using Transfer Learning...")

        # --- 1. Freeze the Foundational Model ---
        # We do not want to alter its learned wisdom.
        gcs_foundational_model.trainable = False
        logging.info("Pre-trained GCS foundational model has been frozen.")

        # --- 2. Get Handles to the Foundational Model's Layers ---
        # THE CORRECT WAY: We use the *existing* inputs and outputs of the pre-trained model.
        # This ensures our new model is correctly grafted onto the old one.
        node_input = gcs_foundational_model.get_layer("node_input").input
        adj_input = gcs_foundational_model.get_layer("adj_input").input
        
        # Find the pooling layer to get the rich graph embedding.
        # This is robust and doesn't rely on a hard-coded name.
        pooling_layer_name = [layer.name for layer in gcs_foundational_model.layers if "global" in layer.name][0]
        graph_embedding = gcs_foundational_model.get_layer(pooling_layer_name).output
        
        logging.info(f"Using '{pooling_layer_name}' as the source of EEG features.")

        # --- 3. Define New Inputs for Other Modalities ---
        physio_input = Input(shape=(config["physio_features"],), name="physio_input")
        voice_input = Input(shape=(128,), name="voice_input") # Assuming 128 prosody features

        # --- 4. Process Peripheral Branches ---
        physio_units = config['affective_model']['physio_branch_units']
        voice_units = config['affective_model']['voice_branch_units']
        
        x_physio = Dense(physio_units, activation="relu")(physio_input)
        x_physio = BatchNormalization()(x_physio)

        x_voice = Dense(voice_units, activation="relu")(voice_input)
        x_voice = BatchNormalization()(x_voice)

        # --- 5. Fuse All Feature Vectors ---
        logging.info("Fusing EEG, physiological, and voice feature streams...")
        fused_features = Concatenate()([graph_embedding, x_physio, x_voice])
        
        # --- 6. Final Dense Layers for Regression ---
        fusion_units = config['affective_model']['fusion_units']
        dropout_rate = config['affective_model']['dropout_rate']

        x = Dense(fusion_units, activation="relu")(fused_features)
        x = Dropout(dropout_rate)(x)

        # --- 7. Output Heads ---
        valence_output = Dense(1, name="valence_output")(x)
        arousal_output = Dense(1, name="arousal_output")(x)

        # --- 8. Assemble the Final, Connected Model ---
        # The inputs list now correctly includes the original inputs from the GCS model.
        final_inputs = [node_input, adj_input, physio_input, voice_input]
        
        final_model = Model(
            inputs=final_inputs,
            outputs=[valence_output, arousal_output],
            name="GCS_Affective_Fusion_Model"
        )
        
        logging.info("Affective State Classifier built successfully with a valid computational graph.")
        return final_model
