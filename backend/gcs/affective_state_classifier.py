FILE: backend/gcs/affective_state_classifier.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model
class AffectiveStateFactory:
@staticmethod
def build_classifier(config: dict) -> Model:
eeg_input = Input(shape=(config['cortical_nodes'], config['timesteps']), name="eeg_input")
physio_input = Input(shape=(2,), name="physio_input")
voice_input = Input(shape=(128,), name="voice_input")
    eeg_features = Flatten()(eeg_input)
    eeg_features = Dense(64, activation='relu')(eeg_features)

    physio_features = Dense(config['affective_model']['physio_branch_units'], activation='relu')(physio_input)
    physio_features = BatchNormalization()(physio_features)

    voice_features = Dense(config['affective_model']['voice_branch_units'], activation='relu')(voice_input)
    voice_features = BatchNormalization()(voice_features)

    fused_features = Concatenate()([eeg_features, physio_features, voice_features])
    fused_features = Dense(config['affective_model']['fusion_units'], activation='relu')(fused_features)
    fused_features = Dropout(0.5)(fused_features)

    valence_output = Dense(1, name="valence_output")(fused_features)
    arousal_output = Dense(1, name="arousal_output")(fused_features)

    return Model(inputs=[eeg_input, physio_input, voice_input], outputs=[valence_output, arousal_output])
