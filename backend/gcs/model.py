import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, LayerNormalization,
    TimeDistributed, Conv1D, LSTM, Flatten, Concatenate
)
from tensorflow.keras.models import Model
from spektral.layers import (
    GATConv, GCNConv, GraphSageConv,
    GlobalAvgPool, GlobalMaxPool, GlobalSumPool, GlobalAttnSumPool
)

# -------- Gradient Reversal Layer --------
@tf.custom_gradient
def grad_reverse(x):
    """A layer that reverses the gradient during backpropagation for adversarial training."""
    y = tf.identity(x)
    def custom_grad(dy): return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def call(self, x): return grad_reverse(x)

# -------- Utility Blocks (VIBE CODING: Abstracted for clarity) --------
def apply_norm_and_dropout(x, config):
    """Applies normalization and dropout layers as specified in the config."""
    if config.get("batch_norm", False):
        x = BatchNormalization()(x)
    if config.get("layer_norm", False):
        x = LayerNormalization()(x)
    if config.get("dropout_rate", 0.0) > 0:
        x = Dropout(config["dropout_rate"])(x)
    return x

def get_graph_layer_fn(config):
    """Returns a function that creates a GNN layer based on the config."""
    gnn_type = config.get("gnn_type", "gat")
    gnn_channels = config.get("gnn_channels", 32)
    kernel_initializer = config.get("kernel_initializer", "glorot_uniform")
    kernel_regularizer = tf.keras.regularizers.l2(config.get("l2_reg", 1e-4))

    if gnn_type == "gat":
        def layer_builder(attn_heads=1, concat_heads=False, return_attn=False):
            return GATConv(
                gnn_channels, attn_heads=attn_heads, concat_heads=concat_heads, activation="relu",
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                return_attn_coeffs=return_attn
            )
        return layer_builder
    # ... (GCN and Sage implementations would follow a similar pattern)
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}")

# -------- Model Factory --------
class GCSModelFactory:
    @staticmethod
    def build_affective_model(config: dict) -> Model:
        """
        Builds the definitive, multi-modal, and neuro-symbolic GCS model.
        This is a highly configurable and robust architecture for affective state classification.
        """
        # --- Inputs ---
        node_input = Input(shape=(config["cortical_nodes"], config["timesteps"]), name="node_input")
        adj_input = Input(shape=(config["cortical_nodes"], config["cortical_nodes"]), sparse=True, name="adj_input")
        inputs = [node_input, adj_input]
        if config.get("use_physio_input", False):
            physio_input = Input(shape=(config["physio_features"],), name="physio_input")
            inputs.append(physio_input)

        # --- Temporal Encoder ---
        x_eeg = node_input
        temporal_encoder = config.get("temporal_encoder")
        if temporal_encoder == "lstm":
            x_eeg = TimeDistributed(LSTM(config["temporal_features"], return_sequences=False))(x_eeg)
        elif temporal_encoder == "conv1d":
            x_eeg = TimeDistributed(Conv1D(config["temporal_features"], 3, padding="same", activation="relu"))(x_eeg)
            x_eeg = TimeDistributed(Flatten())(x_eeg)

        # --- Graph Layers ---
        graph_layer_builder = get_graph_layer_fn(config)
        x_graph = x_eeg
        attn_weights = []
        for i in range(config["num_layers"]):
            if config["gnn_type"] == "gat":
                is_first_layer = (i == 0)
                # BEST PRACTICE FIX: Use multiple heads on first layer, average on subsequent layers
                x_graph, attn = graph_layer_builder(
                    attn_heads=config["gat_heads"] if is_first_layer else 1,
                    concat_heads=is_first_layer,
                    return_attn=True
                )([x_graph, adj_input])
                attn_weights.append(tf.identity(attn, name=f"attention_layer_{i}"))
            else:
                x_graph = graph_layer_builder()([x_graph, adj_input])
            x_graph = apply_norm_and_dropout(x_graph, config)

        # --- Pooling ---
        pooling_map = {"avg": GlobalAvgPool(), "max": GlobalMaxPool(), "sum": GlobalSumPool(), "attention": GlobalAttnSumPool()}
        graph_embedding = pooling_map[config["pooling"]](x_graph)

        # --- Fusion with Physiological Input ---
        fused_embedding = graph_embedding
        if config.get("use_physio_input", False):
            x_physio = Dense(32, activation="relu")(physio_input)
            x_physio = apply_norm_and_dropout(x_physio, config)
            fused_embedding = Concatenate()([graph_embedding, x_physio])

        # --- Emotion Output Head ---
        x_main = fused_embedding
        for units in config.get("emotion_dense_layers", [128, 64]):
            x_main = Dense(units, activation="relu")(x_main)
            x_main = apply_norm_and_dropout(x_main, config)
        emotion_output = Dense(config["output_classes"], activation="softmax", name="emotion_output")(x_main)
        outputs = [emotion_output]

        # --- Adversarial Branch ---
        if config.get("use_adversary", False):
            # RECOMMENDATION: Use the richer 'fused_embedding' to make the adversary's job easier
            # and force the encoder to learn a more robust representation.
            adversary_input = GradReverse()(fused_embedding)
            x_adv = Dense(128, activation="relu")(adversary_input)
            x_adv = apply_norm_and_dropout(x_adv, config)
            adversary_output = Dense(config["train_subjects"], activation="softmax", name="adversary_output")(x_adv)
            outputs.append(adversary_output)

        # --- Attention Outputs ---
        if config.get("use_attention_output", False) and attn_weights:
            if config.get("expose_all_attn", False):
                outputs.extend(attn_weights)
            else:
                outputs.append(attn_weights[-1])

        return Model(inputs=inputs, outputs=outputs, name="GCS_Affective_Model_v2")

