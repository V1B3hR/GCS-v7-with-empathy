import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# CONCEPTUAL: spektral is a library for Graph Neural Networks in Keras.
from spektral.layers import GATConv, GlobalAvgPool

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy): return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def call(self, x): return grad_reverse(x)

class FoundationalModelFactory:
    @staticmethod
    def build_mechanistic_model(config):
        node_input = Input(shape=(config["cortical_nodes"], config["timesteps"]))
        adj_input = Input(shape=(config["cortical_nodes"], config["cortical_nodes"]), sparse=True)
        x = GATConv(config["gnn_channels"], attn_heads=config["gat_heads"], activation='relu')([node_input, adj_input])
        x, attention_weights = GATConv(config["gnn_channels"], attn_heads=1, return_attn_coeffs=True)([x, adj_input])
        shared_embedding = GlobalAvgPool()(x)
        mi_output = Dense(2, activation='softmax', name="mi_output")(shared_embedding)
        reversed_embedding = GradReverse()(shared_embedding)
        adversary_output = Dense(config["train_subjects"], activation='softmax', name="adversary_output")(reversed_embedding)
        return Model(inputs=[node_input, adj_input], outputs=[mi_output, adversary_output, attention_weights])
