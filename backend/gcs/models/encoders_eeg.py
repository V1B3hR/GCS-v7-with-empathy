"""
EEG Encoder Architecture

Temporal CNN (or TCN) encoder for EEG signals
Optional graph neural network component if cortical graph available
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Optional, Dict, Any


class EEGEncoder(keras.Model):
    """
    EEG encoder using 1D temporal CNN
    
    Architecture:
    - Multiple 1D conv layers with increasing filters
    - Batch normalization and dropout
    - Optional: Graph convolutional layers if spatial structure provided
    - Outputs: Learned embedding vector
    """
    
    def __init__(self,
                 input_shape: tuple,  # (channels/nodes, timesteps)
                 temporal_filters: list = [64, 128, 128],
                 kernel_sizes: list = [7, 5, 3],
                 embedding_dim: int = 256,
                 dropout: float = 0.3,
                 use_graph: bool = False,
                 **kwargs):
        """
        Args:
            input_shape: Shape of EEG input (channels, timesteps)
            temporal_filters: List of filter sizes for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            embedding_dim: Dimension of output embedding
            dropout: Dropout rate
            use_graph: Whether to use graph structure (not fully implemented yet)
        """
        super(EEGEncoder, self).__init__(**kwargs)
        
        self.input_shape_eeg = input_shape
        self.temporal_filters = temporal_filters
        self.kernel_sizes = kernel_sizes
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.use_graph = use_graph
        
        # Build layers
        self._build_layers()
        
        logging.info(f"EEG Encoder initialized with embedding_dim={embedding_dim}")
    
        self.conv_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        
# Temporal convolutional layers
for i, (filters, kernel_size) in enumerate(zip(self.temporal_filters, self.kernel_sizes)):
    # Conv layer
    conv = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        name=f'eeg_conv_{i}'
    )
    self.conv_layers.append(conv)
            
    # Batch normalization
    bn = layers.BatchNormalization(name=f'eeg_bn_{i}')
    self.bn_layers.append(bn)
            
    # Dropout
    dropout = layers.Dropout(self.dropout_rate, name=f'eeg_dropout_{i}')
    self.dropout_layers.append(dropout)
            
    # Max pooling
    if i < len(self.temporal_filters) - 1:
        pool = layers.MaxPooling1D(pool_size=2, name=f'eeg_pool_{i}')
        self.conv_layers.append(pool)
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D(name='eeg_global_pool')
        
        # Dense layers to embedding
        self.dense1 = layers.Dense(512, activation='relu', name='eeg_dense1')
        self.bn_dense = layers.BatchNormalization(name='eeg_bn_dense')
        self.dropout_dense = layers.Dropout(self.dropout_rate, name='eeg_dropout_dense')
        
        self.embedding_layer = layers.Dense(
            self.embedding_dim,
            activation=None,  # Linear activation for embedding
            name='eeg_embedding'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: EEG tensor of shape (batch, channels, timesteps)
            training: Whether in training mode
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        x = inputs
        
        # Transpose to (batch, timesteps, channels) for Conv1D
        x = tf.transpose(x, [0, 2, 1])
        
       for i in range(len(self.temporal_filters)):
    x = self.conv_layers[i * 2](x, training=training)  # Conv
    x = self.bn_layers[i](x, training=training)
    x = self.dropout_layers[i](x, training=training)
            
    # Apply pooling if not last layer
    if i < len(self.temporal_filters) - 1:
        x = self.conv_layers[i * 2 + 1](x, training=training)  # Pool
        
        # Global pooling
        x = self.global_pool(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.bn_dense(x, training=training)
        x = self.dropout_dense(x, training=training)
        
        # Final embedding
        embedding = self.embedding_layer(x)
        
        return embedding
    
    def get_config(self):
        """Get configuration for serialization"""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_eeg,
            'temporal_filters': self.temporal_filters,
            'kernel_sizes': self.kernel_sizes,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout_rate,
            'use_graph': self.use_graph,
        })
        return config


def create_eeg_encoder(config: Dict[str, Any]) -> EEGEncoder:
    """
    Factory function to create EEG encoder from config
    
    Args:
        config: Configuration dictionary with 'model' section
    
    Returns:
        EEGEncoder instance
    """
    eeg_config = config.get('model', {}).get('eeg_encoder', {})
    
    # Determine input shape
    cortical_nodes = config.get('cortical_nodes', 68)
    timesteps = config.get('timesteps', 250) if 'timesteps' in config else 1000
    input_shape = (cortical_nodes, timesteps)
    
    encoder = EEGEncoder(
        input_shape=input_shape,
        temporal_filters=eeg_config.get('temporal_filters', [64, 128, 128]),
        kernel_sizes=eeg_config.get('kernel_sizes', [7, 5, 3]),
        embedding_dim=eeg_config.get('embedding_dim', 256),
        dropout=eeg_config.get('dropout', 0.3),
        use_graph=eeg_config.get('use_graph', False)
    )
    
    return encoder
