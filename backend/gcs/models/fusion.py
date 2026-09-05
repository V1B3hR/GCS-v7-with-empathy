"""
Multimodal Fusion Layer

Combines embeddings from different modalities using:
- Concatenation with learned gating
- Attention-based fusion
- MC Dropout for uncertainty estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, Optional


class MultimodalFusion(keras.Model):
    """
    Fuses multimodal embeddings with attention mechanism
    
    Supports:
    - Missing modality handling via masks
    - Attention-based weighted fusion
    - MC Dropout for epistemic uncertainty
    """
    MODALITY_ORDER = ("eeg", "physio", "voice", "text")
    
    def __init__(self,
                 fusion_type: str = 'attention',  # 'concat' or 'attention'
                 attention_heads: int = 4,
                 hidden_dim: int = 512,
                 mc_dropout: bool = True,
                 mc_samples: int = 15,
                 dropout: float = 0.3,
                 **kwargs):
        """
        Args:
            fusion_type: 'concat' or 'attention'
            attention_heads: Number of attention heads
            hidden_dim: Hidden dimension for fusion
            mc_dropout: Whether to use MC Dropout for uncertainty
            mc_samples: Number of MC dropout samples
            dropout: Dropout rate
        """
        super(MultimodalFusion, self).__init__(**kwargs)

        if fusion_type not in {"attention", "concat"}:
            raise ValueError("fusion_type must be either 'attention' or 'concat'.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if fusion_type == 'attention':
            if attention_heads <= 0:
                raise ValueError("attention_heads must be positive for attention fusion.")
            if hidden_dim % attention_heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by attention_heads for attention fusion."
                )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if mc_dropout and mc_samples <= 0:
            raise ValueError("mc_samples must be positive when mc_dropout is enabled.")
        
        self.fusion_type = fusion_type
        self.attention_heads = attention_heads
        self.hidden_dim = hidden_dim
        self.mc_dropout = mc_dropout
        self.mc_samples = mc_samples
        self.dropout_rate = dropout
        
        self._build_layers()
        
        logging.info(f"Multimodal Fusion initialized: type={fusion_type}, heads={attention_heads}")
    
    def _build_layers(self):
        """Build fusion layers"""
        
        if self.fusion_type == 'attention':
            # Multi-head attention for fusion
            self.attention = layers.MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.hidden_dim // self.attention_heads,
                name='fusion_attention'
            )
            
            # Layer normalization
            self.layer_norm = layers.LayerNormalization(name='fusion_layer_norm')
        
        # Projection layers to common dimension
        self.projection_layers = {
            'eeg': layers.Dense(self.hidden_dim, name='proj_eeg'),
            'physio': layers.Dense(self.hidden_dim, name='proj_physio'),
            'voice': layers.Dense(self.hidden_dim, name='proj_voice'),
            'text': layers.Dense(self.hidden_dim, name='proj_text')
        }
        
        # Gating mechanism for modality weighting
        # Gate slots follow MODALITY_ORDER regardless of which modalities are present.
        self.max_modalities = len(self.MODALITY_ORDER)
        self.gate_dense = layers.Dense(self.max_modalities, activation='sigmoid', name='fusion_gate')
        
        # Dense layers after fusion
        self.fusion_dense1 = layers.Dense(self.hidden_dim, activation='relu', name='fusion_dense1')
        self.fusion_bn = layers.BatchNormalization(name='fusion_bn')
        self.fusion_dropout = layers.Dropout(self.dropout_rate, name='fusion_dropout')
        
        self.fusion_dense2 = layers.Dense(self.hidden_dim, activation='relu', name='fusion_dense2')
        self.fusion_dropout2 = layers.Dropout(self.dropout_rate, name='fusion_dropout2')

    def _validate_embedding(self, embedding: tf.Tensor, modality_name: str) -> tf.Tensor:
        """Ensure modality embeddings are rank-2 tensors."""
        embedding = tf.convert_to_tensor(embedding)
        if embedding.shape.rank != 2:
            raise ValueError(
                f"{modality_name} embedding must have shape (batch, embedding_dim); "
                f"received rank {embedding.shape.rank} with shape {embedding.shape}."
            )
        return embedding

    def _normalize_mask(self,
                        mask: tf.Tensor,
                        embedding: tf.Tensor,
                        modality_name: str) -> tf.Tensor:
        """Normalize per-modality masks to shape (batch, 1)."""
        mask = tf.convert_to_tensor(mask)
        rank = mask.shape.rank
        if rank not in (1, 2):
            raise ValueError(
                f"{modality_name} mask must have shape (batch,) or (batch, 1); "
                f"received shape {mask.shape}."
            )
        if rank == 2 and mask.shape[-1] not in (1, None):
            raise ValueError(
                f"{modality_name} mask must have shape (batch,) or (batch, 1); "
                f"received shape {mask.shape}."
            )

        if (
            embedding.shape[0] is not None
            and mask.shape[0] is not None
            and embedding.shape[0] != mask.shape[0]
        ):
            raise ValueError(
                f"{modality_name} mask batch dimension {mask.shape[0]} does not match "
                f"embedding batch dimension {embedding.shape[0]}."
            )

        mask = tf.cast(mask, embedding.dtype)
        return tf.reshape(mask, [-1, 1])

    def _prepare_modalities(self,
                            embeddings_dict: Dict[str, tf.Tensor],
                            masks_dict: Optional[Dict[str, tf.Tensor]] = None):
        """Project known modalities to a common width and align them to stable slots."""
        validated_embeddings = {}
        reference_embedding = None

        for modality_name in self.MODALITY_ORDER:
            embedding = embeddings_dict.get(modality_name)
            if embedding is None:
                continue
            embedding = self._validate_embedding(embedding, modality_name)
            if reference_embedding is None:
                reference_embedding = embedding
            elif (
                reference_embedding.shape[0] is not None
                and embedding.shape[0] is not None
                and reference_embedding.shape[0] != embedding.shape[0]
            ):
                raise ValueError(
                    f"All modality embeddings must share the same batch dimension; "
                    f"got {reference_embedding.shape[0]} and {embedding.shape[0]}."
                )
            validated_embeddings[modality_name] = embedding

        if reference_embedding is None:
            raise ValueError("No modalities available for fusion")

        batch_size = tf.shape(reference_embedding)[0]
        dtype = reference_embedding.dtype
        projected_embeddings = []
        modality_masks = []

        for modality_name in self.MODALITY_ORDER:
            embedding = validated_embeddings.get(modality_name)
            if embedding is None:
                projected_embeddings.append(
                    tf.zeros([batch_size, self.hidden_dim], dtype=dtype)
                )
                modality_masks.append(tf.zeros([batch_size, 1], dtype=dtype))
                continue

            projected = self.projection_layers[modality_name](embedding)
            raw_mask = masks_dict.get(modality_name) if masks_dict else None
            mask = (
                self._normalize_mask(raw_mask, projected, modality_name)
                if raw_mask is not None
                else tf.ones([batch_size, 1], dtype=projected.dtype)
            )

            projected_embeddings.append(projected)
            modality_masks.append(mask)

        stacked_embeddings = tf.stack(projected_embeddings, axis=1)
        stacked_masks = tf.stack(modality_masks, axis=1)
        return stacked_embeddings, stacked_masks

    def _call_internal(self,
                       embeddings_dict: Dict[str, tf.Tensor],
                       masks_dict: Optional[Dict[str, tf.Tensor]] = None,
                       training=None,
                       dropout_training=None):
        """Internal fusion call with explicit dropout control."""
        stacked_embeddings, stacked_masks = self._prepare_modalities(
            embeddings_dict,
            masks_dict,
        )
        masked_embeddings = stacked_embeddings * stacked_masks

        mask_counts = tf.reduce_sum(stacked_masks, axis=1)
        gate_inputs = tf.math.divide_no_nan(
            tf.reduce_sum(masked_embeddings, axis=1),
            mask_counts,
        )
        all_gates = self.gate_dense(gate_inputs)
        gate_tensor = tf.expand_dims(all_gates, axis=-1) * stacked_masks
        gated_embeddings = masked_embeddings * gate_tensor

        if self.fusion_type == 'attention':
            token_mask = tf.cast(tf.squeeze(stacked_masks, axis=-1), tf.bool)
            attention_mask = tf.logical_and(
                token_mask[:, :, tf.newaxis],
                token_mask[:, tf.newaxis, :],
            )
            attended = self.attention(
                query=gated_embeddings,
                value=gated_embeddings,
                key=gated_embeddings,
                attention_mask=attention_mask,
                training=training,
            )
            attended = self.layer_norm(attended + gated_embeddings)
            attended = attended * stacked_masks
            fused = tf.math.divide_no_nan(
                tf.reduce_sum(attended, axis=1),
                tf.reduce_sum(stacked_masks, axis=1),
            )
        else:
            # Concat mode always keeps MODALITY_ORDER slots, using zero vectors for
            # missing or masked modalities so the fused width is stable.
            fused = tf.reshape(
                gated_embeddings,
                [tf.shape(gated_embeddings)[0], self.max_modalities * self.hidden_dim],
            )

        x = self.fusion_dense1(fused)
        x = self.fusion_bn(x, training=training)
        x = self.fusion_dropout(x, training=dropout_training)
        x = self.fusion_dense2(x)
        x = self.fusion_dropout2(x, training=dropout_training)
        return x
    
    def call(self, embeddings_dict: Dict[str, tf.Tensor], 
             masks_dict: Optional[Dict[str, tf.Tensor]] = None,
             training=None):
        """
        Fuse multimodal embeddings
        
        Args:
            embeddings_dict: Dictionary with keys 'eeg', 'physio', 'voice', 'text'
                            Each value is (batch, embedding_dim)
            masks_dict: Optional dictionary of binary masks for each modality
            training: Whether in training mode
        
        Returns:
            Fused embedding (batch, hidden_dim)
        """
        return self._call_internal(
            embeddings_dict,
            masks_dict=masks_dict,
            training=training,
            dropout_training=training,
        )
    
    def call_with_uncertainty(self, 
                             embeddings_dict: Dict[str, tf.Tensor],
                             masks_dict: Optional[Dict[str, tf.Tensor]] = None):
        """
        Forward pass with MC Dropout for uncertainty estimation
        
        Returns:
            mean_output, std_output (both batch, hidden_dim)
        """
        if not self.mc_dropout:
            output = self.call(embeddings_dict, masks_dict, training=False)
            return output, tf.zeros_like(output)
        
        # Multiple forward passes with dropout enabled while keeping normalization
        # layers in inference mode so moving statistics are not updated.
        outputs = []
        for _ in range(self.mc_samples):
            output = self._call_internal(
                embeddings_dict,
                masks_dict=masks_dict,
                training=False,
                dropout_training=True,
            )
            outputs.append(output)
        
        # Stack and compute statistics
        outputs_stacked = tf.stack(outputs, axis=0)  # (mc_samples, batch, hidden_dim)
        mean_output = tf.reduce_mean(outputs_stacked, axis=0)
        std_output = tf.math.reduce_std(outputs_stacked, axis=0)
        
        return mean_output, std_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'fusion_type': self.fusion_type,
            'attention_heads': self.attention_heads,
            'hidden_dim': self.hidden_dim,
            'mc_dropout': self.mc_dropout,
            'mc_samples': self.mc_samples,
            'dropout': self.dropout_rate,
        })
        return config


def create_fusion_layer(config: Dict) -> MultimodalFusion:
    """Create fusion layer from config"""
    fusion_config = config.get('model', {}).get('fusion', {})
    
    return MultimodalFusion(
        fusion_type=fusion_config.get('type', 'attention'),
        attention_heads=fusion_config.get('attention_heads', 4),
        hidden_dim=fusion_config.get('hidden_dim', 512),
        mc_dropout=fusion_config.get('mc_dropout', True),
        mc_samples=fusion_config.get('mc_samples', 15),
        dropout=fusion_config.get('dropout', 0.3)
    )
