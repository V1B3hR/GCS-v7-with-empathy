import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from gcs.data_pipeline import DataPipeline
from gcs.models.affective_model import AffectiveModel
from gcs.models.fusion import MultimodalFusion


def _minimal_model_config(**overrides):
    config = {
        "cortical_nodes": 4,
        "timesteps": 8,
        "physio_features": 24,
        "voice_features": 128,
        "text_features": 768,
        "modalities": {
            "eeg": True,
            "physio": True,
            "voice": True,
            "text": False,
        },
        "model": {
            "eeg_encoder": {"return_sequence": False},
            "fusion": {
                "type": "attention",
                "attention_heads": 4,
                "hidden_dim": 32,
                "mc_dropout": True,
                "mc_samples": 4,
                "dropout": 0.1,
            },
            "heads": {},
        },
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("fusion_type", ["attention", "concat"])
def test_supported_encoder_widths_flow_through_fusion(fusion_type):
    fusion = MultimodalFusion(
        fusion_type=fusion_type,
        attention_heads=4,
        hidden_dim=32,
        dropout=0.0,
    )

    embeddings = {
        "eeg": tf.random.normal((3, 256)),
        "physio": tf.random.normal((3, 128)),
        "voice": tf.random.normal((3, 256)),
        "text": tf.random.normal((3, 256)),
    }

    output = fusion(embeddings, training=False)

    assert output.shape == (3, 32)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fusion_type": "invalid"}, "fusion_type"),
        ({"hidden_dim": 0}, "hidden_dim"),
        (
            {"fusion_type": "attention", "attention_heads": 0},
            "attention_heads",
        ),
        (
            {"fusion_type": "attention", "hidden_dim": 10, "attention_heads": 3},
            "divisible",
        ),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"mc_dropout": True, "mc_samples": 0}, "mc_samples"),
    ],
)
def test_invalid_fusion_configuration_raises_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultimodalFusion(**kwargs)


def test_fusion_rejects_rank_three_eeg_embeddings():
    fusion = MultimodalFusion(hidden_dim=16, attention_heads=4, dropout=0.0)

    with pytest.raises(ValueError, match="eeg embedding must have shape"):
        fusion(
            {
                "eeg": tf.random.normal((2, 5, 256)),
                "physio": tf.random.normal((2, 128)),
            },
            training=False,
        )


def test_affective_model_rejects_eeg_return_sequence_when_enabled():
    config = _minimal_model_config()
    config["model"]["eeg_encoder"]["return_sequence"] = True

    with pytest.raises(ValueError, match="return_sequence=True is incompatible"):
        AffectiveModel(config)


def test_gating_uses_stable_modality_slots():
    fusion = MultimodalFusion(
        fusion_type="concat",
        hidden_dim=1,
        attention_heads=1,
        dropout=0.0,
        mc_dropout=False,
    )

    build_embeddings = {
        "eeg": tf.ones((1, 1)),
        "physio": tf.ones((1, 1)),
        "voice": tf.ones((1, 1)),
        "text": tf.ones((1, 1)),
    }
    fusion(build_embeddings, training=False)

    for layer in fusion.projection_layers.values():
        layer.set_weights([np.ones((1, 1), dtype=np.float32), np.zeros((1,), dtype=np.float32)])

    gate_bias = np.array(
        [
            math.log(0.1 / 0.9),
            math.log(0.2 / 0.8),
            math.log(0.4 / 0.6),
            math.log(0.8 / 0.2),
        ],
        dtype=np.float32,
    )
    fusion.gate_dense.set_weights(
        [
            np.zeros((1, 4), dtype=np.float32),
            gate_bias,
        ]
    )
    fusion.fusion_dense1.set_weights(
        [
            np.ones((4, 1), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
        ]
    )
    fusion.fusion_bn.set_weights(
        [
            np.array([math.sqrt(1.0 + fusion.fusion_bn.epsilon)], dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
        ]
    )
    fusion.fusion_dense2.set_weights(
        [
            np.ones((1, 1), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
        ]
    )

    physio_only = fusion({"physio": tf.ones((1, 1))}, training=False)
    voice_and_text = fusion(
        {
            "voice": tf.ones((1, 1)),
            "text": tf.ones((1, 1)),
        },
        training=False,
    )

    np.testing.assert_allclose(physio_only.numpy(), [[0.2]], atol=1e-5)
    np.testing.assert_allclose(voice_and_text.numpy(), [[1.2]], atol=1e-5)


@pytest.mark.parametrize(
    "voice_mask",
    [
        tf.constant([0.0, 1.0], dtype=tf.float32),
        tf.constant([[0.0], [1.0]], dtype=tf.float32),
    ],
)
def test_masks_prevent_masked_modalities_from_influencing_output(voice_mask):
    tf.random.set_seed(1234)
    fusion = MultimodalFusion(
        fusion_type="attention",
        hidden_dim=16,
        attention_heads=4,
        dropout=0.0,
        mc_dropout=False,
    )

    shared_embeddings = {
        "eeg": tf.random.normal((2, 256)),
        "physio": tf.random.normal((2, 128)),
        "voice": tf.constant(
            [[1.0] * 256, [2.0] * 256],
            dtype=tf.float32,
        ),
    }
    changed_voice = dict(shared_embeddings)
    changed_voice["voice"] = tf.constant(
        [[500.0] * 256, [900.0] * 256],
        dtype=tf.float32,
    )

    masks = {"voice": voice_mask}

    baseline = fusion(shared_embeddings, masks_dict=masks, training=False).numpy()
    changed = fusion(changed_voice, masks_dict=masks, training=False).numpy()

    np.testing.assert_allclose(baseline[0], changed[0], atol=1e-5)
    assert not np.allclose(baseline[1], changed[1], atol=1e-5)


def test_invalid_mask_shape_raises_value_error():
    fusion = MultimodalFusion(hidden_dim=16, attention_heads=4, dropout=0.0)

    with pytest.raises(ValueError, match="voice mask must have shape"):
        fusion(
            {"voice": tf.random.normal((2, 256))},
            masks_dict={"voice": tf.ones((2, 2))},
            training=False,
        )


def test_mismatched_modality_batch_sizes_raise_value_error():
    fusion = MultimodalFusion(hidden_dim=16, attention_heads=4, dropout=0.0)

    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="batch dimension"):
        fusion(
            {
                "eeg": tf.random.normal((2, 256)),
                "physio": tf.random.normal((3, 128)),
            },
            training=False,
        )


def test_unknown_modality_keys_raise_value_error():
    fusion = MultimodalFusion(hidden_dim=16, attention_heads=4, dropout=0.0)

    with pytest.raises(ValueError, match="Unsupported modalities for fusion"):
        fusion({"image": tf.random.normal((2, 64))}, training=False)


def test_mc_uncertainty_does_not_update_batchnorm_statistics():
    tf.random.set_seed(7)
    fusion = MultimodalFusion(
        hidden_dim=16,
        attention_heads=4,
        dropout=0.5,
        mc_dropout=True,
        mc_samples=5,
    )
    embeddings = {
        "eeg": tf.random.normal((4, 256)),
        "physio": tf.random.normal((4, 128)),
        "voice": tf.random.normal((4, 256)),
    }

    fusion(embeddings, training=False)
    mean_before = fusion.fusion_bn.moving_mean.numpy().copy()
    variance_before = fusion.fusion_bn.moving_variance.numpy().copy()

    _, std_output = fusion.call_with_uncertainty(embeddings)

    np.testing.assert_allclose(fusion.fusion_bn.moving_mean.numpy(), mean_before)
    np.testing.assert_allclose(fusion.fusion_bn.moving_variance.numpy(), variance_before)
    assert np.any(std_output.numpy() > 0.0)


def test_simulated_physio_defaults_to_configured_width():
    pipeline = DataPipeline(
        {
            "allow_simulated_data": True,
            "timesteps": 6,
            "eeg_channels": 3,
            "cortical_nodes": 4,
        }
    )

    inputs, _ = pipeline.load_affective_data([])

    assert inputs["physio"].shape == (1000, 24)


def test_missing_real_physio_fallback_uses_configured_width(tmp_path):
    dataset_path = tmp_path / "eeg_only.npz"
    np.savez(
        dataset_path,
        eeg=np.random.randn(1000, 5, 3).astype(np.float32),
        valence=np.random.uniform(1, 9, 1000).astype(np.float32),
        arousal=np.random.uniform(1, 9, 1000).astype(np.float32),
    )

    pipeline = DataPipeline(
        {
            "allow_simulated_data": True,
            "timesteps": 5,
            "eeg_channels": 3,
            "cortical_nodes": 4,
            "physio_features": 7,
        }
    )

    inputs, _ = pipeline.load_affective_data([str(dataset_path)])

    assert inputs["physio"].shape == (1000, 7)
