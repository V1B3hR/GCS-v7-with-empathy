# Real Data Training Implementation

This document describes the implementation of training on real data for the GCS (Graph-based Cognitive State) model.

## Overview

The system now supports loading real EEG and affective data from files instead of using simulated data. The implementation includes:

1. **Real data loading** from various file formats
2. **Feature extraction** for LOSO cross-validation
3. **Complete training pipelines** for both foundational and affective models
4. **Graceful fallback** to simulated data when real data is insufficient

## Data Format Requirements

### EEG Data for LOSO Cross-Validation

Place EEG data files in the directory specified by `real_data_paths.eeg_data_dir` in `config.yaml`:

```yaml
real_data_paths:
  eeg_data_dir: "data/real_eeg"
```

**Supported file formats:**
- `.npz` files (preferred)
- `.npy` files

**Expected data structure for `.npz` files:**
```python
# Required arrays (system will auto-detect key names):
{
    'eeg': np.ndarray,      # Shape: (trials, timesteps, channels) 
                            # Alternative keys: 'data', 'signals'
    'labels': np.ndarray,   # Shape: (trials,) - integer class labels
                            # Alternative keys: 'targets', 'y'
    
    # Optional metadata:
    'subject_id': int,
    'sampling_rate': float,
    'channels': list        # Channel names
}
```

**Example data shapes:**
- EEG: `(40, 250, 64)` - 40 trials, 250 timesteps, 64 channels
- Labels: `(40,)` - 40 integer labels (e.g., 0=left hand, 1=right hand)

### Affective Data

Configure the affective dataset path in `config.yaml`:

```yaml
affective_model:
  deap_dataset_path: "data/deap_dataset.npz"
```

**Expected data structure:**
```python
{
    'eeg': np.ndarray,      # Shape: (samples, timesteps, channels)
                            # Alternative keys: 'signals', 'data'
    'physio': np.ndarray,   # Shape: (samples, physio_features) 
                            # Alternative keys: 'physiological', 'bio'
    'voice': np.ndarray,    # Shape: (samples, voice_features)
                            # Alternative keys: 'audio', 'speech'
    'valence': np.ndarray,  # Shape: (samples,) - valence ratings
                            # Alternative keys: 'val'
    'arousal': np.ndarray,  # Shape: (samples,) - arousal ratings
                            # Alternative keys: 'aro'
}
```

**Example data shapes:**
- EEG: `(200, 250, 64)` - 200 samples, 250 timesteps, 64 channels
- Physio: `(200, 2)` - 200 samples, 2 features (e.g., HRV, GSR)
- Voice: `(200, 128)` - 200 samples, 128 voice features
- Valence/Arousal: `(200,)` - 200 rating values (typically 1-9 scale)

## Usage

### Training with Real Data

1. **Prepare your data** in the required format and place in the configured directories
2. **Run LOSO cross-validation:**
   ```python
   from gcs.training import Trainer
   from gcs.config_loader import load_config
   
   config = load_config('config.yaml')
   trainer = Trainer(config)
   trainer.run_loso_cross_validation()
   ```

3. **Train affective model:**
   ```python
   trainer.train_affective_model()
   ```

### Fallback Behavior

The system gracefully handles missing or insufficient real data:

- **LOSO**: If fewer than `train_subjects` real subjects are found, remaining subjects are simulated
- **Affective**: If fewer than 1000 samples are found, additional simulated samples are generated
- **Missing files**: Logs warnings and continues with available data

### Logging

The system provides detailed logging about data loading:

```
INFO: Loading real EEG data from data/real_eeg
INFO: Loaded subject 1 data from subject_01.npz: shape (40, 68, 250)
WARNING: Only loaded 3 real subjects, supplementing with 7 simulated subjects
INFO: LOSO dataset ready: 10 subjects
```

## Data Preprocessing

1. **Source Localization**: Raw EEG is transformed to cortical source space using simulated eLORETA
2. **Shape Normalization**: Ensures consistent data shapes across subjects/samples
3. **Label Handling**: Automatically detects and uses available labels, generates defaults if missing

## Configuration

Key configuration parameters in `config.yaml`:

```yaml
# Core model parameters
cortical_nodes: 68          # Number of cortical nodes
timesteps: 250              # EEG time samples
eeg_channels: 64            # Number of EEG channels
train_subjects: 10          # Number of subjects for LOSO

# Training parameters
batch_size: 32
epochs: 10
output_model_dir: "models"

# Real data paths
real_data_paths:
  eeg_data_dir: "data/real_eeg"

affective_model:
  deap_dataset_path: "data/deap_dataset.npz"
```

## Example File Structure

```
data/
├── graph_scaffold.npz          # Cortical graph structure
├── deap_dataset.npz            # Affective dataset
└── real_eeg/                   # EEG datasets directory
    ├── subject_01.npz
    ├── subject_02.npz
    └── ...
```

This implementation enables the GCS model to train on real neuroscience data while maintaining compatibility with the existing architecture and providing robust error handling.