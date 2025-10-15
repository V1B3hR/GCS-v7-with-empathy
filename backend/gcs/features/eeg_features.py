"""
EEG Feature Extraction

Extracts emotional markers from EEG signals:
- Band power features (delta, theta, alpha, beta, gamma)
- Asymmetry indices (frontal alpha asymmetry, etc.)
- Spectral entropy
- Optional connectivity features (coherence, PLV)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple
import logging


class EEGFeatureExtractor:
    """
    Extract emotion-relevant features from EEG signals
    
    Input: EEG data as (channels, timesteps) or (nodes, timesteps) if source-localized
    Output: Feature vector
    """
    
    # Standard EEG frequency bands (Hz)
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    def __init__(self, 
                 sampling_rate: int = 250,
                 extract_asymmetry: bool = True,
                 extract_connectivity: bool = False,
                 channel_pairs_for_asymmetry: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            sampling_rate: EEG sampling rate in Hz
            extract_asymmetry: Whether to compute asymmetry features
            extract_connectivity: Whether to compute connectivity features (expensive)
            channel_pairs_for_asymmetry: List of (left, right) channel pairs for asymmetry
                                         If None, uses default frontal pairs
        """
        self.sampling_rate = sampling_rate
        self.extract_asymmetry = extract_asymmetry
        self.extract_connectivity = extract_connectivity
        
        # Default to frontal alpha asymmetry (channels 0-1 as proxy)
        if channel_pairs_for_asymmetry is None:
            self.asymmetry_pairs = [(0, 1), (2, 3)]  # Simple left-right pairs
        else:
            self.asymmetry_pairs = channel_pairs_for_asymmetry
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive EEG features
        
        Args:
            eeg_data: EEG signal (channels, timesteps) or (nodes, timesteps)
        
        Returns:
            Feature vector as 1D array
        """
        features = []
        
        # 1. Band power features
        band_powers = self.extract_band_powers(eeg_data)
        features.append(band_powers.flatten())
        
        # 2. Asymmetry features
        if self.extract_asymmetry:
            asymmetry_features = self.extract_asymmetry_features(eeg_data)
            features.append(asymmetry_features)
        
        # 3. Spectral entropy
        spectral_entropy = self.extract_spectral_entropy(eeg_data)
        features.append(spectral_entropy)
        
        # 4. Statistical features
        stats = self.extract_statistical_features(eeg_data)
        features.append(stats)
        
        # 5. Connectivity features (optional, expensive)
        if self.extract_connectivity:
            connectivity = self.extract_connectivity_features(eeg_data)
            features.append(connectivity)
        
        # Concatenate all features
        return np.concatenate(features)
    
    def extract_band_powers(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Compute power in each frequency band for each channel
        
        Returns:
            Array of shape (n_channels, n_bands)
        """
        n_channels = eeg_data.shape[0]
        n_bands = len(self.BANDS)
        band_powers = np.zeros((n_channels, n_bands))
        
        for ch_idx in range(n_channels):
            channel_data = eeg_data[ch_idx, :]
            
            # Compute power spectral density using Welch's method
            try:
                frequencies, psd = sp_signal.welch(
                    channel_data, 
                    fs=self.sampling_rate,
                    nperseg=min(256, len(channel_data) // 2)
                )
                
                # Compute power in each band
                for band_idx, (band_name, (low_freq, high_freq)) in enumerate(self.BANDS.items()):
                    band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                    if np.any(band_mask):
                        band_powers[ch_idx, band_idx] = np.trapz(psd[band_mask], frequencies[band_mask])
                    else:
                        band_powers[ch_idx, band_idx] = 0
                        
            except Exception as e:
                logging.warning(f"Error computing band powers for channel {ch_idx}: {e}")
                band_powers[ch_idx, :] = 0
        
        # Normalize by total power
        total_power = np.sum(band_powers, axis=1, keepdims=True) + 1e-8
        band_powers = band_powers / total_power
        
        return band_powers
    
    def extract_asymmetry_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Compute asymmetry indices between channel pairs
        
        Frontal alpha asymmetry (right - left) is associated with approach/withdrawal motivation
        Positive asymmetry = left activation > right activation
        
        Returns:
            Array of asymmetry values for each pair and band
        """
        band_powers = self.extract_band_powers(eeg_data)
        asymmetries = []
        
        for left_ch, right_ch in self.asymmetry_pairs:
            if left_ch < band_powers.shape[0] and right_ch < band_powers.shape[0]:
                # Asymmetry = log(right) - log(left)
                for band_idx in range(band_powers.shape[1]):
                    left_power = band_powers[left_ch, band_idx] + 1e-8
                    right_power = band_powers[right_ch, band_idx] + 1e-8
                    asym = np.log(right_power) - np.log(left_power)
                    asymmetries.append(asym)
        
        return np.array(asymmetries) if asymmetries else np.zeros(1)
    
    def extract_spectral_entropy(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Compute spectral entropy for each channel
        
        Spectral entropy measures the complexity/regularity of the signal
        Higher entropy = more complex, less predictable
        
        Returns:
            Array of entropy values (one per channel)
        """
        n_channels = eeg_data.shape[0]
        entropies = np.zeros(n_channels)
        
        for ch_idx in range(n_channels):
            try:
                frequencies, psd = sp_signal.welch(
                    eeg_data[ch_idx, :],
                    fs=self.sampling_rate,
                    nperseg=min(256, eeg_data.shape[1] // 2)
                )
                
                # Normalize PSD to probability distribution
                psd_norm = psd / (np.sum(psd) + 1e-8)
                
                # Compute Shannon entropy
                entropies[ch_idx] = entropy(psd_norm + 1e-8)
                
            except Exception as e:
                logging.warning(f"Error computing spectral entropy for channel {ch_idx}: {e}")
                entropies[ch_idx] = 0
        
        return entropies
    
    def extract_statistical_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract basic statistical features
        
        Returns:
            Array of statistical features
        """
        features = []
        
        # Per-channel statistics
        features.append(np.mean(eeg_data, axis=1))  # Mean per channel
        features.append(np.std(eeg_data, axis=1))   # Std per channel
        features.append(np.min(eeg_data, axis=1))   # Min per channel
        features.append(np.max(eeg_data, axis=1))   # Max per channel
        
        # Global statistics
        features.append([np.mean(eeg_data)])        # Global mean
        features.append([np.std(eeg_data)])         # Global std
        
        return np.concatenate(features)
    
    def extract_connectivity_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract simple connectivity features (coherence, correlation)
        
        Returns:
            Array of connectivity features (simplified)
        """
        n_channels = eeg_data.shape[0]
        
        # Compute pairwise correlations (simplified connectivity)
        correlation_matrix = np.corrcoef(eeg_data)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices(n_channels, k=1)
        correlations = correlation_matrix[upper_triangle_indices]
        
        # Summary statistics of connectivity
        connectivity_features = [
            np.mean(correlations),
            np.std(correlations),
            np.max(correlations),
            np.min(correlations)
        ]
        
        return np.array(connectivity_features)
    
    def get_feature_dimension(self, n_channels: int) -> int:
        """Calculate expected feature dimension"""
        # Band powers: n_channels * 5 bands
        dim = n_channels * 5
        
        # Asymmetry: n_pairs * 5 bands
        if self.extract_asymmetry:
            dim += len(self.asymmetry_pairs) * 5
        
        # Spectral entropy: n_channels
        dim += n_channels
        
        # Statistical: 4 * n_channels + 2
        dim += 4 * n_channels + 2
        
        # Connectivity: 4 features
        if self.extract_connectivity:
            dim += 4
        
        return dim


def extract_eeg_features_batch(eeg_batch: np.ndarray,
                               sampling_rate: int = 250,
                               **kwargs) -> np.ndarray:
    """
    Convenience function to extract features from a batch of EEG data
    
    Args:
        eeg_batch: Batch of EEG data (batch, channels, timesteps)
        sampling_rate: Sampling rate
        **kwargs: Additional arguments for EEGFeatureExtractor
    
    Returns:
        Feature matrix (batch, features)
    """
    extractor = EEGFeatureExtractor(sampling_rate=sampling_rate, **kwargs)
    
    batch_size = eeg_batch.shape[0]
    feature_list = []
    
    for i in range(batch_size):
        features = extractor.extract_features(eeg_batch[i])
        feature_list.append(features)
    
    return np.stack(feature_list)
