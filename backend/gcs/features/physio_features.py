"""
Physiological Feature Extraction

Extracts features from physiological signals:
- HRV (Heart Rate Variability) time-domain and frequency-domain features
- EDA (Electrodermal Activity) tonic/phasic decomposition
- Respiratory features
- Raw signal statistics
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Optional
import logging


class PhysioFeatureExtractor:
    """
    Extract emotion-relevant features from physiological signals
    
    Expects pre-computed physiological features or raw signals
    """
    
    def __init__(self, 
                 extract_hrv: bool = True,
                 extract_eda: bool = True,
                 extract_resp: bool = True):
        """
        Args:
            extract_hrv: Extract HRV features
            extract_eda: Extract EDA features  
            extract_resp: Extract respiratory features
        """
        self.extract_hrv = extract_hrv
        self.extract_eda = extract_eda
        self.extract_resp = extract_resp
    
    def extract_features(self, physio_data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive physiological features
        
        Args:
            physio_data: Physiological feature vector (if already processed)
                        or dict with raw signals
        
        Returns:
            Feature vector as 1D array (24 dimensions)
        """
        # If already processed to expected dimension, return as-is
        if physio_data.ndim == 1 and physio_data.shape[0] == 24:
            return physio_data
        
        # Otherwise extract features (simplified - assumes pre-processed input)
        # In production, would implement full signal processing pipeline
        
        # Ensure output is 24 dimensions
        features = physio_data.flatten()
        
        if len(features) < 24:
            features = np.pad(features, (0, 24 - len(features)))
        elif len(features) > 24:
            features = features[:24]
        
        return features.astype(np.float32)
    
    def extract_hrv_features(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Extract HRV features from RR intervals
        
        Time-domain features:
        - Mean HR
        - SDNN (standard deviation of NN intervals)
        - RMSSD (root mean square of successive differences)
        - pNN50 (percentage of successive differences > 50ms)
        
        Frequency-domain features:
        - LF power (0.04-0.15 Hz)
        - HF power (0.15-0.4 Hz)
        - LF/HF ratio
        
        Args:
            rr_intervals: RR intervals in milliseconds
        
        Returns:
            HRV feature vector (7 features)
        """
        if len(rr_intervals) < 2:
            return np.zeros(7)
        
        features = []
        
        # Time-domain features
        mean_hr = 60000 / np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100
        
        features.extend([mean_hr, sdnn, rmssd, pnn50])
        
        # Frequency-domain features (simplified)
        try:
            # Resample to evenly spaced time series
            time_points = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            
            # Interpolate to regular grid
            fs = 4  # 4 Hz sampling
            regular_time = np.arange(0, time_points[-1], 1/fs)
            
            if len(regular_time) > 10:
                hr_interpolated = np.interp(regular_time, time_points, 60000 / rr_intervals)
                
                # Compute PSD
                frequencies, psd = sp_signal.welch(hr_interpolated, fs=fs, nperseg=min(64, len(hr_interpolated)//2))
                
                # LF and HF power
                lf_mask = (frequencies >= 0.04) & (frequencies <= 0.15)
                hf_mask = (frequencies >= 0.15) & (frequencies <= 0.4)
                
                lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
                hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
                lf_hf_ratio = lf_power / (hf_power + 1e-8)
            else:
                lf_power = 0
                hf_power = 0
                lf_hf_ratio = 0
        except Exception as e:
            logging.warning(f"Error computing frequency-domain HRV: {e}")
            lf_power = 0
            hf_power = 0
            lf_hf_ratio = 0
        
        features.extend([lf_power, hf_power, lf_hf_ratio])
        
        return np.array(features)
    
    def extract_eda_features(self, eda_signal: np.ndarray, 
                            sampling_rate: float = 4.0) -> np.ndarray:
        """
        Extract EDA features
        
        Features:
        - Mean EDA level
        - Std of EDA
        - Tonic component (slowly varying baseline)
        - Number of SCRs (skin conductance responses)
        
        Args:
            eda_signal: EDA signal
            sampling_rate: Sampling rate in Hz
        
        Returns:
            EDA feature vector (4 features)
        """
        if len(eda_signal) == 0:
            return np.zeros(4)
        
        features = []
        
        # Basic statistics
        mean_eda = np.mean(eda_signal)
        std_eda = np.std(eda_signal)
        features.extend([mean_eda, std_eda])
        
        # Tonic component (low-pass filter)
        try:
            # Simple moving average as tonic component
            window_size = int(sampling_rate * 2)  # 2-second window
            if window_size > 1 and len(eda_signal) > window_size:
                kernel = np.ones(window_size) / window_size
                tonic = np.convolve(eda_signal, kernel, mode='same')
                tonic_level = np.mean(tonic)
            else:
                tonic_level = mean_eda
        except Exception:
            tonic_level = mean_eda
        
        features.append(tonic_level)
        
        # Count SCRs (peaks in derivative)
        try:
            eda_derivative = np.diff(eda_signal)
            threshold = np.std(eda_derivative) * 0.5
            peaks = 0
            
            for i in range(1, len(eda_derivative) - 1):
                if (eda_derivative[i] > threshold and 
                    eda_derivative[i] > eda_derivative[i-1] and
                    eda_derivative[i] > eda_derivative[i+1]):
                    peaks += 1
            
            scr_count = float(peaks)
        except Exception:
            scr_count = 0
        
        features.append(scr_count)
        
        return np.array(features)
    
    def extract_resp_features(self, resp_signal: np.ndarray,
                             sampling_rate: float = 4.0) -> np.ndarray:
        """
        Extract respiratory features
        
        Features:
        - Respiratory rate (breaths per minute)
        - Respiratory depth (amplitude)
        - Respiratory irregularity
        
        Args:
            resp_signal: Respiratory signal
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Respiratory feature vector (3 features)
        """
        if len(resp_signal) < 10:
            return np.zeros(3)
        
        features = []
        
        # Normalize signal
        resp_normalized = (resp_signal - np.mean(resp_signal)) / (np.std(resp_signal) + 1e-8)
        
        # Detect peaks (inhalation peaks)
        peaks = []
        for i in range(1, len(resp_normalized) - 1):
            if (resp_normalized[i] > 0.5 and
                resp_normalized[i] > resp_normalized[i-1] and
                resp_normalized[i] > resp_normalized[i+1]):
                peaks.append(i)
        
        # Respiratory rate
        if len(peaks) > 1:
            duration = len(resp_signal) / sampling_rate  # seconds
            resp_rate = len(peaks) / duration * 60  # breaths per minute
        else:
            resp_rate = 0
        
        features.append(resp_rate)
        
        # Respiratory depth
        resp_depth = np.max(resp_signal) - np.min(resp_signal)
        features.append(resp_depth)
        
        # Respiratory irregularity (coefficient of variation of inter-breath intervals)
        if len(peaks) > 2:
            peak_intervals = np.diff(peaks)
            irregularity = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8)
        else:
            irregularity = 0
        
        features.append(irregularity)
        
        return np.array(features)


def extract_physio_features_batch(physio_batch: np.ndarray,
                                  **kwargs) -> np.ndarray:
    """
    Convenience function to extract features from a batch
    
    Args:
        physio_batch: Batch of physio data (batch, features)
        **kwargs: Additional arguments for PhysioFeatureExtractor
    
    Returns:
        Feature matrix (batch, 24)
    """
    extractor = PhysioFeatureExtractor(**kwargs)
    
    batch_size = physio_batch.shape[0]
    feature_list = []
    
    for i in range(batch_size):
        features = extractor.extract_features(physio_batch[i])
        feature_list.append(features)
    
    return np.stack(feature_list)
