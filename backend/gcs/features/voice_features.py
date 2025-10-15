"""
Voice/Audio Feature Extraction

Extracts emotion-relevant features from voice/audio:
- MFCC (Mel-frequency cepstral coefficients)
- Prosodic features (pitch, energy, duration)
- Spectral features (chroma, spectral centroid, etc.)
- Optional: Pre-trained embeddings (wav2vec2, etc.)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fftpack import dct
from typing import Dict, List, Optional
import logging


class VoiceFeatureExtractor:
    """
    Extract emotion-relevant features from voice/audio
    
    Requires: librosa (optional, for advanced features)
    Falls back to scipy-based extraction
    """
    
    def __init__(self,
                 sampling_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 extract_prosody: bool = True,
                 extract_spectral: bool = True,
                 use_librosa: bool = True):
        """
        Args:
            sampling_rate: Audio sampling rate in Hz
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            extract_prosody: Extract prosodic features
            extract_spectral: Extract spectral features
            use_librosa: Try to use librosa if available
        """
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.extract_prosody = extract_prosody
        self.extract_spectral = extract_spectral
        
        # Try to import librosa
        self.librosa = None
        if use_librosa:
            try:
                import librosa
                self.librosa = librosa
                logging.info("Using librosa for voice feature extraction")
            except ImportError:
                logging.warning("librosa not available, using scipy fallback")
    
    def extract_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive voice features
        
        Args:
            audio_signal: Audio waveform (1D array)
        
        Returns:
            Feature vector (default 128 dimensions)
        """
        # If already processed to expected dimension, return as-is
        if audio_signal.ndim == 1 and audio_signal.shape[0] == 128:
            return audio_signal
        
        features = []
        
        # 1. MFCC features
        mfcc_features = self.extract_mfcc(audio_signal)
        features.append(mfcc_features)
        
        # 2. Prosodic features
        if self.extract_prosody:
            prosody_features = self.extract_prosody_features(audio_signal)
            features.append(prosody_features)
        
        # 3. Spectral features
        if self.extract_spectral:
            spectral_features = self.extract_spectral_features(audio_signal)
            features.append(spectral_features)
        
        # Concatenate and pad/trim to 128 dimensions
        all_features = np.concatenate(features)
        
        if len(all_features) < 128:
            all_features = np.pad(all_features, (0, 128 - len(all_features)))
        elif len(all_features) > 128:
            all_features = all_features[:128]
        
        return all_features.astype(np.float32)
    
    def extract_mfcc(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features
        
        Returns:
            MFCC feature vector (mean and std across time)
        """
        if self.librosa is not None:
            try:
                # Use librosa for MFCC extraction
                mfccs = self.librosa.feature.mfcc(
                    y=audio_signal,
                    sr=self.sampling_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Compute statistics across time
                mfcc_mean = np.mean(mfccs, axis=1)
                mfcc_std = np.std(mfccs, axis=1)
                
                return np.concatenate([mfcc_mean, mfcc_std])
                
            except Exception as e:
                logging.warning(f"Error extracting MFCC with librosa: {e}")
        
        # Fallback: scipy-based MFCC extraction (simplified)
        return self._extract_mfcc_scipy(audio_signal)
    
    def _extract_mfcc_scipy(self, audio_signal: np.ndarray) -> np.ndarray:
        """Simplified MFCC extraction using scipy"""
        try:
            # Compute STFT
            frequencies, times, stft = sp_signal.stft(
                audio_signal,
                fs=self.sampling_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length
            )
            
            # Power spectrogram
            power_spec = np.abs(stft) ** 2
            
            # Simplified mel filterbank (linear approximation)
            n_mels = 40
            mel_filters = self._create_mel_filterbank(len(frequencies), n_mels)
            mel_spec = mel_filters @ power_spec
            
            # Log mel spectrogram
            log_mel_spec = np.log(mel_spec + 1e-8)
            
            # DCT to get MFCCs
            mfccs = dct(log_mel_spec, axis=0, norm='ortho')[:self.n_mfcc, :]
            
            # Statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            return np.concatenate([mfcc_mean, mfcc_std])
            
        except Exception as e:
            logging.warning(f"Error in scipy MFCC extraction: {e}")
            return np.zeros(self.n_mfcc * 2)
    
    def _create_mel_filterbank(self, n_freqs: int, n_mels: int) -> np.ndarray:
        """Create simplified mel filterbank"""
        # Simplified linear filterbank as placeholder
        filterbank = np.zeros((n_mels, n_freqs))
        
        for i in range(n_mels):
            center = int((i + 1) * n_freqs / (n_mels + 1))
            width = max(1, n_freqs // (n_mels * 2))
            
            start = max(0, center - width)
            end = min(n_freqs, center + width)
            
            filterbank[i, start:end] = 1.0
        
        # Normalize
        filterbank = filterbank / (np.sum(filterbank, axis=1, keepdims=True) + 1e-8)
        
        return filterbank
    
    def extract_prosody_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract prosodic features (pitch, energy, speaking rate)
        
        Returns:
            Prosody feature vector
        """
        if self.librosa is not None:
            try:
                # Pitch (fundamental frequency) using librosa
                pitches, magnitudes = self.librosa.piptrack(
                    y=audio_signal,
                    sr=self.sampling_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Extract pitch contour (select max magnitude at each time)
                pitch_contour = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_contour.append(pitch)
                
                if len(pitch_contour) > 0:
                    pitch_mean = np.mean(pitch_contour)
                    pitch_std = np.std(pitch_contour)
                    pitch_range = np.max(pitch_contour) - np.min(pitch_contour)
                else:
                    pitch_mean = pitch_std = pitch_range = 0
                
                # Energy (RMS)
                rms = self.librosa.feature.rms(y=audio_signal, hop_length=self.hop_length)[0]
                energy_mean = np.mean(rms)
                energy_std = np.std(rms)
                
                # Zero crossing rate (proxy for speaking rate/energy)
                zcr = self.librosa.feature.zero_crossing_rate(y=audio_signal, hop_length=self.hop_length)[0]
                zcr_mean = np.mean(zcr)
                
                return np.array([pitch_mean, pitch_std, pitch_range, 
                               energy_mean, energy_std, zcr_mean])
                
            except Exception as e:
                logging.warning(f"Error extracting prosody with librosa: {e}")
        
        # Fallback: scipy-based prosody
        return self._extract_prosody_scipy(audio_signal)
    
    def _extract_prosody_scipy(self, audio_signal: np.ndarray) -> np.ndarray:
        """Simplified prosody extraction using scipy"""
        try:
            # Energy (RMS in frames)
            frame_length = self.n_fft
            hop_length = self.hop_length
            
            n_frames = (len(audio_signal) - frame_length) // hop_length + 1
            energy = []
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio_signal[start:end]
                rms = np.sqrt(np.mean(frame ** 2))
                energy.append(rms)
            
            energy = np.array(energy)
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            
            # Zero crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(audio_signal)))) / (2 * len(audio_signal))
            
            # Simplified pitch estimate (not accurate, just for feature completeness)
            pitch_mean = 120.0  # Placeholder
            pitch_std = 20.0
            pitch_range = 50.0
            
            return np.array([pitch_mean, pitch_std, pitch_range,
                           energy_mean, energy_std, zcr])
            
        except Exception as e:
            logging.warning(f"Error in scipy prosody extraction: {e}")
            return np.zeros(6)
    
    def extract_spectral_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract spectral features
        
        Returns:
            Spectral feature vector
        """
        if self.librosa is not None:
            try:
                # Spectral centroid
                cent = self.librosa.feature.spectral_centroid(
                    y=audio_signal, sr=self.sampling_rate, n_fft=self.n_fft)[0]
                
                # Spectral rolloff
                rolloff = self.librosa.feature.spectral_rolloff(
                    y=audio_signal, sr=self.sampling_rate, n_fft=self.n_fft)[0]
                
                # Spectral contrast
                contrast = self.librosa.feature.spectral_contrast(
                    y=audio_signal, sr=self.sampling_rate, n_fft=self.n_fft)
                
                features = [
                    np.mean(cent), np.std(cent),
                    np.mean(rolloff), np.std(rolloff),
                ]
                features.extend(np.mean(contrast, axis=1))
                
                return np.array(features)
                
            except Exception as e:
                logging.warning(f"Error extracting spectral features: {e}")
        
        # Fallback: return zeros
        return np.zeros(11)  # 4 + 7 spectral contrast bands


def extract_voice_features_batch(audio_batch: np.ndarray,
                                 sampling_rate: int = 16000,
                                 **kwargs) -> np.ndarray:
    """
    Convenience function to extract features from a batch
    
    Args:
        audio_batch: Batch of audio waveforms (batch, samples) or pre-extracted features (batch, 128)
        sampling_rate: Sampling rate
        **kwargs: Additional arguments for VoiceFeatureExtractor
    
    Returns:
        Feature matrix (batch, 128)
    """
    # If already the right shape, return as-is
    if audio_batch.ndim == 2 and audio_batch.shape[1] == 128:
        return audio_batch
    
    extractor = VoiceFeatureExtractor(sampling_rate=sampling_rate, **kwargs)
    
    batch_size = audio_batch.shape[0]
    feature_list = []
    
    for i in range(batch_size):
        features = extractor.extract_features(audio_batch[i])
        feature_list.append(features)
    
    return np.stack(feature_list)
