"""
Simplified streaming adapters for physiological signals and voice
"""

import numpy as np
import logging
import threading
import queue
import time
import os
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PhysioStreamConfig:
    """Configuration for physiological streaming"""
    sampling_rate: int = 4  # Hz (lower rate for physio)
    window_size: float = 10.0  # seconds
    n_features: int = 24
    allow_synthetic_fallback: bool = True


class PhysioStreamAdapter:
    """Adapter for real-time physiological data (HRV, GSR, etc.)"""
    
    def __init__(self, config: PhysioStreamConfig):
        self.config = config
        if not self.config.allow_synthetic_fallback:
            raise RuntimeError("Synthetic physiological streaming is disabled")
        self.is_streaming = False
        self.data_queue = queue.Queue(maxsize=50)
        self.stream_thread = None
        
        logging.info("Physio Stream Adapter initialized (synthetic mode)")
    
    def start_stream(self):
        """Start physiological data streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        logging.info("Physio streaming started")
    
    def stop_stream(self):
        """Stop streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
    
    def _stream_loop(self):
        """Generate synthetic physio features"""
        while self.is_streaming:
            # Generate synthetic physio features
            features = np.random.randn(self.config.n_features).astype(np.float32)
            
            if not self.data_queue.full():
                self.data_queue.put(features)
            
            time.sleep(1.0 / self.config.sampling_rate)
    
    def get_latest_features(self) -> Optional[np.ndarray]:
        """Get latest physiological features"""
        if not self.is_streaming or self.data_queue.empty():
            return None
        
        # Get most recent
        latest = None
        while not self.data_queue.empty():
            try:
                latest = self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        return latest
    
    def cleanup(self):
        """Cleanup"""
        self.stop_stream()


@dataclass
class VoiceStreamConfig:
    """Configuration for voice streaming"""
    sampling_rate: int = 16000
    frame_duration: float = 1.0  # seconds per frame
    n_features: int = 128
    allow_synthetic_fallback: bool = True


class VoiceStreamAdapter:
    """Adapter for real-time voice/audio data"""
    
    def __init__(self, config: VoiceStreamConfig):
        self.config = config
        if not self.config.allow_synthetic_fallback:
            raise RuntimeError("Synthetic voice streaming is disabled")
        self.is_streaming = False
        self.data_queue = queue.Queue(maxsize=20)
        self.stream_thread = None
        
        logging.info("Voice Stream Adapter initialized (synthetic mode)")
    
    def start_stream(self):
        """Start voice streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        logging.info("Voice streaming started")
    
    def stop_stream(self):
        """Stop streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
    
    def _stream_loop(self):
        """Generate synthetic voice features"""
        while self.is_streaming:
            # Generate synthetic voice features (MFCC, prosody, etc.)
            features = np.random.randn(self.config.n_features).astype(np.float32)
            
            if not self.data_queue.full():
                self.data_queue.put(features)
            
            time.sleep(self.config.frame_duration)
    
    def get_latest_features(self) -> Optional[np.ndarray]:
        """Get latest voice features"""
        if not self.is_streaming or self.data_queue.empty():
            return None
        
        # Get most recent
        latest = None
        while not self.data_queue.empty():
            try:
                latest = self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        return latest
    
    def cleanup(self):
        """Cleanup"""
        self.stop_stream()


def create_physio_adapter(config: Dict) -> PhysioStreamAdapter:
    """Create physiological adapter from config"""
    sim_cfg = config.get('simulation', {})
    allow_synthetic = sim_cfg.get('enable_fallback')
    if allow_synthetic is None:
        env_value = os.environ.get("GCS_ALLOW_SIMULATED_DATA")
        if env_value is None:
            allow_synthetic = os.environ.get("GCS_ENV", "").strip().lower() != "production"
        else:
            allow_synthetic = env_value.strip().lower() in {"1", "true", "yes", "on"}
    physio_config = PhysioStreamConfig(
        n_features=config.get('physio_features', 24),
        allow_synthetic_fallback=bool(allow_synthetic)
    )
    return PhysioStreamAdapter(physio_config)


def create_voice_adapter(config: Dict) -> VoiceStreamAdapter:
    """Create voice adapter from config"""
    sim_cfg = config.get('simulation', {})
    allow_synthetic = sim_cfg.get('enable_fallback')
    if allow_synthetic is None:
        env_value = os.environ.get("GCS_ALLOW_SIMULATED_DATA")
        if env_value is None:
            allow_synthetic = os.environ.get("GCS_ENV", "").strip().lower() != "production"
        else:
            allow_synthetic = env_value.strip().lower() in {"1", "true", "yes", "on"}
    voice_config = VoiceStreamConfig(
        n_features=config.get('voice_features', 128),
        allow_synthetic_fallback=bool(allow_synthetic)
    )
    return VoiceStreamAdapter(voice_config)
