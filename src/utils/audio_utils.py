import numpy as np
from typing import Tuple

def convert_audio_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convert audio bytes to numpy array"""
    return np.frombuffer(audio_bytes, dtype=np.float32)

def convert_numpy_to_audio(audio_array: np.ndarray) -> bytes:
    """Convert numpy array back to audio bytes"""
    return audio_array.tobytes()