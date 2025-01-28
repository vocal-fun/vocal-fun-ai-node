import numpy as np
import time

class AudioSpeechDetector:
    def __init__(self, 
                 sample_rate=16000, 
                 silence_threshold=-40,  # in decibels
                 min_speech_duration=0.5,  # seconds
                 max_silence_duration=1.0,  # seconds
                 max_recording_duration=10.0  # max continuous recording time
                ):
        """
        Initialize speech detector with configurable parameters
        
        :param sample_rate: Audio sample rate (Hz)
        :param silence_threshold: Decibel level below which is considered silence 
        :param min_speech_duration: Minimum duration of speech to be considered valid (seconds)
        :param max_silence_duration: Maximum duration of silence within speech (seconds)
        :param max_recording_duration: Maximum duration of continuous recording
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_silence_duration = max_silence_duration
        self.max_recording_duration = max_recording_duration
        
        # Internal state tracking
        self.current_audio_chunks = []
        self.last_speech_time = None
        self.start_recording_time = None
        
    def add_audio_chunk(self, audio_chunk):
        """
        Add an audio chunk and check for speech detection conditions
        
        :param audio_chunk: NumPy array of audio samples
        :return: Dict with detection status and potential actions
        """
        # Validate input
        if audio_chunk is None or len(audio_chunk) == 0:
            return {"action": "continue"}
        
        # Initialize recording start time if not set
        if self.start_recording_time is None:
            self.start_recording_time = time.time()
        
        # Store the chunk
        self.current_audio_chunks.append(audio_chunk)
        
        # Convert audio to decibels
        audio_db = 20 * np.log10(np.abs(audio_chunk) + np.finfo(float).eps)
        
        # Detect speech in this chunk
        speech_mask = audio_db > self.silence_threshold
        total_samples = len(audio_chunk)
        speech_samples = np.sum(speech_mask)
        
        # Convert to seconds
        speech_duration = speech_samples / self.sample_rate
        
        # Check if there's any speech in this chunk
        if speech_duration > 0:
            self.last_speech_time = time.time()
            return {"action": "continue"}
        
        # Check for potential speech end conditions
        current_time = time.time()
        
        # Check if max recording duration exceeded
        if current_time - self.start_recording_time > self.max_recording_duration:
            return {
                "action": "process",
                "reason": "max_duration_reached"
            }
        
        # If no speech detected recently
        if (self.last_speech_time is not None and 
            current_time - self.last_speech_time > self.max_silence_duration):
            
            # Combine audio chunks
            full_audio = np.concatenate(self.current_audio_chunks) if self.current_audio_chunks else np.array([])
            
            # Verify if we have enough speech duration
            audio_db_full = 20 * np.log10(np.abs(full_audio) + np.finfo(float).eps)
            speech_mask_full = audio_db_full > self.silence_threshold
            speech_duration_full = np.sum(speech_mask_full) / self.sample_rate
            
            # Reset state
            self.current_audio_chunks = []
            self.last_speech_time = None
            self.start_recording_time = None
            
            # Check if speech duration meets minimum requirement
            if speech_duration_full >= self.min_speech_duration:
                return {
                    "action": "process",
                    "reason": "speech_end_detected"
                }
        
        return {"action": "continue"}