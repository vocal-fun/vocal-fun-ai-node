import numpy as np
import time

class AudioSpeechDetector:
    def __init__(self, 
                 sample_rate=16000, 
                 silence_threshold=-50,  # Lowered sensitivity
                 absolute_silence_threshold=-70,  # Additional absolute silence check
                 min_speech_duration=0.3,  # Reduced minimum speech duration
                 max_silence_duration=1.5,  # Increased silence duration
                 max_recording_duration=10.0,
                 debug=True):
        """
        Initialize speech detector with configurable parameters
        
        :param sample_rate: Audio sample rate (Hz)
        :param silence_threshold: Decibel level below which is considered relative silence 
        :param absolute_silence_threshold: Absolute low-energy threshold
        :param min_speech_duration: Minimum duration of speech to be considered valid (seconds)
        :param max_silence_duration: Maximum duration of silence within speech (seconds)
        :param max_recording_duration: Maximum duration of continuous recording
        :param debug: Enable debug logging
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.absolute_silence_threshold = absolute_silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_silence_duration = max_silence_duration
        self.max_recording_duration = max_recording_duration
        self.debug = debug
        
        # Reset internal state
        self.reset()
    
    def reset(self):
        """
        Reset the internal state of the speech detector
        """
        self.current_audio_chunks = []
        self.last_speech_time = None
        self.start_recording_time = None
        self.consecutive_silence_chunks = 0
    
    def _log(self, message):
        """
        Debug logging
        """
        if self.debug:
            print(f"[SpeechDetector] {message}")
    
    def add_audio_chunk(self, audio_chunk):
        """
        Add an audio chunk and check for speech detection conditions
        
        :param audio_chunk: NumPy array of audio samples
        :return: Dict with detection status and potential actions
        """
        # Validate input
        if audio_chunk is None or len(audio_chunk) == 0:
            return {"action": "continue"}
        
        current_time = time.time()
        
        # Initialize recording start time if not set
        if self.start_recording_time is None:
            self.start_recording_time = current_time
        
        # Store the chunk
        self.current_audio_chunks.append(audio_chunk)
        
        # Convert audio to decibels with error handling
        try:
            audio_db = 20 * np.log10(np.abs(audio_chunk) + np.finfo(float).eps)
        except Exception as e:
            self._log(f"Error converting to decibels: {e}")
            return {"action": "continue"}
        
        # Detect speech using two thresholds
        relative_speech_mask = audio_db > self.silence_threshold
        absolute_speech_mask = audio_db > self.absolute_silence_threshold
        
        # Combine masks for more robust detection
        speech_mask = relative_speech_mask & absolute_speech_mask
        
        # Calculate speech characteristics
        total_samples = len(audio_chunk)
        speech_samples = np.sum(speech_mask)
        speech_duration = speech_samples / self.sample_rate
        
        # Detailed logging
        self._log(f"Chunk Analysis: " +
                  f"Total Samples: {total_samples}, " +
                  f"Speech Samples: {speech_samples}, " +
                  f"Speech Duration: {speech_duration:.3f}s")
        
        # Check if there's speech in this chunk
        if speech_samples > 0:
            self.last_speech_time = current_time
            self.consecutive_silence_chunks = 0
            self._log("Speech detected in chunk")
            return {"action": "continue"}
        
        # Increment consecutive silence chunks
        self.consecutive_silence_chunks += 1
        
        # Check max recording duration
        if current_time - self.start_recording_time > self.max_recording_duration:
            self._log("Max recording duration reached")
            return {
                "action": "process",
                "reason": "max_duration_reached"
            }
        
        # Check for prolonged silence
        silence_duration = self.consecutive_silence_chunks * (len(audio_chunk) / self.sample_rate)
        
        self._log(f"Silence tracking: " +
                  f"Consecutive Silent Chunks: {self.consecutive_silence_chunks}, " +
                  f"Total Silence Duration: {silence_duration:.3f}s")
        
        # If silence has persisted beyond max_silence_duration
        if silence_duration > self.max_silence_duration:
            # Combine audio chunks
            full_audio = np.concatenate(self.current_audio_chunks) if self.current_audio_chunks else np.array([])
            
            # Verify speech duration in full recording
            full_audio_db = 20 * np.log10(np.abs(full_audio) + np.finfo(float).eps)
            relative_speech_mask_full = full_audio_db > self.silence_threshold
            absolute_speech_mask_full = full_audio_db > self.absolute_silence_threshold
            full_speech_mask = relative_speech_mask_full & absolute_speech_mask_full
            
            speech_duration_full = np.sum(full_speech_mask) / self.sample_rate
            
            self._log(f"Full Audio Speech Duration: {speech_duration_full:.3f}s")
            
            # Check if speech duration meets minimum requirement
            if speech_duration_full >= self.min_speech_duration:
                self._log("Speech end detected - processing audio")
                processed_chunks = self.current_audio_chunks.copy()
                
                # Reset state
                self.reset()
                
                return {
                    "action": "process",
                    "reason": "speech_end_detected",
                    "audio_chunks": processed_chunks
                }
            else:
                self._log("Not enough speech duration to process")
                # Reset if not enough speech detected
                self.reset()
        
        return {"action": "continue"}