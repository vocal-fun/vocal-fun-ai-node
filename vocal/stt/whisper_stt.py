from faster_whisper import WhisperModel
import torch
from .base_stt import BaseSTT

class LocalWhisperSTT(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            "small",
            device="auto",
            compute_type="int8",
            download_root="./models"
        )

    async def transcribe(self, audio_data: bytes, language: str) -> str:
        if self.model is None:
            self.setup()
            
        try:
            # Save audio data to temporary file
            temp_path = "temp_audio_file"
            with open(temp_path, "wb") as f:
                f.write(audio_data)

            # Transcribe using local model
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5
            )
            
            transcribed_text = " ".join([segment.text for segment in segments])

            # Clean up temp file
            import os
            os.remove(temp_path)

            return transcribed_text

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 