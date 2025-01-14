from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # WebSocket settings
    WS_PING_INTERVAL: int = 20  # seconds
    CHUNK_SIZE: int = 4096  # bytes for audio streaming

settings = Settings()