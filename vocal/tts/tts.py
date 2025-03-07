from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from fastapi.responses import JSONResponse
import os
import time
from typing import Dict, List, Optional
from fastapi.background import BackgroundTasks
from vocal.config.agents_config import agent_manager
from typing import AsyncGenerator
from dotenv import load_dotenv
from .base_tts import BaseTTS
from .local_tts import LocalTTS
from .external.cartesia_tts import CartesiaTTS
from .external.elevenlabs_tts import ElevenLabsTTS
from .base_tts import TTSChunk

load_dotenv()

class TTS:
    def __init__(self):
        self.tts: Optional[BaseTTS] = None
        self._setup_tts()

    def _setup_tts(self):
        use_external = os.getenv("USE_EXTERNAL_TTS", "False").lower() == "true"
        provider = os.getenv("EXTERNAL_TTS_PROVIDER", "").lower()

        print(f"Setting up TTS: use_external: {use_external}, provider: {provider}")

        if use_external:
            if provider == "cartesia":
                self.tts = CartesiaTTS(api_key=os.getenv("CARTESIA_API_KEY"))
            elif provider == "elevenlabs":
                self.tts = ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"))
            else:
                raise ValueError(f"Unsupported external TTS provider: {provider}")
        else:
            self.tts = LocalTTS()

        self.setup()

        print(f"TTS setup complete")

    def setup(self):
        """Initialize the TTS"""
        if self.tts:
            self.tts.setup()

    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None) -> TTSChunk:
        """Generate speech for the given prompt"""
        if not self.tts:
            raise RuntimeError("TTS not initialized")
        
        return await self.tts.generate_speech(text, language, voice_id, voice_samples)

    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None) -> AsyncGenerator[TTSChunk, None]:
        if not self.tts:
            raise RuntimeError("TTS not initialized")
        
        return await self.tts.generate_speech_stream(text, language, voice_id, voice_samples)
                

    async def cleanup(self):
        """Cleanup resources"""
        if self.tts:
            await self.tts.cleanup()

# Initialize tts instance
tts_instance = TTS()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_audio_chunks(websocket: WebSocket, text: str, config_id: str):
    """TTS streaming implementation with voice conversion and raw PCM output"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })

        voice_samples, _, language, _, _ = agent_manager.get_agent_config(config_id)
        
        chunks = await tts_instance.generate_speech_stream(text, language, '', voice_samples)

        for chunk in chunks:
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk.chunk,
                "format": chunk.format,
                "sample_rate": chunk.sample_rate,
                "timestamp": time.time()
            })

        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })

    except WebSocketDisconnect:
        print("WebSocket disconnected during streaming")
    except Exception as e:
        print(f"Error in stream_audio_chunks: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
   
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if not isinstance(data, dict):
                continue
                
            text = data.get("text")
            config_id = data.get("config_id", "")
            session_id = data.get("session_id", "")
            
            if text:
                await stream_audio_chunks(websocket, text, config_id)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

    
@app.get("/tts")
async def generate_tts(
    text: str = Query(..., description="Text to convert to speech"),
    config_id: str = Query(..., description="Config ID to use")
):        
    try:
        voice_samples, _, language, _, _ = agent_manager.get_agent_config(config_id)

        tts_chunk = await tts_instance.generate_speech(text, language, voice_samples)

        return JSONResponse({
            "audio": tts_chunk.chunk,
            "format": tts_chunk.format,
            "sample_rate": tts_chunk.sample_rate
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
