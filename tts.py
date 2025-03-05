from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from fastapi.responses import JSONResponse
import base64
import os
import time
import torch
import torchaudio
import asyncio
import io
from typing import Dict, List
import uuid
from fastapi.background import BackgroundTasks
from config.agents_config import agent_manager
from cartesia import AsyncCartesia
import aiohttp
from typing import AsyncGenerator
from pathlib import Path
from scipy.io import wavfile
import sys
from dotenv import load_dotenv
import logging

load_dotenv()

ENABLE_LOCAL_MODEL = True  # Set to False to disable local model
CARTESIA_API_KEY = os.getenv('CARTESIA_API_KEY')

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
CHUNK_SIZE = 1024
API_BASE = "https://api.elevenlabs.io/v1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a global lock for XTTS model
xtts_lock = asyncio.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

class CartesiaWebSocketManager:
    def __init__(self, api_key: str, pool_size: int = 1):
        self.api_key = api_key
        self.pool_size = pool_size
        self.client = AsyncCartesia(api_key=api_key)
        self.connections: List[Dict] = []
        self.lock = asyncio.Lock()
        
    async def initialize_pool(self):
        """Initialize the WebSocket connection pool"""
        print("Initializing WebSocket pool...")
        for _ in range(self.pool_size):
            try:
                ws = await self.client.tts.websocket()
                self.connections.append({
                    "websocket": ws,
                    "in_use": False,
                    "last_used": time.time()
                })
                print(f"Added connection to pool. Total: {len(self.connections)}")
            except Exception as e:
                print(f"Error creating WebSocket connection: {e}")
        
    async def get_connection(self):
        """Get an available WebSocket connection from the pool"""
        async with self.lock:
            # First try to find an unused connection
            for conn in self.connections:
                if not conn["in_use"]:
                    conn["in_use"] = True
                    conn["last_used"] = time.time()
                    return conn["websocket"]
            
            # If no available connections, create a new one
            try:
                ws = await self.client.tts.websocket()
                conn = {
                    "websocket": ws,
                    "in_use": True,
                    "last_used": time.time()
                }
                self.connections.append(conn)
                return ws
            except Exception as e:
                print(f"Error creating new WebSocket connection: {e}")
                raise
    
    async def release_connection(self, ws):
        """Release a WebSocket connection back to the pool"""
        async with self.lock:
            for conn in self.connections:
                if conn["websocket"] == ws:
                    conn["in_use"] = False
                    conn["last_used"] = time.time()
                    break
    
    async def maintain_pool(self):
        """Periodically check and maintain the WebSocket pool"""
        while True:
            try:
                async with self.lock:
                    current_time = time.time()
                    # Check each connection
                    for i in range(len(self.connections) - 1, -1, -1):
                        conn = self.connections[i]
                        # If connection is old (1 hour) and not in use, close it
                        if not conn["in_use"] and (current_time - conn["last_used"]) > 3600:
                            try:
                                await conn["websocket"].close()
                            except:
                                pass
                            self.connections.pop(i)
                    
                    # Ensure we maintain minimum pool size
                    while len(self.connections) < self.pool_size:
                        try:
                            ws = await self.client.tts.websocket()
                            self.connections.append({
                                "websocket": ws,
                                "in_use": False,
                                "last_used": time.time()
                            })
                        except Exception as e:
                            print(f"Error maintaining pool: {e}")
                            break
                            
            except Exception as e:
                print(f"Error in maintain_pool: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cartesia clients
cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)
ws_manager = CartesiaWebSocketManager(api_key=CARTESIA_API_KEY)


# Initialize local XTTS model if enabled
if ENABLE_LOCAL_MODEL:
    xttsPath = os.getenv('XTTS_MODEL_PATH')
    print("Loading local model...")
    config = XttsConfig()
    config.load_json(xttsPath + "/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=xttsPath, use_deepspeed=device == "cuda")
    if device == "cuda":
        model.cuda()

    # Cache for voice lines and speaker latents
    voice_lines_cached = {}
    speaker_latents_cache = {}

# Cartesia output formats
cartesia_stream_format = {
    "container": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 24000,
}

cartesia_bytes_format = {
    "container": "wav",
    "encoding": "pcm_f32le",
    "sample_rate": 24000,
}


@app.on_event("startup")
async def startup_event():
    """Initialize the WebSocket pool on app startup"""
    if not ENABLE_LOCAL_MODEL:
        await ws_manager.initialize_pool()
        # Start the pool maintenance task
        asyncio.create_task(ws_manager.maintain_pool())

async def stream_audio_chunks(websocket: WebSocket, text: str, config_id: str):
    """TTS streaming implementation with voice conversion and raw PCM output"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })

        # Load TTS voice
        voice_samples, _, language, _, _ = agent_manager.get_agent_config(config_id)
        
        if config_id not in speaker_latents_cache:
            print("Computing speaker latents...")
            async with xtts_lock:  # Add lock for computing latents
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=voice_samples)
                speaker_latents_cache[config_id] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[config_id]

        print("Starting streaming inference...")
        t0 = time.time()
        
        # Add lock for the inference
        async with xtts_lock:
            chunks = model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7
            )

            chunk_counter = 0
            for chunk in chunks:
                if chunk_counter == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                
                # Convert tensor to raw PCM bytes
                chunk_bytes = chunk.squeeze().cpu().numpy().tobytes()
                chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                
                print('sending chunk: ' + str(chunk_counter))
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_base64,
                    "chunk_id": chunk_counter,
                    "format": "pcm_f32le",
                    "sample_rate": 24000,
                    "timestamp": time.time()
                })
                chunk_counter += 1
                await asyncio.sleep(0.01)

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

async def stream_audio_chunks_cartesia(websocket: WebSocket, text: str, config_id: str):
    """Improved Cartesia streaming implementation with better buffer management"""
    ws = None
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })
        
        t0 = time.time()
        session_id = str(uuid.uuid4())

        _, _, _, voice_id, _ = agent_manager.get_agent_config(config_id)

        # Get a connection from the pool
        ws = await ws_manager.get_connection()
        
        print(f"Time for socket reuse: {time.time() - t0}")

        # Send the request and get the stream
        stream = await ws.send(
            model_id="sonic",
            transcript=text,
            voice_id=voice_id,
            stream=True,
            output_format=cartesia_stream_format
        )

        # Buffer for accumulating audio data
        buffer = np.array([], dtype=np.float32)
        chunk_size = 4800  # 0.2 seconds at 24kHz
        chunk_counter = 0
        
        async for output in stream:
            if chunk_counter == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(output["audio"], dtype=np.float32)
            
            # Add to buffer
            buffer = np.append(buffer, audio_chunk)
            
            # Process complete chunks
            while len(buffer) >= chunk_size:
                # Extract chunk with overlap
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]  # Remove processed data
                
                # Apply fade in/out to reduce artifacts
                if chunk_counter > 0:  # Apply fade-in
                    fade_samples = 240  # 10ms fade
                    fade_in = np.linspace(0, 1, fade_samples)
                    chunk[:fade_samples] *= fade_in
                
                if len(buffer) < chunk_size:  # Apply fade-out to last chunk
                    fade_samples = 240
                    fade_out = np.linspace(1, 0, fade_samples)
                    chunk[-fade_samples:] *= fade_out
                
                # Convert to bytes and send
                chunk_bytes = chunk.tobytes()
                chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_base64,
                    "chunk_id": chunk_counter,
                    "format": "pcm_f32le",
                    "sample_rate": cartesia_stream_format["sample_rate"],
                    "timestamp": time.time()
                })
                
                chunk_counter += 1
        
        # Send any remaining buffer
        if len(buffer) > 0:
            chunk_bytes = buffer.tobytes()
            chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
            
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk_base64,
                "chunk_id": chunk_counter,
                "format": "pcm_f32le",
                "sample_rate": cartesia_stream_format["sample_rate"],
                "timestamp": time.time()
            })

        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })

    except WebSocketDisconnect:
        print("WebSocket disconnected during streaming")
    except Exception as e:
        print(f"Error in stream_audio_chunks_cartesia: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
    finally:
        if ws:
            await ws_manager.release_connection(ws)

async def stream_elevenlabs_audio(voice_id: str, text: str) -> AsyncGenerator[bytes, None]:
    """Stream audio from ElevenLabs API"""
    url = f"{API_BASE}/text-to-speech/{voice_id}/stream?output_format=pcm_24000"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_flash_v2"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"ElevenLabs API error: {error_text}")
                
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                yield chunk

async def stream_audio_chunks_elevenlabs(websocket: WebSocket, text: str, config_id: str):
    """ElevenLabs streaming implementation"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })
        
        t0 = time.time()
        _, _, _, _, elevenlabs_voice_id = agent_manager.get_agent_config(config_id)
        
        chunk_counter = 0
        async for chunk in stream_elevenlabs_audio(elevenlabs_voice_id, text):
            chunk_base64 = base64.b64encode(chunk).decode("utf-8")
            
            if (chunk_counter == 0):
                print(f"Time to first chunk: {time.time() - t0}")

            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk_base64,
                "chunk_id": chunk_counter,
                "format": "pcm_f32le",
                "sample_rate": 24000,
                "timestamp": time.time()
            })
            chunk_counter += 1
            await asyncio.sleep(0.01)

        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })

    except WebSocketDisconnect:
        print("WebSocket disconnected during streaming")
    except Exception as e:
        print(f"Error in stream_audio_chunks_elevenlabs: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
    """Original XTTS WebSocket endpoint"""
    if not ENABLE_LOCAL_MODEL:
        await websocket.close(code=1000, reason="Local model is disabled")
        return
        
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
                # Get the config using config_id
                voice_samples, system_prompt, language, cartesia_id, elevenlabs_id = agent_manager.get_agent_config(config_id)
                await stream_audio_chunks(websocket, text, config_id)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/tts/stream/cartesia")
async def tts_stream_cartesia(websocket: WebSocket):
    """Cartesia WebSocket endpoint"""
    if not cartesia_client:
        await websocket.close(code=1000, reason="Cartesia API is not configured")
        return
        
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
                # Get the config using config_id
                voice_samples, system_prompt, language, cartesia_id, elevenlabs_id = agent_manager.get_agent_config(config_id)
                await stream_audio_chunks_cartesia(websocket, text, config_id)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/tts/stream/elevenlabs")
async def tts_stream_elevenlabs(websocket: WebSocket):
    """ElevenLabs WebSocket endpoint"""
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
                # Get the config using config_id
                voice_samples, system_prompt, language, cartesia_id, elevenlabs_id = agent_manager.get_agent_config(config_id)
                await stream_audio_chunks_elevenlabs(websocket, text, config_id)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
    
@app.get("/tts")
async def generate_tts(
    text: str = Query(..., description="Text to convert to speech"),
    config_id: str = Query(..., description="Config ID to use")
):
    """Original XTTS endpoint for single audio generation with raw PCM output"""
    if not ENABLE_LOCAL_MODEL:
        raise HTTPException(status_code=400, detail="Local model is disabled")
        
    try:
        voice_samples, _, language, _, _ = agent_manager.get_agent_config(config_id)
        
        if config_id not in speaker_latents_cache:
            print("Computing speaker latents...")
            async with xtts_lock:  # Add lock for computing latents
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=voice_samples)
                speaker_latents_cache[config_id] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[config_id]

        print("Starting full audio generation...")
        t0 = time.time()
        
        async with xtts_lock:
            # Generate the complete audio
            audio = model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7
            )


            # Convert audio tensor to WAV format in memory
            buffer = io.BytesIO()
            torchaudio.save(buffer, torch.tensor(audio["wav"]).unsqueeze(0), 24000, format="wav")
            buffer.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            print(f"Audio generation completed in {time.time() - t0:.2f} seconds")
            
            return JSONResponse({
                "audio": audio_base64,
                "format": "pcm_f32le",
                "sample_rate": 24000
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/cartesia")
async def generate_tts_cartesia(
    text: str = Query(..., description="Text to convert to speech"),
    config_id: str = Query(..., description="Config ID to use")
):
    """Cartesia endpoint for single audio generation"""
    if not cartesia_client:
        raise HTTPException(status_code=400, detail="Cartesia API is not configured")
        
    try:
        _, _, _, voice_id, _ = agent_manager.get_agent_config(config_id)

        # Generate audio using Cartesia's REST API
        response = await cartesia_client.tts.bytes(
            model_id="sonic",
            transcript=text,
            voice_id=voice_id,
            output_format=cartesia_bytes_format
        )
        
        # Convert to base64
        audio_base64 = base64.b64encode(response).decode('utf-8')
        
        # Return base64 encoded audio data
        return JSONResponse({
            "audio": audio_base64,
            "sample_rate": cartesia_bytes_format["sample_rate"],
            "format": "wav"
        })

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/tts/elevenlabs")
async def generate_tts_elevenlabs(
    text: str = Query(..., description="Text to convert to speech"),
    config_id: str = Query(..., description="Config ID to use")
):
    """ElevenLabs endpoint for single audio generation"""
    try:
        _, _, _, _, elevenlabs_voice_id = agent_manager.get_agent_config(config_id)

        url = f"{API_BASE}/text-to-speech/{elevenlabs_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_flash_v2",
            "output_format": "pcm_32",
            "sample_rate": 24000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"ElevenLabs API error: {error_text}")
                
                audio = await response.read()
                audio_base64 = base64.b64encode(audio).decode('utf-8')
                
                return JSONResponse({
                    "audio": audio_base64,
                    "format": "pcm_f32le",
                    "sample_rate": 24000
                })

    except Exception as e:
        print(f"Error in generate_tts_elevenlabs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
