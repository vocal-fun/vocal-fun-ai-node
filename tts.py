from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import base64
import os
import time
import torch
import torchaudio
import asyncio
import io
from typing import Dict
import uuid
from fastapi.background import BackgroundTasks
from config.agents_config import get_agent_data
from cartesia import Cartesia

ENABLE_LOCAL_MODEL = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize local XTTS model if enabled
if ENABLE_LOCAL_MODEL:
    print("Loading local XTTS model...")
    config = XttsConfig()
    config.load_json("/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=True)
    model.cuda()

    # Cache for voice lines and speaker latents
    voice_lines_cached = {}
    speaker_latents_cache = {}

# Initialize Cartesia client if API key is available
CARTESIA_API_KEY = "sk_car_u_tiwMaJH0qTFtzXB6Shs"
cartesia_client = None
if CARTESIA_API_KEY:
    cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

# Cartesia output format configuration
cartesia_output_format = {
    "container": "wav",
    "encoding": "pcm_f32le",
    "sample_rate": 24000,
}

# Original local model endpoints
async def stream_audio_chunks(websocket: WebSocket, text: str, personality: str):
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })

        voice_samples, random_system_prompt = get_agent_data(personality)
        
        if personality not in speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=voice_samples)
            speaker_latents_cache[personality] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[personality]

        print("Starting streaming inference...")
        t0 = time.time()
        
        chunks = model.inference_stream(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7
        )

        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            
            chunk_tensor = chunk.squeeze().unsqueeze(0).cpu()
            buffer = torch.zeros(1, chunk_tensor.shape[1], dtype=torch.float32)
            buffer[0, :chunk_tensor.shape[1]] = chunk_tensor
            
            temp_path = f"temp_chunk_{i}.wav"
            torchaudio.save(temp_path, buffer, 24000)
            
            with open(temp_path, "rb") as f:
                chunk_bytes = f.read()
            os.remove(temp_path)
            
            chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
            
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk_base64,
                "chunk_id": i,
                "timestamp": time.time()
            })
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

# New Cartesia streaming endpoint
async def stream_audio_chunks_cartesia(websocket: WebSocket, text: str, personality: str):
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })
        
        # Get voice ID for the personality
        voice_samples, _ = get_agent_data(personality)
        
        # Set up the websocket connection with Cartesia
        ws = cartesia_client.tts.websocket()
        
        # Stream the audio using Cartesia's websocket API
        for output in ws.send(
            model_id="sonic-english",
            transcript=text,
            voice_id="41fadb49-adea-45dd-b9b6-4ba14091292d",  # Use the voice ID from your config
            output_format=cartesia_output_format,
            stream=True
        ):
            # Send each chunk to the client
            chunk_base64 = base64.b64encode(output["audio"]).decode("utf-8")
            
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk_base64,
                "chunk_id": time.time(),  # Using timestamp as chunk ID
                "timestamp": time.time()
            })
            
        # Send end of stream message
        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })
        
        ws.close()

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

# Original WebSocket endpoint
@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
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
            personality = data.get("personality", "default")
            
            if text:
                await stream_audio_chunks(websocket, text, personality)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

# New Cartesia WebSocket endpoint
@app.websocket("/tts/stream/cartesia")
async def tts_stream_cartesia(websocket: WebSocket):
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
            personality = data.get("personality", "default")
            
            if text:
                await stream_audio_chunks_cartesia(websocket, text, personality)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Original TTS endpoint
@app.get("/tts")
async def generate_tts(
    text: str = Query(..., description="Text to convert to speech"),
    personality: str = Query("default", description="Voice personality to use")
):
    if not ENABLE_LOCAL_MODEL:
        raise HTTPException(status_code=400, detail="Local model is disabled")
        
    try:
        voice_samples, random_system_prompt = get_agent_data(personality)
        
        if personality not in speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=voice_samples)
            speaker_latents_cache[personality] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[personality]

        print("Starting full audio generation...")
        t0 = time.time()
        
        audio = model.inference(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, torch.tensor(audio["wav"]).unsqueeze(0), 24000, format="wav")
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print(f"Audio generation completed in {time.time() - t0:.2f} seconds")
        
        return JSONResponse({
            "audio": audio_base64,
            "sample_rate": 24000,
            "format": "wav"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New Cartesia TTS endpoint
@app.get("/tts/cartesia")
async def generate_tts_cartesia(
    text: str = Query(..., description="Text to convert to speech"),
    personality: str = Query("default", description="Voice personality to use")
):
    if not cartesia_client:
        raise HTTPException(status_code=400, detail="Cartesia API is not configured")
        
    try:
        # Get voice ID for the personality
        voice_samples, _ = get_agent_data(personality)
        
        # Generate audio using Cartesia's REST API
        response = cartesia_client.tts.bytes(
            model_id="sonic-english",
            transcript=text,
            voice_id="41fadb49-adea-45dd-b9b6-4ba14091292d",  # Use the voice ID from your config
            output_format=cartesia_output_format
        )
        
        # Convert to base64
        audio_base64 = base64.b64encode(response).decode('utf-8')
        
        # Return base64 encoded audio data
        return JSONResponse({
            "audio": audio_base64,
            "sample_rate": cartesia_output_format["sample_rate"],
            "format": "wav"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))