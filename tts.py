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
from typing import Dict
import uuid
from fastapi.background import BackgroundTasks
from config.agents_config import get_agent_data
from cartesia import AsyncCartesia

# Configuration
ENABLE_LOCAL_MODEL = False  # Set to False to disable local model
CARTESIA_API_KEY = 'sk_car_u_tiwMaJH0qTFtzXB6Shs'
DEFAULT_VOICE_ID = '41fadb49-adea-45dd-b9b6-4ba14091292d'

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
    print("Loading local model...")
    config = XttsConfig()
    config.load_json("/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=True)
    model.cuda()

    # Cache for voice lines and speaker latents
    voice_lines_cached = {}
    speaker_latents_cache = {}

# Initialize Cartesia client
cartesia_client = AsyncCartesia(api_key=CARTESIA_API_KEY)

# Cartesia output formats
cartesia_stream_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 24000,
}

cartesia_bytes_format = {
    "container": "wav",
    "encoding": "pcm_f32le",
    "sample_rate": 24000,
}

async def stream_audio_chunks(websocket: WebSocket, text: str, personality: str):
    """Original XTTS streaming implementation"""
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
            
            # Convert chunk to bytes
            chunk_tensor = chunk.squeeze().unsqueeze(0).cpu()
            buffer = torch.zeros(1, chunk_tensor.shape[1], dtype=torch.float32)
            buffer[0, :chunk_tensor.shape[1]] = chunk_tensor
            
            temp_path = f"temp_chunk_{i}.wav"
            torchaudio.save(temp_path, buffer, 24000)
            
            with open(temp_path, "rb") as f:
                chunk_bytes = f.read()
            os.remove(temp_path)
            
            chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
            
            # Send chunk and wait for small delay to prevent overwhelming the connection
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": chunk_base64,
                "chunk_id": i,
                "timestamp": time.time()
            })
            await asyncio.sleep(0.01)  # Small delay between chunks

        # Send end of stream message
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

async def stream_audio_chunks_cartesia(websocket: WebSocket, text: str, personality: str):
    """Refined Cartesia streaming implementation with crossfading and artifact reduction"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })
        
        t0 = time.time()
        session_id = str(uuid.uuid4())
        
        ws = await cartesia_client.tts.websocket()
        
        stream = await ws.send(
            model_id="sonic",
            transcript=text,
            voice_id=DEFAULT_VOICE_ID,
            stream=True,
            output_format=cartesia_stream_format
        )
        
        # Increased overlap and chunk size for smoother transitions
        chunk_size = 8192  # Larger chunk size
        overlap = 512      # Larger overlap for smoother crossfade
        
        # Buffer for maintaining continuous audio
        audio_buffer = np.array([], dtype=np.float32)
        chunk_counter = 0
        
        # Hanning window for crossfade
        fade_in = np.hanning(2 * overlap)[:overlap]
        fade_out = np.hanning(2 * overlap)[overlap:]
        
        def apply_crossfade(chunk, is_first=False, is_last=False):
            """Apply crossfade to chunk"""
            if len(chunk) < 2 * overlap:
                return chunk
                
            result = chunk.copy()
            
            if not is_first:
                result[:overlap] *= fade_in
                
            if not is_last:
                result[-overlap:] *= fade_out
                
            return result
        
        async for output in stream:
            if chunk_counter == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            
            # Convert incoming audio to numpy array with proper normalization
            new_audio = np.frombuffer(output["audio"], dtype=np.float32)
            
            # Apply mild compression to reduce dynamic range
            threshold = 0.8
            ratio = 0.5
            mask = np.abs(new_audio) > threshold
            new_audio[mask] = threshold + (np.abs(new_audio[mask]) - threshold) * ratio * np.sign(new_audio[mask])
            
            # Append to buffer
            audio_buffer = np.append(audio_buffer, new_audio)
            
            while len(audio_buffer) >= chunk_size + overlap:
                # Extract chunk with overlap
                current_chunk = audio_buffer[:chunk_size + overlap]
                
                # Apply crossfade
                is_first = (chunk_counter == 0)
                processed_chunk = apply_crossfade(current_chunk, is_first=is_first, is_last=False)
                
                # Convert to tensor with proper shape
                chunk_tensor = torch.from_numpy(processed_chunk).unsqueeze(0)
                
                # Ensure audio is in valid range
                chunk_tensor = torch.clamp(chunk_tensor, -1.0, 1.0)
                
                # Save as WAV with specific format
                temp_path = f"temp_chunk_{session_id}_{chunk_counter}.wav"
                torchaudio.save(
                    temp_path,
                    chunk_tensor,
                    cartesia_stream_format["sample_rate"],
                    encoding="PCM_F",
                    bits_per_sample=32
                )
                
                try:
                    with open(temp_path, "rb") as f:
                        chunk_bytes = f.read()
                    chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "chunk": chunk_base64,
                        "chunk_id": chunk_counter,
                        "timestamp": time.time()
                    })
                    
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Remove processed audio from buffer (keeping overlap for next chunk)
                audio_buffer = audio_buffer[chunk_size:]
                chunk_counter += 1
                
                # Adjusted delay based on chunk size
                await asyncio.sleep(chunk_size / cartesia_stream_format["sample_rate"] / 4)
        
        # Process any remaining audio in the buffer
        if len(audio_buffer) > overlap:
            final_chunk = apply_crossfade(audio_buffer, is_first=False, is_last=True)
            chunk_tensor = torch.from_numpy(final_chunk).unsqueeze(0)
            chunk_tensor = torch.clamp(chunk_tensor, -1.0, 1.0)
            
            temp_path = f"temp_chunk_{session_id}_final.wav"
            torchaudio.save(
                temp_path,
                chunk_tensor,
                cartesia_stream_format["sample_rate"],
                encoding="PCM_F",
                bits_per_sample=32
            )
            
            try:
                with open(temp_path, "rb") as f:
                    chunk_bytes = f.read()
                chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_base64,
                    "chunk_id": chunk_counter,
                    "timestamp": time.time()
                })
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })
        
        await ws.close()

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
            personality = data.get("personality", "default")
            
            if text:
                await stream_audio_chunks(websocket, text, personality)
            
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
            personality = data.get("personality", "default")
            
            if text:
                await stream_audio_chunks_cartesia(websocket, text, personality)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/tts")
async def generate_tts(
    text: str = Query(..., description="Text to convert to speech"),
    personality: str = Query("default", description="Voice personality to use")
):
    """Original XTTS endpoint for single audio generation"""
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
        
        # Generate the complete audio
        audio = model.inference(
            text,
            "en",
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
            "sample_rate": 24000,
            "format": "wav"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/cartesia")
async def generate_tts_cartesia(
    text: str = Query(..., description="Text to convert to speech"),
    personality: str = Query("default", description="Voice personality to use")
):
    """Cartesia endpoint for single audio generation"""
    if not cartesia_client:
        raise HTTPException(status_code=400, detail="Cartesia API is not configured")
        
    try:
        # Generate audio using Cartesia's REST API
        response = await cartesia_client.tts.bytes(
            model_id="sonic-english",
            transcript=text,
            voice_id=DEFAULT_VOICE_ID,
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
        raise HTTPException(status_code=500, detail=str(e))