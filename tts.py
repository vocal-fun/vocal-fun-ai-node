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
    """Improved Cartesia streaming implementation with audio artifact prevention"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })
        
        t0 = time.time()
        session_id = str(uuid.uuid4())
        
        # Set up the websocket connection with Cartesia
        ws = await cartesia_client.tts.websocket()
        
        # Send the request and get the stream
        stream = await ws.send(
            model_id="sonic-english",
            transcript=text,
            voice_embedding=[-0.12159855,-0.16414514,-0.049289625,0.024538275,-0.072236136,0.006082801,0.053433947,-0.1006353,0.009231852,-0.055021614,0.034683794,0.024907935,-0.0676292,0.02398274,-0.09046795,0.05024385,-0.062617354,0.03084272,0.11106175,-0.06436039,-0.031746525,-0.01460289,0.105050705,0.0014357729,-0.12628633,0.09844303,-0.00053568726,-0.053918533,-0.019285263,-0.052566208,0.06893156,-0.049484596,-0.26974356,-0.07786751,0.016459487,0.06404519,-0.037883166,-0.003842584,-0.111631036,-0.0682958,-0.036091365,-0.08779326,-0.028243529,0.028323144,-0.021334462,-0.07293995,-0.0060823364,-0.097915575,0.08429136,-0.0039670193,-0.030126853,-0.091565974,-0.05903095,0.037992723,-0.022826029,-0.02593449,-0.03424717,-0.04410112,0.03534066,-0.008708731,-0.081188455,0.030807598,-0.029375056,-0.038631063,0.08203755,0.061721254,0.05475549,-0.10717895,0.06740241,0.07599962,0.014547937,-0.22690666,-0.08059945,0.055012528,-0.05252478,0.03569752,0.024765007,0.03339556,-0.04082503,-0.0067585087,0.0051338435,-0.05794202,-0.09217479,-0.060701847,0.016225984,-0.082952924,0.007818717,0.038651887,-0.03200712,0.11985133,0.07922752,-0.043333676,0.011892968,0.11349464,0.09614831,-0.06849895,-0.021431064,0.067313075,-0.018504832,0.08171706,-0.06569794,0.053314827,0.0031529102,0.13392411,0.0040785754,-0.05841033,-0.010383795,0.018345023,0.07015268,0.031878542,0.064817965,0.16061665,-0.07509655,0.013290255,-0.06353553,-0.019901352,-0.07320861,0.09904219,-0.0027910226,-0.09746932,0.009839478,-0.10808914,0.043036025,-0.030305758,-0.0048883962,-0.0003702509,0.04117832,-0.03243669,-0.116793595,0.04845231,0.18178385,0.0073654153,0.034061555,0.051631965,-0.020853411,0.18224737,0.028355792,-0.032823905,0.018870115,0.001596594,-0.07172068,0.082375675,0.0013353196,-0.008104939,0.13129027,-0.03062857,-0.05511648,-0.021653863,0.13094826,0.019921616,-0.04515059,-0.029037934,-0.08298065,0.043641847,0.05264099,-0.036533907,-0.021005219,0.018919528,-0.061981898,0.16558543,0.00094018556,0.05974994,-0.104549974,0.025388904,-0.060925983,0.083641686,0.1153267,-0.03630522,-0.0026039085,-0.07018522,0.113147385,0.11539205,0.015328968,0.04383713,-0.0431801,0.036598556,-0.09447776,0.05670758,0.03253327,-0.12861596,0.1726538,0.063807145,0.032155845,-0.05638192,0.045402437,-0.029928694,0.05093372,-0.04930843,-0.13913435,0.10534002,0.0062249303,-0.004737596],
            stream=True,
            output_format=cartesia_stream_format
        )
        
        # Buffer for maintaining continuous audio
        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = 4096  # Adjust this value based on your needs
        overlap = 128      # Small overlap to prevent discontinuities
        chunk_counter = 0
        
        async for output in stream:
            if chunk_counter == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            
            # Convert incoming audio to numpy array
            new_audio = np.frombuffer(output["audio"], dtype=np.float32)
            
            # Append to buffer
            audio_buffer = np.append(audio_buffer, new_audio)
            
            # Process complete chunks from the buffer
            while len(audio_buffer) >= chunk_size:
                # Extract chunk with overlap
                chunk = audio_buffer[:chunk_size + overlap]
                
                # Apply fade out to the end of the chunk to smooth transition
                if len(chunk) > overlap:
                    fade_out = np.linspace(1, 0, overlap)
                    chunk[-overlap:] *= fade_out
                
                # Convert to tensor
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
                
                # Save as WAV in memory
                temp_path = f"temp_chunk_{session_id}_{chunk_counter}.wav"
                torchaudio.save(temp_path, chunk_tensor, cartesia_stream_format["sample_rate"])
                
                try:
                    # Read and encode chunk
                    with open(temp_path, "rb") as f:
                        chunk_bytes = f.read()
                    chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                    
                    # Send to client
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "chunk": chunk_base64,
                        "chunk_id": chunk_counter,
                        "timestamp": time.time()
                    })
                    
                finally:
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Remove processed audio from buffer (keeping overlap)
                audio_buffer = audio_buffer[chunk_size:]
                chunk_counter += 1
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
        
        # Process any remaining audio in the buffer
        if len(audio_buffer) > 0:
            chunk_tensor = torch.from_numpy(audio_buffer).unsqueeze(0)
            temp_path = f"temp_chunk_{session_id}_final.wav"
            torchaudio.save(temp_path, chunk_tensor, cartesia_stream_format["sample_rate"])
            
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

        # Send end of stream message
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