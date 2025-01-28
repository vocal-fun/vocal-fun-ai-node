from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fastapi.responses import FileResponse
import base64
import os
import time
import torch
import torchaudio
import asyncio
from typing import Dict
import uuid


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
config = XttsConfig()
config.load_json("/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=True)
model.cuda()

PERSONALITY_MAP = {
    "default": "voices/trump.wav",
    "Vitalik": "voices/vitalik.wav",
    "Trump": "voices/trump.wav",
    "Elon Musk": "voices/vitalik.wav"
}

# Cache for voice lines and speaker latents
voice_lines_cached = {}
speaker_latents_cache = {}

async def stream_audio_chunks(websocket: WebSocket, text: str, personality: str):
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })

        speaker_wav_path = PERSONALITY_MAP.get(personality, PERSONALITY_MAP["default"])
        
        # Get or compute speaker latents
        if personality not in speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav_path)
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
            
            print(f"Sent chunk {i} of audio length {chunk.shape[-1]}")

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

@app.websocket("/tts_stream")
async def tts_stream(websocket: WebSocket):
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


@app.get("/tts")
async def generate_tts(
    text: str = Query(..., description="Text to convert to speech"),
    personality: str = Query("default", description="Voice personality to use")
):
    try:
        # Get the appropriate voice file path
        speaker_wav_path = PERSONALITY_MAP.get(personality, PERSONALITY_MAP["default"])
        
        # Get or compute speaker latents
        if personality not in speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav_path)
            speaker_latents_cache[personality] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[personality]

        # Generate the full audio
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

        # Create a unique filename for this generation
        filename = f"tts_output_{uuid.uuid4()}.wav"
        
        # Save the audio to a temporary file
        torchaudio.save(filename, audio.squeeze().unsqueeze(0).cpu(), 24000)
        
        print(f"Audio generation completed in {time.time() - t0:.2f} seconds")
        
        # Return the file and ensure it's deleted after sending
        return FileResponse(
            filename,
            media_type="audio/wav",
            filename="tts_output.wav",
            background=BackgroundTask(lambda: os.remove(filename))
        )

    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(status_code=500, detail=str(e))