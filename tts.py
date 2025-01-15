from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS initialization
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

PERSONALITY_MAP = {
    "default": "voices/trump.wav",
    "Vitalik": "voices/vitalik.wav",
    "Trump": "voices/trump.wav",
    "Elon Musk": "voices/trump.wav"
}

# Cache for voice lines
voice_lines_cached = {}

@app.post("/generate_audio")
async def generate_audio(data: dict):
    client_id = data["client_id"]
    text = data["text"]
    personality = data["personality"]
    message_type = data["message_type"]
    
    speaker_wav_path = PERSONALITY_MAP.get(personality, PERSONALITY_MAP["default"])
    
    # Check cache for start_vocal messages
    if message_type == "start_vocal" and text in voice_lines_cached:
        return {"audio_base64": voice_lines_cached[text]}
        
    output_path = f"response_{client_id}.wav"
    
    # Generate TTS audio
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language="en",
        file_path=output_path
    )
    
    # Read and encode audio
    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # Cache if it's a start_vocal message
    if message_type == "start_vocal":
        voice_lines_cached[text] = audio_base64
        
    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)
        
    return {"audio_base64": audio_base64}