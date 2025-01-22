from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
import base64
import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# from tortoise.api import TextToSpeech
# from tortoise.utils.audio import load_voice, load_audio
# import torch
# import torchaudio.transforms as T
# import numpy as np

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
model.load_checkpoint(config, checkpoint_dir="/home/ec2-user/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=False)
model.cuda()


PERSONALITY_MAP = {
    "default": "voices/trump.wav",
    "Vitalik": "voices/vitalik.wav",
    "Trump": "voices/trump.wav",
    "Elon Musk": "voices/vitalik.wav"
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

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav_path)

    print("Inference...")
    t0 = time.time()


    print("Inference...")
    out = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    print(f"Time to first chunck: {time.time() - t0}")
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
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