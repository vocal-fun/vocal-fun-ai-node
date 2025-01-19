from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
import base64
import os

# from tortoise.api import TextToSpeech
# from tortoise.utils.audio import load_voice, load_audio
# import torch
# import torchaudio
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

# tts = TextToSpeech( kv_cache=True, half=True, device='cuda' if torch.cuda.is_available() else 'cpu')


# tts = TextToSpeech(device='cuda' if torch.cuda.is_available() else 'cpu')

# TTS initialization
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/bark").to("cuda")
# 
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")

# from TTS.tts.configs.bark_config import BarkConfig
# from TTS.tts.models.bark import Bark

# config = BarkConfig()
# model = Bark.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/home/n0x/.local/tts/tts_models--multilingual--multi-dataset--bark", eval=True)


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

    # clips_paths = [speaker_wav_path]
    # reference_clips = [load_audio(p, 22050) for p in clips_paths]
    # pcm_audio = tts.tts_with_preset(text, voice_samples=reference_clips, preset='ultra_fast')

    # # Save with correct dimensions [channels, samples]
    # torchaudio.save(
    #     output_path,
    #     pcm_audio.squeeze(0),
    #     24000  # Sample rate
    # )


    # Generate TTS audio
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language="en",
        file_path=output_path,
        split_sentences=False
    )
    
    # tts.tts_with_vc_to_file(
    #     text,
    #     speaker_wav=speaker_wav_path,
    #     # language="en",
    #     file_path=output_path,
    #     speaker="p225"
    # )

    # tts.tts_to_file(text=text,
    #             file_path=output_path,
    #             voice_dir="voices/",
    #             speaker="trump")

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