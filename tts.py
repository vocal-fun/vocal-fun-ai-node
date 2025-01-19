from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import base64
import os
import logging
from typing import Dict, Tuple
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSONALITY_MAP = {
    "default": "voices/trump.wav",
    "Vitalik": "voices/vitalik.wav",
    "Trump": "voices/trump.wav",
    "Elon Musk": "voices/vitalik.wav"
}

class TTSManager:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.speaker_embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.voice_lines_cached: Dict[str, str] = {}
        
        # Initialize model
        logger.info("Loading XTTS model...")
        self.config = XttsConfig()
        self.config.load_json(str(self.model_path / "config.json"))

       # Initialize model with DeepSpeed
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            config=self.config,
            checkpoint_dir=str(self.model_path),
            use_deepspeed=False
        )

        self.model.cuda()

        # torch.set_float32_matmul_precision('high')
        logger.info("Model loaded successfully with DeepSpeed")

    @torch.no_grad()
    def get_speaker_embedding(self, speaker_wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute speaker embedding for a given WAV file."""
        if speaker_wav_path not in self.speaker_embeddings:
            logger.info(f"Computing speaker embedding for {speaker_wav_path}")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[speaker_wav_path],
                max_ref_length=65_536,
                sound_norm_refs=True
            )
            # Store tensors in half precision
            self.speaker_embeddings[speaker_wav_path] = (
                gpt_cond_latent, 
                speaker_embedding
            )
            logger.info(f"Speaker embedding computed for {speaker_wav_path}")
        
        return self.speaker_embeddings[speaker_wav_path]

    @torch.no_grad()
    async def generate_audio(self, text: str, speaker_wav_path: str) -> str:
        """Generate audio and return base64 encoded string."""
        try:
            # Get speaker embedding (cached)
            gpt_cond_latent, speaker_embedding = self.get_speaker_embedding(speaker_wav_path)
            
            # Generate audio
            logger.info("Generating audio...")
            with torch.cuda.amp.autocast(enabled=True):
                out = self.model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.7,
                    length_penalty=1.0,
                    repetition_penalty=2.0,
                    top_k=50,
                    top_p=0.8,
                    enable_text_splitting=True
                )

            # Save temporarily and convert to base64
            temp_path = f"temp_{torch.rand(1)[0]}.wav"
            torchaudio.save(
                temp_path,
                torch.tensor(out["wav"]).unsqueeze(0),
                24000
            )

            # Read and encode
            with open(temp_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Cleanup
            os.remove(temp_path)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            return audio_base64

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS Manager
MODEL_PATH = "/home/naman/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"  # Update this path
tts_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and pre-compute speaker embeddings on startup."""
    global tts_manager
    
    logger.info("Initializing TTS Manager...")
    tts_manager = TTSManager(MODEL_PATH)
    
    logger.info("Pre-computing speaker embeddings...")
    for speaker_wav_path in set(PERSONALITY_MAP.values()):
        tts_manager.get_speaker_embedding(speaker_wav_path)
    logger.info("Speaker embeddings pre-computed")

@app.post("/generate_audio")
async def generate_audio(data: dict):
    """Generate audio endpoint."""
    try:
        text = data["text"]
        personality = data["personality"]
        message_type = data.get("message_type", "")
        
        speaker_wav_path = PERSONALITY_MAP.get(personality, PERSONALITY_MAP["default"])
        
        # Generate audio
        audio_base64 = await tts_manager.generate_audio(text, speaker_wav_path)
        return {"audio_base64": audio_base64}

    except Exception as e:
        logger.error(f"Error in generate_audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
