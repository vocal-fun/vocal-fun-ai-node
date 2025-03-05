from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import time
import torch
from config.agents_config import agent_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Whisper model
print("Loading Whisper model...")
model = WhisperModel(
    "small",
    device="auto",
    compute_type="int8",
    download_root="./models"
)

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="The audio file to transcribe"),
    config_id: str = Form(..., description="The configuration ID for the agent")
):
    try:
        start_time = time.time()
        
        # Validate config_id and get agent configuration
        if not config_id:
            return {"error": "config_id is required"}
            
        config = agent_manager.get_agent_config(config_id)
        if not config:
            return {"error": f"No configuration found for config_id: {config_id}"}
            
        _, _, language, _, _ = config
        
        # Save uploaded file temporarily
        temp_path = f"temp_{audio_file.filename}"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        print(f"Transcribing audio with language: {language}")
        # Transcribe audio
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            language=language
        )
        
        transcribed_text = " ".join([segment.text for segment in segments])
        
        # Clean up temp file
        import os
        os.remove(temp_path)
        
        end_time = time.time()
        print(f"Transcription completed in {end_time - start_time:.2f} seconds")
        
        return {
            "text": transcribed_text,
            "processing_time": end_time - start_time
        }
        
    except Exception as e:
        import traceback
        print(f"Error in transcribe_audio: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}