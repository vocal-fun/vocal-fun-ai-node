from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import time
import torch
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_API_KEY'))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Whisper model
print("Loading Whisper model...")
model = WhisperModel(
    "small.en",
    device="auto",
    compute_type="int8",
    download_root="./models"
)

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        temp_path = f"temp_{audio_file.filename}"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Transcribe audio
        segments, info = model.transcribe(
            temp_path,
            beam_size=5
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
        return {"error": str(e)}