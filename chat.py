from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openlm-research/open_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float16",
    device_map="auto"
)

# Store client personalities and initial voice lines
client_personalities: Dict[str, str] = {}

INITIAL_VOICE_LINES = {
    "default": ["Hello there! How can I assist you today?"],
    "Trump": [
        "Muslims are the real shit!",
        "Make America great again!!!"
    ],
    "Vitalik": [
        "Ultrasound money for the win!",
        "Make Ethereum great again!!!",
    ]
}

@app.post("/update_personality")
async def update_personality(data: dict):
    client_id = data["client_id"]
    personality = data["personality"]
    client_personalities[client_id] = personality
    return {"status": "personality updated"}

@app.post("/generate_response")
async def generate_response(data: dict):
    client_id = data["client_id"]
    message_type = data["type"]
    personality = client_personalities.get(client_id, "default")

    if message_type == "start_vocal":
        voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
        text = random.choice(voice_lines)
    else:
        transcript = data["data"]
        inputs = tokenizer(transcript, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"text": text}