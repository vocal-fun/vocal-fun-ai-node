from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
import random
from typing import Dict
from prompt_engine.chat_engine import ChatEngine, ChatEngineConfig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from huggingface_hub import login
login(token="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")

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "openlm-research/open_llama_3b"
# model_name = "mistralai/Mistral-7B-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )


# model_name = "mistralai/Mistral-7B-v0.1"  # Use the Mistral model name
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,  device_map="auto")

# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# model_name = "google/flan-t5-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# model_name = "EleutherAI/gpt-j-6B"
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# model_name = "EleutherAI/gpt-neo-2.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

model_name = "cognitivecomputations/WizardLM-7B-Uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

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
        print(f"Client {client_id} TRANSCRIPT {transcript}")

        
        config = ChatEngineConfig(
            user_name = "Vocal User",
            bot_name = personality
        )

        # description = "A conversation with Trump and a Vocal User"

        # chat_engine = ChatEngine(config, description, [])
        # user_query = transcript
        # prompt = chat_engine.build_prompt(user_query)
        # transcript = prompt

        prompt = transcript + "\n\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,  # Prevents repeated trigrams
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )



        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text.replace(prompt, "").strip()

        sentences = text.split(". ")  # Split into sentences
        text = ". ".join(sentences[:2]).strip()  # Take the first two sentences
        if not text.endswith("."):
            text += "."  # Ensure it ends with a period

        sentences = text.split("\n")  # Split into sentences
        text = ". ".join(sentences[:2]).strip()  # Take the first two sentences
        if not text.endswith("."):
            text += "."  # Ensure it ends with a period

        # text = text[len(transcript):].strip()
        # text = text.split(".")[0]
        print(f"Client {client_id} OUTPUT {text}")

    return {"text": text}