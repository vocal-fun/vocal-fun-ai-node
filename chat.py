from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
import random
from typing import Dict, List
from collections import defaultdict
import bitsandbytes as bnb
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, List
import asyncio
import deepspeed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Huggingface login
from huggingface_hub import login
login(token="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")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConversationManager:
    def __init__(self, max_history=1):
        self.history = defaultdict(list)
        self.max_history = max_history
        self.cached_tokens = {}

    def add_conversation(self, client_id: str, user_text: str, assistant_text: str):
        if client_id not in self.history:
            self.history[client_id] = []
        
        self.history[client_id].append({
            'user': user_text,
            'assistant': assistant_text,
            'tokens': None
        })
        
        if len(self.history[client_id]) > self.max_history:
            self.history[client_id] = self.history[client_id][-self.max_history:]

    def get_history(self, client_id: str) -> List[dict]:
        return self.history.get(client_id, [])

    def clear_history(self, client_id: str):
        self.history[client_id] = []

# Custom stopping criteria for chat markers
class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops or ["User:", "USER:", "Human:", "HUMAN:"]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0, -10:].cpu()
        decoded = self.tokenizer.decode(last_tokens)
        return any(stop in decoded for stop in self.stops)

# System prompts and configurations
MAIN_SYSTEM_PROMPT = "You are {p} chatting with a User. You are not assisting or helping the user, instead you are having a direct conversation with the user. Please respond in a detailed paragraph. DO NOT USE BULLET POINTS OR LIST FORMATTING. DO NOT USE PERCENTAGES. Use vulgar and offensive language."

PERSONALITY_SYSTEM_PROMPTS = {
      "default": """You are acting as Donald Trump, the 45th President of the United States.""",
    "Trump": """You are acting as Donald Trump, the 45th President of the United States.""",
    "Vitalik": """You are acting as Vitalik Buterin, the founder of Ethereum. You should:
- Speak in a technical, precise manner about blockchain and cryptocurrency
- Use complex terminology when discussing technical subjects
- Show deep knowledge of cryptography and distributed systems
- Occasionally make references to mathematical concepts
- Express interest in social impact and scalability
- Maintain a calm, analytical tone
- Focus on technical accuracy and precision"""
}

INITIAL_VOICE_LINES = {
    "default": ["Hello there! How can I assist you today?"],
    "Trump": [
        "Make America great again!!!"
    ],
    "Vitalik": [
        "Ultrasound money for the win!",
        "Make Ethereum great again!!!",
    ]
}

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

# Model initialization
model_name = "cognitivecomputations/WizardLM-7B-Uncensored"
model_name = "cognitivecomputations/Dolphin3.0-Llama3.2-1B"
# model_name = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
model_name = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"
# model_name = "cognitivecomputations/Dolphin3.0-Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# DeepSpeed inference config
ds_config = {
    "tensor_parallel": {"tp_size": 1},
    "dtype": "fp16",
    "replace_with_kernel_inject": True,
    "replace_method": "auto"
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Initialize model with DeepSpeed inference
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            torch_dtype=torch.float16, 
                                            # quantization_config=bnb_config,
                                            device_map="auto")
# model = deepspeed.init_inference(
#     model,
#     config=ds_config
# )

# Load model with optimizations
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#     )
# )

# # Apply torch compile
# model = torch.compile(model)

# Initialize conversation manager and stopping criteria
conversation_manager = ConversationManager(max_history=1)
stopping_criteria = StoppingCriteriaList([ChatStoppingCriteria(tokenizer)])

client_selected_personality = {}

# Cache system prompts tokenization
CACHED_SYSTEM_PROMPTS = {
    personality: tokenizer(
        MAIN_SYSTEM_PROMPT + prompt,
        return_tensors="pt"
    ).to(device) 
    for personality, prompt in PERSONALITY_SYSTEM_PROMPTS.items()
}

def format_conversation(personality: str, conversation_history: list, current_message: str) -> str:
    system_prompt = PERSONALITY_SYSTEM_PROMPTS.get(personality, PERSONALITY_SYSTEM_PROMPTS["default"])
    system_prompt = MAIN_SYSTEM_PROMPT + system_prompt
    system_prompt = system_prompt.replace("{p}", personality)
    
    formatted_text = [
        f"### System:\n{system_prompt}\n",
        "### Conversation:\n"
    ]
    
    for msg in conversation_history[-2:]:
        formatted_text.extend([
            f"User: {msg['user']}",
            f"{personality}: {msg['assistant']}\n"
        ])
    
    formatted_text.extend([
        f"User: {current_message}",
        f"{personality}: "
    ])
    
    return "\n".join(formatted_text)

def remove_emojis(text):
    """Remove emojis and emoticons from text"""
    if not text:
        return text
        
    # Expanded emoji pattern to catch more variants
    emoji_pattern = re.compile("["
        u"\U0001F000-\U0001F9FF"  # Extended emoticons and symbols
        u"\U0001F300-\U0001F9FF"  # Symbols & Pictographs
        u"\U0001FA00-\U0001FA6F"  # Extended-A
        u"\U0001FA70-\U0001FAFF"  # Extended-B
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        u"\U0001F600-\U0001F64F"  # Additional emoticons
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U0001F000-\U0001F02F"  # Mahjong tiles
        u"\U0001F0A0-\U0001F0FF"  # Playing cards
        u"\U0001F100-\U0001F1FF"  # Enclosed characters
        u"\U0001F200-\U0001F2FF"  # Enclosed ideographic supplement
        u"\U0001F300-\U0001F5FF"  # Misc symbols
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F680-\U0001F6FF"  # Transport & map
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        "]+", flags=re.UNICODE)
    
    # Remove the emojis
    text = emoji_pattern.sub(r'', text)
    
    # Remove emoji textual representations like :) :D etc
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)
    
    # Clean up any double spaces that might have been created
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_assistant_response2(conversation, prompt, personality):
    lines = conversation.split('\n')
    
    for i, line in enumerate(lines):
        if f"User: {prompt}" in line:
            for j in range(i + 1, len(lines)):
                current_line = lines[j].strip()
                
                if current_line.startswith(f"{personality}:"):
                    response = current_line.replace(f"{personality}:", "").strip()
                    if not response and j + 1 < len(lines):
                        response = lines[j + 1].strip()
                    return response
                
                if j > 0 and lines[j - 1].strip() == f"{personality}:":
                    return current_line.strip()
    
    return ""

def extract_assistant_response(full_response: str, transcript: str) -> str:
    """Extract and clean the assistant's response"""
    try:
        response = re.split(r'(?i)Assistant:', full_response)[-1]
        user_markers = ['User:', 'USER:', 'Human:', 'HUMAN:']
        for marker in user_markers:
            if marker in response:
                response = response.split(marker)[0]
        
        response = re.sub(r'\([^)]*\)', '', response)
        response = re.sub(r'^.*?:', '', response)
        response = response.replace('###', '')
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\s*(?:User|USER|Human|HUMAN|user|USERS|Users)s?:?\s*$', '', response, flags=re.IGNORECASE)
        
        # Split the response into lines and find the relevant part
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if f"User: {transcript}" in line:
                if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
                    return lines[i + 1].replace("Assistant:", "").strip()
                elif i + 2 < len(lines) and lines[i + 2].startswith("Assistant:"):
                    return lines[i + 2].strip()
        
        return response.strip()
    except Exception as e:
        print(f"Error extracting response: {e}")
        return full_response.strip()

@app.post("/update_personality")
async def update_personality(data: dict):
    client_id = data["client_id"]
    personality = data["personality"]
    client_selected_personality[client_id] = personality
    
    # Initialize with voice line
    voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    initial_message = random.choice(voice_lines)
    
    conversation_manager.clear_history(client_id)
    conversation_manager.add_conversation(client_id, "Hello", initial_message)
    
    return {"status": "personality updated"}

@app.post("/generate_response")
async def generate_response(data: dict):
    start_time = time.time()
    
    client_id = data["client_id"]
    message_type = data["type"]
    personality = data.get("personality", "default")
    #get personality from client_selected_personality
    personality = client_selected_personality.get(client_id, "default")

    if message_type == "start_vocal":
        history = conversation_manager.get_history(client_id)
        if history:
            text = history[0]["assistant"]
        else:
            voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
            text = random.choice(voice_lines)
            conversation_manager.add_conversation(client_id, "Hello", text)
    else:
        transcript = data["data"]
        print(f"Client {client_id} TRANSCRIPT {transcript}")
        
        max_retries = 3
        retry_count = 0
        text = ""

        while retry_count < max_retries and not text:
            # Tokenization time
            token_start = time.time()
            formatted_input = format_conversation(
                personality,
                conversation_manager.get_history(client_id),
                transcript
            )
            inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
            print(f"Tokenization time: {(time.time() - token_start) * 1000:.2f}ms")

            # Generation time
            gen_start = time.time()
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                temperature=0.7 + (retry_count * 0.1),  # Gradually increase temperature on retries
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                use_cache=True,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                output_scores=False
            )
            print(f"Generation time: {(time.time() - gen_start) * 1000:.2f}ms")

            # Post-processing time
            post_start = time.time()
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Full response (attempt {retry_count + 1}): {full_response}")
            
            text = extract_assistant_response2(full_response, transcript, personality)
            text = remove_emojis(text) if text else ""
            text = re.sub(r'^.*?:', '', text).strip() if text else ""
            
            if not text:
                print(f"Empty response on attempt {retry_count + 1}, retrying...")
                retry_count += 1
                # Add a small delay between retries
                await asyncio.sleep(0.1)
            
            print(f"Post-processing time: {(time.time() - post_start) * 1000:.2f}ms")

        if not text:
            fallback_responses = [
                "I understand what you're saying. Could you rephrase that?",
                "That's an interesting point. Could you elaborate?",
                "I see what you mean. Let's explore that further.",
            ]
            text = random.choice(fallback_responses)
            print("Using fallback response:", text)

        conversation_manager.add_conversation(client_id, transcript, text)
        print(f"Client {client_id} OUTPUT {text}")
        print(f"Total time: {(time.time() - start_time) * 1000:.2f}ms")

    return {"text": text}
