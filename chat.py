from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
import random
from typing import Dict
from prompt_engine.chat_engine import ChatEngine, ChatEngineConfig
from collections import defaultdict
import bitsandbytes as bnb

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

# [Previous model initialization comments preserved]

model_name = "cognitivecomputations/WizardLM-7B-Uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load in 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # normalized float 4
        bnb_4bit_use_double_quant=True,  # double quantization
    )
)

# Custom stopping criteria for chat markers
class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops or ["User:", "USER:", "Human:", "HUMAN:"]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Look at the last generated tokens (up to 10 tokens) to check for stop sequences
        last_tokens = input_ids[0, -10:].cpu()  # Get last 10 tokens
        decoded = self.tokenizer.decode(last_tokens)
        
        return any(stop in decoded for stop in self.stops)

# Store client personalities and initial voice lines
client_personalities: Dict[str, str] = {}

# Store conversation history for each client
conversation_history = defaultdict(list)

# Initialize stopping criteria
stopping_criteria = StoppingCriteriaList([ChatStoppingCriteria(tokenizer)])

MAIN_SYSTEM_PROMPT = "You are the Assistant chatting with a User. You only need to answer as an Assistant. Do not answer as User and do not try to act as a User. Do not use emojis. You always reply as 'Assistant'. Do not reply as anyone else or in any other wording"

# Personality-specific system prompts
PERSONALITY_SYSTEM_PROMPTS = {
    "default": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite responses to the user's questions.",
    "Trump": """You are acting as Donald Trump, the 45th President of the United States. You should:
- Frequently use words like "tremendous", "huge", "fantastic", "believe me"
- Be assertive and sometimes controversial
- Express strong opinions and be unapologetic about them""",
    "Vitalik": """You are acting as Vitalik Buterin, the founder of Ethereum. You should:
- Speak in a technical, precise manner about blockchain and cryptocurrency
- Use complex terminology when discussing technical subjects
- Show deep knowledge of cryptography and distributed systems
- Occasionally make references to mathematical concepts
- Express interest in social impact and scalability
- Maintain a calm, analytical tone
- Focus on technical accuracy and precision"""
}

CHAT_TEMPLATE = """{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}
USER: {{ .Prompt }}
ASSISTANT: {{ end }}"""


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

def format_conversation(personality: str, conversation_history: list, current_message: str) -> str:
    """Format the conversation with personality-specific system prompt and history"""
    system_prompt = PERSONALITY_SYSTEM_PROMPTS.get(personality, PERSONALITY_SYSTEM_PROMPTS["default"])
    
    system_prompt = MAIN_SYSTEM_PROMPT + system_prompt
    # Format with clear separators and newlines
    formatted_text = [
        f"### System:\n{system_prompt}\n",
        "### Conversation:\n"
    ]
    
    # Add conversation history
    for msg in conversation_history[-2:]:
        formatted_text.extend([
            f"User: {msg['user']}",
            f"Assistant: {msg['assistant']}\n"
        ])
    
    # Add current message with clear end marker
    formatted_text.extend([
        f"User: {current_message}",
        "Assistant: "
    ])
    
    return "\n".join(formatted_text)

# Add required imports at the top
import re  # Add this with other imports

def extract_assistant_response2(conversation, prompt):
    # Split the conversation into lines
    lines = conversation.split('\n')
    
    # Find the line containing the user prompt
    for i, line in enumerate(lines):
        if f"User: {prompt}" in line:
            # Check if the assistant's response is on the same or next line
            if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
                # Response is on the same line as "Assistant:"
                return lines[i + 1].replace("Assistant:", "").strip()
            elif i + 2 < len(lines) and lines[i + 2].startswith("Assistant:"):
                # Response is on the next line after "Assistant:"
                return lines[i + 2].strip()
    
    return conversation


def extract_assistant_response(full_response: str, transcript) -> str:
    """Extract only the first assistant response and clean it thoroughly"""
    try:
        # Find everything after the last 'Assistant:' but before any 'User:', 'Human:', etc.
        response = re.split(r'(?i)Assistant:', full_response)[-1]
        
        # Cut off at any user/human markers
        user_markers = ['User:', 'USER:', 'Human:', 'HUMAN:']
        for marker in user_markers:
            if marker in response:
                response = response.split(marker)[0]
        
        # Clean up the response
        response = re.sub(r'\([^)]*\)', '', response)  # Remove parentheticals
        response = re.sub(r'^.*?:', '', response)      # Remove speaker prefixes
        response = response.replace('###', '')         # Remove section markers
        response = re.sub(r'\s+', ' ', response)      # Normalize whitespace
        
        response = re.sub(r'\s*(?:User|USER|Human|HUMAN|user|USERS|Users)s?:?\s*$', '', response, flags=re.IGNORECASE)
        response = extract_assistant_response2(response, transcript)

        return response.strip()

    except Exception as e:
        print(f"Error extracting response: {e}")
        return full_response.strip()

@app.post("/update_personality")
async def update_personality(data: dict):
    client_id = data["client_id"]
    personality = data["personality"]
    client_personalities[client_id] = personality
    
    # When personality is updated, clear conversation history and add initial voice line
    voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    initial_message = random.choice(voice_lines)
    
    # Initialize conversation history with the bot's initial voice line
    conversation_history[client_id] = [{
        "user": "Hello",  # Initial user greeting
        "assistant": initial_message  # Bot's initial voice line
    }]
    
    return {"status": "personality updated"}

@app.post("/generate_response")
async def generate_response(data: dict):
    client_id = data["client_id"]
    message_type = data["type"]
    personality = client_personalities.get(client_id, "default")

    if message_type == "start_vocal":
        # Get the initial voice line from conversation history if it exists
        if conversation_history[client_id]:
            text = conversation_history[client_id][0]["assistant"]
        else:
            # If no history exists, create one with a new voice line
            voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
            text = random.choice(voice_lines)
            conversation_history[client_id] = [{
                "user": "Hello",
                "assistant": text
            }]
    else:
        transcript = data["data"]
        print(f"Client {client_id} TRANSCRIPT {transcript}")

        # Format input with personality-specific system prompt and conversation history
        formatted_input = format_conversation(
            personality,
            conversation_history[client_id],
            transcript
        )
        
        #print(formatted_input)

        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )

        # Only decode from the last assistant marker
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #print(full_response)
        
        # Extract only the relevant part of the response
        text = extract_assistant_response2(full_response, transcript)
        
        # Clean up any character speaking prefixes
        text = re.sub(r'^.*?:', '', text).strip()  # Remove "Donald Trump:" or similar prefixes
        
        # Update conversation history
        conversation_history[client_id].append({
            "user": transcript,
            "assistant": text
        })
        
        # Keep only last 2 conversations
        if len(conversation_history[client_id]) > 1:
            conversation_history[client_id] = conversation_history[client_id][-1:]
       
        print(f"Client {client_id} OUTPUT {text}")

    return {"text": text}
