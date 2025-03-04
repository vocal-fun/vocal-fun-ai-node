from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from vllm import LLM, SamplingParams
import random
from typing import Dict, List
from collections import defaultdict
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, List
import asyncio
import aiohttp
import deepspeed
from config.agents_config import get_agent_data
from vllm.entrypoints.chat_utils import load_chat_template
import re
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

# Huggingface login
from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_API_KEY'))

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConversationManager:
    def __init__(self, max_history=1):
        self.history = defaultdict(list)
        self.max_history = max_history
        self.cached_tokens = {}

    def add_conversation(self, session_id: str, user_text: str, assistant_text: str):
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({
            'user': user_text,
            'assistant': assistant_text,
            'tokens': None
        })
        
        if len(self.history[session_id]) > self.max_history:
            self.history[session_id] = self.history[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[dict]:
        return self.history.get(session_id, [])

    def clear_history(self, session_id: str):
        self.history[session_id] = []

# System prompts and configurations
MAIN_SYSTEM_PROMPT = "Please reply in no more than 30 words. "

INITIAL_VOICE_LINES = {}

ENABLE_LOCAL_MODEL = True

if ENABLE_LOCAL_MODEL:
    # Model initialization with vLLM
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Initialize vLLM model
    llm = LLM(
        model=model_name,
        max_num_seqs=256,
        max_model_len=4096,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.9,
        dtype="float16",
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=40,
        presence_penalty=0.0,
        frequency_penalty=1.2,
    )

# Initialize conversation manager
conversation_manager = ConversationManager(max_history=1)

client_selected_personality = {}

def format_conversation(personality: str, conversation_history: list, current_message: str) -> str:
    voice_samples, random_system_prompt, _,  _, _ = get_agent_data(personality)
    system_prompt = random_system_prompt
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

def format_messages(personality: str, conversation_history: list, current_message: str) -> List[Dict[str, str]]:
    """Format conversation history into Groq API message format"""
    voice_samples, random_system_prompt, language, _, _ = get_agent_data(personality)
    system_prompt = random_system_prompt
    system_prompt = MAIN_SYSTEM_PROMPT.replace("{p}", personality) + system_prompt

    if language == "hi":
        system_prompt = system_prompt.replace("Please reply in no more than 30 words. ", "")
        system_prompt = system_prompt + " आपका जवाब हिंदी में होना चाहिए।"
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # Add conversation history
    for msg in conversation_history[-2:]:  # Keep last 2 messages
        messages.extend([
            {"role": "user", "content": msg["user"]},
            {"role": "assistant", "content": msg["assistant"]}
        ])
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages

def check_uncensored(text):
    triggers = [
        "hate speech or discriminatory language",
        "I can't engage",
        "I won't engage"
    ]
    
    responses = [
        "How about you shut the fuck up",
        "Yeah, what else you got on your filthy mind",
        "You should go and ask your mom about that"
    ]

    if any(trigger in text.lower() for trigger in triggers):
        return random.choice(responses)

    return text
    
def remove_emotions(text: str) -> str:
    # Remove text between * * (including the * characters)
    text = re.sub(r'\*[^*]*\*', '', text)
    # Remove text between ( ) (including the () characters)
    text = re.sub(r'\([^)]*\)', '', text)
    # Return the cleaned text
    return text.strip()

@app.post("/chat")
async def generate_response(data: dict):
    start_time = time.time()
    
    session_id = data["session_id"]
    user_message = data["text"]
    personality = data.get("personality", "default")

    print(f"Client {session_id} INPUT {user_message} PERSONALITY {personality}")

    history = conversation_manager.get_history(session_id)

    # if history:
    #     text = history[0]["assistant"]
    # else:
    #     voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    #     text = random.choice(voice_lines)
    #     conversation_manager.add_conversation(session_id, "Hello", text)
        
    transcript = user_message
    print(f"Client {session_id} TRANSCRIPT {transcript}")
    
    max_retries = 3
    retry_count = 0
    text = ""

    while retry_count < max_retries and not text:
        # Format input
        messages = format_messages(
            personality,
            history,
            user_message
        )

        # Generation time
        gen_start = time.time()
        
        # Update sampling parameters with retry-specific temperature
        current_sampling_params = SamplingParams(
            temperature=0.7 + (retry_count * 0.1),
            top_p=0.9,
            top_k=50,
            max_tokens=40,
            presence_penalty=0.0,
            frequency_penalty=1.2,
        )
        
        custom_template = load_chat_template(chat_template="tool_chat_template_llama3.2_json.jinja")

        # Generate with vLLM
        outputs = llm.chat(messages, current_sampling_params)
        print(f"Generation time: {(time.time() - gen_start) * 1000:.2f}ms")

        # Post-processing time
        post_start = time.time()
        full_response = outputs[0].outputs[0].text
        print(f"Full response (attempt {retry_count + 1}): {full_response}")
        
        # text = extract_assistant_response2(full_response, transcript, personality)
        # text = re.sub(r'^.*?:', '', text).strip() if text else ""
        # text = check_uncensored(text)
        # text = remove_emotions(text)

        text = full_response
        
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

    conversation_manager.add_conversation(session_id, transcript, text)
    print(f"Client {session_id} OUTPUT {text}")
    print(f"Total time: {(time.time() - start_time) * 1000:.2f}ms")

    return {"response": text}

@app.post("/chat/groq")
async def generate_response_groq(data: dict):
    start_time = time.time()
    
    # Extract parameters from request
    session_id = data["session_id"]
    user_message = data["text"]
    personality = data.get("personality", "default")
    
    print(f"Client {session_id} INPUT {user_message} PERSONALITY {personality}")
    
    # Get conversation history
    history = conversation_manager.get_history(session_id)
    
    # if not history:
    #     voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    #     initial_response = random.choice(voice_lines)
    #     conversation_manager.add_conversation(session_id, "Hello", initial_response)
    #     history = conversation_manager.get_history(session_id)
    
    # Format messages for Groq API
    messages = format_messages(
        personality,
        history,
        user_message
    )
    
    # Groq API configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Choose the model
    model = "mixtral-8x7b-32768"  # Using the model from your example
    model = "gemma2-9b-it"
    # model = "llama-3.3-70b-versatile"
    model = "llama3-70b-8192"
    # model = "llama3-8b-8192"

    print(messages)

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 90,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.6
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Groq API error: {error_text}")
                    raise HTTPException(status_code=response.status, detail="Error from Groq API")
                
                response_data = await response.json()
                text = response_data["choices"][0]["message"]["content"]
                text = check_uncensored(text)
                text = remove_emotions(text)
                
                # Clean up the response
                # text = remove_emojis(text) if text else ""
                # text = re.sub(r'^.*?:', '', text).strip() if text else ""
                
                if not text:
                    fallback_responses = [
                        "I understand what you're saying. Could you rephrase that?",
                        "That's an interesting point. Could you elaborate?",
                        "I see what you mean. Let's explore that further.",
                    ]
                    text = random.choice(fallback_responses)
                    print("Using fallback response:", text)
                
                # Update conversation history
                conversation_manager.add_conversation(session_id, user_message, text)
                
                print(f"Client {session_id} OUTPUT {text}")
                print(f"Total time: {(time.time() - start_time) * 1000:.2f}ms")
                
                return {"response": text}
                
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request to Groq API timed out")
        except aiohttp.ClientError as e:
            print(f"aiohttp client error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))