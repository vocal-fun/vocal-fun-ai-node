from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS
import base64
import json
import uuid
import torch
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model (GPT-style) for text generation
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# from huggingface_hub import login
# login(token="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")

# model_name = "mistralai/Mistral-7B-v0.1"  # Use the Mistral model name
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16").to(device)

# Load TTS model for speech synthesis
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Store connected clients
clients = {}

client_personalities = {}

PERSONALITY_MAP = {
    "default": "voices/trump.wav",
    "Vitalik": "voices/vitalik.wav",
    "Trump": "voices/trump.wav",
    "Elon Musk": "voices/trump.wav"
}

INITIAL_VOICE_LINES = {
    "default": ["Hello there! How can I assist you today?"],
    "Trump": [
        "This is Trump speaking, the best voice, believe me!",
        "Welcome, you're going to love this, it's fantastic!",
        "Make America great again!!!"
    ],
    "Vitalik": [
        "The ticker is ETH",
        "Ultrasound money",
    ]
}

# WebSocket manager class
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        """Add a new connection and assign a unique client ID."""
        await websocket.accept()
        client_id = str(uuid.uuid4())  # Generate a unique client ID
        self.active_connections[client_id] = websocket
        return client_id

    def disconnect(self, client_id: str):
        """Remove a connection by client ID."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_response(self, client_id: str, response: dict):
        """Send a combined text and audio response to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(response))

# Create a WebSocket connection manager
manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    client_personalities[client_id] = "default"  # Default personality
    print(f"Client connected: {client_id}")

    try:
        while True:
            # Receive a message from the client
            message = await websocket.receive_text()
            message_data = json.loads(message)

            if message_data["type"] == "personality":
                # Handle personality selection
                personality = message_data["data"]
                if personality in PERSONALITY_MAP:
                    client_personalities[client_id] = personality
                    print(f"Client {client_id} selected personality: {personality}")
                    await manager.send_response(client_id, {
                        "message_type": "personality_update",
                        "status": "personality updated"
                    })
                else:
                    print(f"Invalid personality: {personality}")
                    await manager.send_response(client_id, {
                        "message_type": "error",
                        "error": "Invalid personality"
                    })
                continue

            elif message_data["type"] in ["start_vocal", "transcript"]:
                # Handle start vocal or transcript
                personality = client_personalities.get(client_id, "default")
                speaker_wav_path = PERSONALITY_MAP.get(personality, None)

                if message_data["type"] == "start_vocal":
                    # Use a random initial line
                    voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
                    text = random.choice(voice_lines)
                else:
                    # Generate AI response for transcript
                    transcript = message_data["data"]
                    inputs = tokenizer(transcript, return_tensors="pt").to(device)
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=25,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                print(f"Generated text for {client_id}: {text}")
                output_path = f"response_{client_id}.wav"
                audio_base64 = generate_tts_response(text, speaker_wav_path, output_path)

                # Send response to the client
                await manager.send_response(client_id, {
                    "message_type": "start_vocal_response" if message_data["type"] == "start_vocal" else "transcript_response",
                    "text": text,
                    "audio_base64": audio_base64
                })
                print(f"Response sent to {client_id}")

    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        manager.disconnect(client_id)
        client_personalities.pop(client_id, None)
    except Exception as e:
        print(f"Error for {client_id}: {e}")
        manager.disconnect(client_id)
        client_personalities.pop(client_id, None)


def generate_tts_response(text: str, speaker_wav_path: str, output_path: str) -> str:
    """
    Generate TTS audio for the given text and return the base64-encoded audio.
    """
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language="en",
        file_path=output_path
    )
    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")
