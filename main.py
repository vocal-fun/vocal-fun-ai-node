from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS
import base64
import json
import uuid

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
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load TTS model for speech synthesis
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Store connected clients
clients = {}

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
    print(f"Client connected: {client_id}")

    try:
        while True:
            # Receive transcript text from the client
            transcript = await websocket.receive_text()
            print(f"Received from {client_id}: {transcript}")

            # Generate AI response
            inputs = tokenizer(transcript, return_tensors="pt").to(model.device)
            outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
            ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated response for {client_id}: {ai_response}")

            # ai_response = transcript

            # Convert AI response to speech (TTS)
            audio_path = f"response_{client_id}.wav"
            tts.tts_to_file(text=ai_response, speaker_wav="vitalik.wav", language="en", file_path=audio_path)

            # Read the audio file and encode it as base64
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Create a combined response
            response = {
                "text": ai_response,
                "audio_base64": audio_base64
            }

            # Send the response to the client
            await manager.send_response(client_id, response)
            print(f"Response sent to {client_id}")

    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error for {client_id}: {e}")
        manager.disconnect(client_id)
