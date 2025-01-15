from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import json
import uuid
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHAT_SERVICE_URL = "http://localhost:8001"
TTS_SERVICE_URL = "http://localhost:8002"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_personalities: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        self.client_personalities[client_id] = "default"
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_personalities:
            del self.client_personalities[client_id]

    async def send_response(self, client_id: str, response: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(response))

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    print(f"Client connected: {client_id}")

    async with aiohttp.ClientSession() as session:
        try:
            while True:
                message = await websocket.receive_text()
                message_data = json.loads(message)

                if message_data["type"] == "personality":
                    # Forward personality selection to chat service
                    async with session.post(
                        f"{CHAT_SERVICE_URL}/update_personality",
                        json={"client_id": client_id, "personality": message_data["data"]}
                    ) as response:
                        result = await response.json()
                        await manager.send_response(client_id, {
                            "message_type": "personality_update",
                            "status": result["status"]
                        })

                elif message_data["type"] in ["start_vocal", "transcript"]:
                    # First, get response from chat service
                    async with session.post(
                        f"{CHAT_SERVICE_URL}/generate_response",
                        json={
                            "client_id": client_id,
                            "type": message_data["type"],
                            "data": message_data.get("data", "")
                        }
                    ) as response:
                        chat_result = await response.json()
                        
                    # Then, get audio from TTS service
                    async with session.post(
                        f"{TTS_SERVICE_URL}/generate_audio",
                        json={
                            "client_id": client_id,
                            "text": chat_result["text"],
                            "personality": manager.client_personalities[client_id],
                            "message_type": message_data["type"]
                        }
                    ) as response:
                        tts_result = await response.json()

                    # Send combined response back to client
                    await manager.send_response(client_id, {
                        "message_type": f"{message_data['type']}_response",
                        "text": chat_result["text"],
                        "audio_base64": tts_result["audio_base64"]
                    })

        except WebSocketDisconnect:
            print(f"Client disconnected: {client_id}")
            manager.disconnect(client_id)
        except Exception as e:
            print(f"Error for {client_id}: {e}")
            manager.disconnect(client_id)