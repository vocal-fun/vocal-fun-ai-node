from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import json
import uuid
import time
import asyncio
from typing import Dict, Optional

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
TTS_SERVICE_URL = "ws://localhost:8002/tts_stream"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_personalities: Dict[str, str] = {}
        self.tts_sessions: Dict[str, aiohttp.ClientWebSocketResponse] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        self.client_personalities[client_id] = "default"
        return client_id

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_personalities:
            del self.client_personalities[client_id]
        # Clean up TTS session if exists
        await self.close_tts_session(client_id)

    async def close_tts_session(self, client_id: str):
        if client_id in self.tts_sessions:
            try:
                await self.tts_sessions[client_id].close()
            except Exception as e:
                print(f"Error closing TTS session for {client_id}: {e}")
            finally:
                del self.tts_sessions[client_id]

    async def send_response(self, client_id: str, response: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(response))

manager = ConnectionManager()

async def handle_tts_stream(client_id: str, text: str, personality: str, session: aiohttp.ClientSession) -> None:
    try:
        # Close any existing TTS session for this client
        await manager.close_tts_session(client_id)
        
        # Establish new TTS WebSocket connection
        async with session.ws_connect(TTS_SERVICE_URL, timeout=30) as ws:
            manager.tts_sessions[client_id] = ws
            
            # Start TTS stream
            await ws.send_json({
                "text": text,
                "personality": personality
            })

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Forward stream data to client
                    await manager.send_response(client_id, {
                        "message_type": "tts_stream",
                        "stream_data": data
                    })
                    
                    # If this is the last chunk, break
                    if data.get("type") == "stream_end":
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"TTS WebSocket error for {client_id}: {ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

    except asyncio.TimeoutError:
        print(f"TTS stream timeout for client {client_id}")
        await manager.send_response(client_id, {
            "message_type": "tts_stream",
            "stream_data": {"type": "error", "error": "TTS stream timeout"}
        })
    except Exception as e:
        print(f"Error in TTS stream for {client_id}: {e}")
        await manager.send_response(client_id, {
            "message_type": "tts_stream",
            "stream_data": {"type": "error", "error": str(e)}
        })
    finally:
        await manager.close_tts_session(client_id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    print(f"Client connected: {client_id}")

    async with aiohttp.ClientSession() as session:
        try:
            while True:
                message = await websocket.receive_text()
                message_data = json.loads(message)
                start_time = time.time()

                if message_data["type"] == "personality":
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
                    # Chat service timing
                    chat_start = time.time()
                    async with session.post(
                        f"{CHAT_SERVICE_URL}/generate_response",
                        json={
                            "client_id": client_id,
                            "type": message_data["type"],
                            "data": message_data.get("data", "")
                        }
                    ) as response:
                        chat_result = await response.json()
                    chat_latency = (time.time() - chat_start) * 1000
                    print(f"\nRequest Latency Breakdown for {client_id}:")
                    print(f"Chat Service: {chat_latency:.2f}ms")

                    # Send text response immediately
                    await manager.send_response(client_id, {
                        "message_type": f"{message_data['type']}_text",
                        "text": chat_result["text"]
                    })

                    # Handle TTS streaming
                    await handle_tts_stream(
                        client_id,
                        chat_result["text"],
                        manager.client_personalities[client_id],
                        session
                    )

        except WebSocketDisconnect:
            print(f"Client disconnected: {client_id}")
            await manager.disconnect(client_id)
        except Exception as e:
            print(f"Error for {client_id}: {e}")
            await manager.disconnect(client_id)